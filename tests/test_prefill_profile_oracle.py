#!/usr/bin/env python3
"""CPU-only tests for the task 24 prefill kernel-share oracle (W4 gate).

Covers the per-role classification and the pre-registered gather-share gate
(proceed, both stop floors, two inconclusive paths, min-across-ranks), the
prefill probe's fail-closed preflight/stale-trace handling and trace-derived
role counts, and the shared guarded-window runner's probe/tag wiring.
"""
from __future__ import annotations

import gzip
import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_prefill_kernel_share as pf  # noqa: E402


def kernel(name: str, dur: float, ts: float = 0.0, **args) -> dict:
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "args": args}


def write_trace(path: Path, kernels: list[dict], gz: bool = False) -> Path:
    if gz:
        path = path.with_suffix(path.suffix + ".gz")
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump({"traceEvents": kernels}, fh)
    else:
        path.write_text(json.dumps({"traceEvents": kernels}))
    return path


def prefill_trace(
    path: Path,
    gather: float,
    gateup: float,
    down: float,
    gemm: float,
    fused: float = 0.0,
    gz: bool = False,
    gather_calls: int = 1,
    gateup_calls: int = 1,
    down_calls: int = 1,
    fused_calls: int = 1,
) -> Path:
    events = [
        kernel("fm_gather_kernel", gather / gather_calls, float(i),
               **{"registers per thread": 96})
        for i in range(gather_calls)
    ]
    events += [
        kernel("fm_gateup_kernel", gateup / gateup_calls, 10.0),
        kernel("fm_down_kernel", down / down_calls, 20.0),
        kernel("cutlass::gemm", gemm, 30.0),
    ]
    events += [
        kernel("fm_gateup_kernel", gateup / gateup_calls, 11.0 + i) for i in range(1, gateup_calls)
    ]
    events += [
        kernel("fm_down_kernel", down / down_calls, 21.0 + i) for i in range(1, down_calls)
    ]
    if fused:
        events += [
            kernel("exl3_moe_kernel<4,256,1>", fused / fused_calls, 40.0 + i)
            for i in range(fused_calls)
        ]
    return write_trace(path, events, gz=gz)


def test_role_classification() -> None:
    assert pf.role_of("fm_gather_kernel") == "gather"
    assert pf.role_of("void fm_gateup_kernel<...>") == "gateup"
    assert pf.role_of("fm_down_kernel") == "down"
    # A generic "gather"/"down" outside the grouped family must not match.
    assert pf.role_of("tensor_gather_kernel") is None
    assert pf.role_of("cutlass::gemm") is None


def test_proceed_when_gather_share_clears_both_floors(tmp_path: Path) -> None:
    trace = prefill_trace(tmp_path / "rank0.pt.trace.json", gather=300, gateup=1000, down=1200, gemm=5000)
    report = pf.audit_prefill([trace])
    rank = report["ranks"]["rank0"]
    assert rank["grouped"]["gather"]["share_of_e3"] == round(300 / 2500, 6)
    assert rank["grouped"]["gather"]["share_of_total"] == round(300 / 7500, 6)
    assert report["decision"]["verdict"] == pf.VERDICT_PROCEED
    assert report["decision"]["e2e_ceiling_if_gather_free"] == round(300 / 7500, 6)


def test_stop_below_e3_floor(tmp_path: Path) -> None:
    trace = prefill_trace(tmp_path / "rank0.pt.trace.json", gather=100, gateup=1000, down=1200, gemm=5000)
    report = pf.audit_prefill([trace])
    assert report["decision"]["verdict"] == pf.VERDICT_STOP
    assert report["decision"]["min_gather_share_of_e3"] < 0.10


def test_stop_below_total_floor_even_with_high_e3_share(tmp_path: Path) -> None:
    """A gather-dominated E3 block that is a rounding error of the whole prefill."""
    trace = prefill_trace(tmp_path / "rank0.pt.trace.json", gather=800, gateup=100, down=100, gemm=100000)
    report = pf.audit_prefill([trace])
    decision = report["decision"]
    assert decision["min_gather_share_of_e3"] > 0.10
    assert decision["min_gather_share_of_total"] < 0.03
    assert decision["verdict"] == pf.VERDICT_STOP


def test_min_across_ranks_is_conservative(tmp_path: Path) -> None:
    good = prefill_trace(tmp_path / "rank0.pt.trace.json", gather=300, gateup=1000, down=1200, gemm=5000)
    bad = prefill_trace(tmp_path / "rank1.pt.trace.json", gather=100, gateup=1000, down=1200, gemm=5000)
    report = pf.audit_prefill([good, bad])
    assert report["decision"]["verdict"] == pf.VERDICT_STOP
    assert report["decision"]["gather_share_of_e3_by_rank"]["rank0"] == round(300 / 2500, 6)


def test_inconclusive_without_grouped_capture(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "rank0.pt.trace.json",
        [kernel("cutlass::gemm", 1000.0), kernel("ncclAllReduce", 500.0)],
    )
    report = pf.audit_prefill([trace])
    assert report["decision"]["verdict"] == pf.VERDICT_NO_GROUPED


def test_inconclusive_when_decode_dominates(tmp_path: Path) -> None:
    trace = prefill_trace(
        tmp_path / "rank0.pt.trace.json", gather=300, gateup=1000, down=1200, gemm=5000, fused=50000
    )
    report = pf.audit_prefill([trace])
    assert report["decision"]["verdict"] == pf.VERDICT_DECODE


def test_inconclusive_when_fused_launches_outnumber_gather(tmp_path: Path) -> None:
    """The structural guard catches what the time ceiling alone would pass."""
    trace = prefill_trace(
        tmp_path / "rank0.pt.trace.json", gather=300, gateup=1000, down=1200, gemm=5000,
        fused=300, fused_calls=3,
    )
    report = pf.audit_prefill([trace])
    decision = report["decision"]
    assert decision["fused_call_ratio_by_rank"]["rank0"] > 2.0
    assert decision["verdict"] == pf.VERDICT_DECODE


def test_measured_60k_prefill_shape_decides_not_inconclusive(tmp_path: Path) -> None:
    """Exact both-rank replay of the 2026-09-09 60k capture.

    Real per-rank kernel times and launch counts from the first cluster window.
    The fused ``exl3_moe`` kernel runs during prefill here (1470 launches beside
    1428 grouped per rank, 12.7% of kernel time), so the capture must decide,
    not read as decode-dominated.
    """
    # (gather, gateup, down, fused, kernel_total) in us, all over 1428 grouped
    # and 1470 fused launches.
    ranks = {
        "rank0": (818056.0, 5224835.0, 3384937.0, 4986065.0, 38974354.0),
        "rank1": (816751.0, 5245374.0, 3407585.0, 4945779.0, 38994804.0),
    }
    traces = []
    for rank, (gather, gateup, down, fused, total) in ranks.items():
        gemm = total - (gather + gateup + down + fused)
        traces.append(
            prefill_trace(
                tmp_path / f"{rank}.pt.trace.json",
                gather=gather, gateup=gateup, down=down, gemm=gemm, fused=fused,
                gather_calls=1428, gateup_calls=1428, down_calls=1428, fused_calls=1470,
            )
        )
    decision = pf.audit_prefill(traces)["decision"]
    assert decision["verdict"] == pf.VERDICT_STOP
    for rank, (gather, gateup, down, fused, total) in ranks.items():
        assert abs(decision["fused_call_ratio_by_rank"][rank] - 1470 / 1428) < 1e-9
        assert decision["decode_share_by_rank"][rank] < 0.30
        assert abs(decision["gather_share_of_e3_by_rank"][rank]
                   - gather / (gather + gateup + down)) < 1e-6
        assert abs(decision["gather_share_of_total_by_rank"][rank] - gather / total) < 1e-6
    assert decision["min_gather_share_of_e3"] < 0.10
    assert decision["min_gather_share_of_total"] < 0.03


def test_audit_cli_exit_codes(tmp_path: Path) -> None:
    proceed = prefill_trace(tmp_path / "r0.pt.trace.json", 300, 1000, 1200, 5000)
    assert pf.main(["--trace", str(proceed), "--json-out", str(tmp_path / "out.json")]) == 0
    assert json.loads((tmp_path / "out.json").read_text())["decision"]["verdict"] == pf.VERDICT_PROCEED
    stop = prefill_trace(tmp_path / "r1.pt.trace.json", 100, 1000, 1200, 5000)
    assert pf.main(["--trace", str(stop)]) == 0
    empty = write_trace(tmp_path / "r2.pt.trace.json", [kernel("cutlass::gemm", 1.0)])
    assert pf.main(["--trace", str(empty)]) == 3


def test_probe_count_grouped_and_roles(tmp_path: Path) -> None:
    import probe_prefill_profile as probe

    prefill_trace(tmp_path / "head.pt.trace.json", 300, 1000, 1200, 5000, fused=10, gz=True)
    captured = probe.count_grouped(tmp_path)
    head = captured["head"]
    assert head["gather_calls"] == 1
    assert head["gateup_calls"] == 1
    assert head["down_calls"] == 1
    assert head["fused_moe_calls"] == 1
    assert head["fused_moe_share"] == round(10 / 7510, 6)


def test_probe_prompt_nonce_leads() -> None:
    """A shared suffix could still hit the APC prefix; the nonce must lead."""
    import probe_prefill_profile as probe

    prompt = probe.make_prompt("abcd1234", 2)
    assert prompt.startswith("[run abcd1234] ")
    assert prompt.count(probe.PROMPT_SEED) == 2


def test_probe_refuses_stale_traces(tmp_path: Path) -> None:
    import probe_prefill_profile as probe

    write_trace(tmp_path / "old.pt.trace.json", [kernel("fm_gather_kernel", 1.0)], gz=True)
    rc = probe.main(["--out", str(tmp_path / "r.json"), "--trace-dir", str(tmp_path)])
    assert rc == 2


def test_probe_refuses_missing_trace_dir(tmp_path: Path) -> None:
    import probe_prefill_profile as probe

    rc = probe.main(["--out", str(tmp_path / "r.json"), "--trace-dir", str(tmp_path / "nope")])
    assert rc == 2


class _Handler(http.server.BaseHTTPRequestHandler):
    armed = False

    def do_GET(self) -> None:  # noqa: N802
        code = 405 if self.armed and self.path == "/start_profile" else 404
        self.send_response(code)
        self.end_headers()

    def log_message(self, *_args) -> None:  # noqa: ANN002
        return


def _serve(armed: bool) -> tuple[str, http.server.ThreadingHTTPServer]:
    _Handler.armed = armed
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


def test_probe_refuses_unarmed_boot(tmp_path: Path) -> None:
    import probe_prefill_profile as probe

    base, server = _serve(armed=False)
    try:
        rc = probe.main(["--base", base, "--out", str(tmp_path / "r.json"),
                         "--trace-dir", str(tmp_path)])
        assert rc == 2
    finally:
        server.shutdown()


def test_runner_probe_argv_and_tag_wiring(tmp_path: Path) -> None:
    import run_decode_profile_window as win

    argv = win.probe_argv(
        Path("/x/probe_prefill_profile.py"),
        ["--prompt-repeats", "3300"],
        tmp_path / "r.json",
        tmp_path / "traces",
    )
    assert argv[1] == "/x/probe_prefill_profile.py"
    assert argv[2:4] == ["--prompt-repeats", "3300"]
    assert argv[4:6] == ["--warmup", "--out"]
    assert argv[-2] == "--trace-dir"

    # --probe/--tag are applied before any phase runs; a missing probe fails
    # closed in preflight with the tag preserved in the receipt.
    state = tmp_path / "state.json"
    rc = win.main(["--from", "preflight", "--to", "preflight", "--tag", "task24",
                   "--probe", str(tmp_path / "missing.py"), "--receipt", str(state)])
    assert rc == 1
    assert win._TAG == "task24"
    assert win._PROBE == tmp_path / "missing.py"
    assert json.loads(state.read_text())["schema"] == 2


def test_runner_resume_keeps_probe_selection(tmp_path: Path) -> None:
    """A resumed window must not silently fall back to the decode probe."""
    import run_decode_profile_window as win

    state = tmp_path / "state.json"
    rc = win.main(["--from", "preflight", "--to", "preflight", "--tag", "task24",
                   "--probe", str(tmp_path / "missing.py"),
                   "--probe-args", "--prompt-repeats 3300",
                   "--receipt", str(state)])
    assert rc == 1
    recorded = json.loads(state.read_text())["probe_selection"]
    assert recorded == {"probe": str(tmp_path / "missing.py"),
                        "probe_args": ["--prompt-repeats", "3300"], "tag": "task24"}

    rc = win.main(["--from", "preflight", "--to", "preflight", "--state", str(state)])
    assert rc == 1
    assert win._TAG == "task24"
    assert win._PROBE == tmp_path / "missing.py"
    assert win._PROBE_ARG == ["--prompt-repeats", "3300"]

    conflict = win.main(["--from", "preflight", "--to", "preflight",
                         "--state", str(state), "--tag", "other"])
    assert conflict == 2


def test_runner_records_absolute_probe_across_cwd(tmp_path: Path) -> None:
    """A relative --probe must be frozen absolute or a resume can select another file."""
    import run_decode_profile_window as win

    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "probe.py").write_text("# probe\n")
    state = tmp_path / "state.json"
    original = os.getcwd()
    try:
        os.chdir(workdir)
        rc = win.main(["--from", "preflight", "--to", "preflight", "--probe", "probe.py",
                       "--receipt", str(state)])
        assert rc == 1
        assert json.loads(state.read_text())["probe_selection"]["probe"] == str(
            (workdir / "probe.py").resolve()
        )

        os.chdir(tmp_path)
        rc = win.main(["--from", "preflight", "--to", "preflight", "--state", str(state)])
        assert rc == 1
        assert win._PROBE == (workdir / "probe.py").resolve()

        # A relative override that resolves elsewhere must be rejected, not
        # silently swapped in after the window was armed.
        (tmp_path / "probe.py").write_text("# other\n")
        rc = win.main(["--from", "preflight", "--to", "preflight", "--state", str(state),
                       "--probe", "probe.py"])
        assert rc == 2
    finally:
        os.chdir(original)


def test_inconclusive_when_a_grouped_role_is_missing(tmp_path: Path) -> None:
    """A lone gather kernel would otherwise read as a 100% share on no evidence."""
    for omit in ("gather", "gateup", "down"):
        events = [
            kernel("fm_gather_kernel", 300.0),
            kernel("fm_gateup_kernel", 1000.0),
            kernel("fm_down_kernel", 1200.0),
            kernel("cutlass::gemm", 5000.0),
        ]
        trace = write_trace(
            tmp_path / f"omit_{omit}.pt.trace.json",
            [event for event in events if omit not in event["name"]],
        )
        report = pf.audit_prefill([trace])
        assert report["decision"]["verdict"] == pf.VERDICT_NO_GROUPED, omit


def test_inconclusive_when_a_grouped_role_has_zero_time(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "zero.pt.trace.json",
        [
            kernel("fm_gather_kernel", 0.0),
            kernel("fm_gateup_kernel", 1000.0),
            kernel("fm_down_kernel", 1200.0),
            kernel("cutlass::gemm", 5000.0),
        ],
    )
    report = pf.audit_prefill([trace])
    assert report["decision"]["verdict"] == pf.VERDICT_NO_GROUPED


def _grouped_trace_kernels(
    per_role: int = 120, fused_calls: int = 0, fused_dur: float = 0.5
) -> list[dict]:
    events: list[dict] = []
    for _ in range(per_role):
        events.append(kernel("fm_gather_kernel", 1.0))
        events.append(kernel("fm_gateup_kernel", 3.0))
        events.append(kernel("fm_down_kernel", 3.0))
    for _ in range(fused_calls):
        events.append(kernel("exl3_moe_kernel<4,256,1>", fused_dur))
    return events


class _StreamHandler(http.server.BaseHTTPRequestHandler):
    """Minimal vLLM-compatible streaming server for the prefill probe."""

    mode = "ok"  # ok | no_usage | incomplete | cached | fused_heavy
    trace_dir: Path | None = None

    def _send(self, code: int, body: bytes = b"", ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/start_profile":
            self._send(405)
        elif self.path == "/metrics":
            self._send(200, b"", "text/plain")
        else:
            self._send(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        if self.path == "/reset_prefix_cache":
            self._send(200, b'{"success": true}')
        elif self.path == "/start_profile":
            self._send(200)
        elif self.path == "/stop_profile":
            if self.trace_dir is not None:
                kernels = (
                    _grouped_trace_kernels(fused_calls=360)
                    if self.mode == "fused_heavy"
                    else _grouped_trace_kernels()
                )
                write_trace(self.trace_dir / "head.pt.trace.json", kernels)
            self._send(200)
        elif self.path == "/v1/chat/completions":
            chunks = [b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}\n\n']
            if self.mode in ("ok", "cached", "fused_heavy"):
                cached = 5000 if self.mode == "cached" else 0
                usage = (
                    'data: {"choices":[],"usage":{"prompt_tokens":10000,'
                    f'"completion_tokens":1,"prompt_tokens_details":{{"cached_tokens":{cached}}}}}}}\n\n'
                )
                chunks.append(usage.encode())
                chunks.append(b"data: [DONE]\n\n")
            elif self.mode == "no_usage":
                chunks.append(b"data: [DONE]\n\n")
            # "incomplete": no usage, no [DONE], no finish_reason
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(chunk)
            self.wfile.flush()
        else:
            self._send(404)

    def log_message(self, *_args) -> None:  # noqa: ANN002
        return


def _serve_stream(trace_dir: Path, mode: str) -> tuple[str, http.server.ThreadingHTTPServer]:
    _StreamHandler.mode = mode
    _StreamHandler.trace_dir = trace_dir
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StreamHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


def test_probe_accepts_usage_only_final_chunk(tmp_path: Path) -> None:
    """vLLM sends usage in a chunk whose choices array is empty."""
    import probe_prefill_profile as probe

    traces = tmp_path / "traces"
    traces.mkdir()
    base, server = _serve_stream(traces, "ok")
    try:
        rc = probe.main(["--base", base, "--warmup", "--min-gather-calls", "1",
                         "--trace-dir", str(traces), "--out", str(tmp_path / "r.json")])
        assert rc == 0
        receipt = json.loads((tmp_path / "r.json").read_text())
        assert receipt["prompt_tokens"] == 10000
        assert receipt["cached_tokens"] == 0
        assert receipt["gather_floor"] == int(0.5 * 10000 / 3584 * 42)
        assert receipt["captured"]["head"]["gather_calls"] == 120
        # The ratio is recorded on success too, not only on the guard failure path.
        assert receipt["fused_call_ratio"] == 0.0
    finally:
        server.shutdown()


def test_probe_fails_without_reported_usage(tmp_path: Path) -> None:
    import probe_prefill_profile as probe

    traces = tmp_path / "traces"
    traces.mkdir()
    base, server = _serve_stream(traces, "no_usage")
    try:
        rc = probe.main(["--base", base, "--min-gather-calls", "1",
                         "--trace-dir", str(traces), "--out", str(tmp_path / "r.json")])
        assert rc == 3
        assert json.loads((tmp_path / "r.json").read_text())["prompt_tokens"] is None
    finally:
        server.shutdown()


def test_probe_fails_on_incomplete_stream(tmp_path: Path) -> None:
    import probe_prefill_profile as probe

    traces = tmp_path / "traces"
    traces.mkdir()
    base, server = _serve_stream(traces, "incomplete")
    try:
        rc = probe.main(["--base", base, "--min-gather-calls", "1",
                         "--trace-dir", str(traces), "--out", str(tmp_path / "r.json")])
        assert rc == 3
        assert json.loads((tmp_path / "r.json").read_text())["request"]["completed"] is False
    finally:
        server.shutdown()


def test_probe_fails_when_prefill_was_cached(tmp_path: Path) -> None:
    import probe_prefill_profile as probe

    traces = tmp_path / "traces"
    traces.mkdir()
    base, server = _serve_stream(traces, "cached")
    try:
        rc = probe.main(["--base", base, "--min-gather-calls", "1",
                         "--trace-dir", str(traces), "--out", str(tmp_path / "r.json")])
        assert rc == 3
    finally:
        server.shutdown()


def test_probe_fails_when_fused_launches_outnumber_grouped(tmp_path: Path) -> None:
    """A decode-heavy capture fails on launches even when its fused time share is low."""
    import probe_prefill_profile as probe

    traces = tmp_path / "traces"
    traces.mkdir()
    base, server = _serve_stream(traces, "fused_heavy")
    try:
        out = tmp_path / "r.json"
        rc = probe.main(["--base", base, "--warmup", "--min-gather-calls", "1",
                         "--trace-dir", str(traces), "--out", str(out)])
    finally:
        server.shutdown()
    assert rc == 3
    receipt = json.loads(out.read_text())
    assert receipt["fused_call_ratio"] == 3.0
    assert receipt["captured"]["head"]["fused_moe_share"] < 0.30


def _run_profiling_helper(dropin: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GLM53_PROFILING_DROPIN": str(dropin)}
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / "enable-gpu-profiling.sh"), *args],
        env=env, capture_output=True, text=True, check=False,
    )


def test_profiling_helper_refuses_foreign_dropin(tmp_path: Path) -> None:
    """The runbook claims no existing file is edited; an unowned file must stop it."""
    dropin = tmp_path / "99-nvidia-profiling.conf"
    dropin.write_text("options nvidia NVreg_SomeOther=1\n")
    refused = _run_profiling_helper(dropin)
    assert refused.returncode == 2
    assert dropin.read_text() == "options nvidia NVreg_SomeOther=1\n"

    forced = _run_profiling_helper(dropin, "--force")
    assert forced.returncode == 0
    body = dropin.read_text()
    assert "managed by glm53" in body
    assert "NVreg_RestrictProfilingToAdminUsers=0" in body
    assert list(tmp_path.glob("*.bak-*")), "a forced replace must keep a backup"

    assert _run_profiling_helper(dropin, "--revert").returncode == 0
    assert not dropin.exists()


def test_profiling_helper_revert_refuses_foreign_dropin(tmp_path: Path) -> None:
    dropin = tmp_path / "d.conf"
    dropin.write_text("options nvidia NVreg_SomeOther=1\n")
    assert _run_profiling_helper(dropin, "--revert").returncode == 2
    assert dropin.exists()


def test_profiling_helper_apply_is_idempotent(tmp_path: Path) -> None:
    dropin = tmp_path / "d.conf"
    for _ in range(2):
        proc = _run_profiling_helper(dropin)
        assert proc.returncode == 0, proc.stderr
    assert dropin.read_text().count("NVreg_RestrictProfilingToAdminUsers=0") == 1
    check = _run_profiling_helper(dropin, "--check")
    assert check.returncode == 0
    assert "managed" in check.stdout


