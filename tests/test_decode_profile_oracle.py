#!/usr/bin/env python3
"""CPU-only tests for the task 29/31 decode-profile oracle.

Covers the chrome-trace auditor (shares, families, graph grouping, thresholds,
fail-closed parsing), the probe's profiler-mounted preflight, and the launcher
contract (both inner scripts carry the argv, the knobs are hash-neutral and
validated).
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

import audit_decode_kernel_share as audit  # noqa: E402

START = ROOT / "start.sh"
PROD_START = (ROOT / "local" / "prod-start.sh").read_text()


def kernel(name: str, dur: float, ts: float = 0.0, **args) -> dict:
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "args": args}


def annotation(name: str, ts: float, dur: float) -> dict:
    return {"ph": "X", "cat": "gpu_user_annotation", "name": name, "ts": ts, "dur": dur, "args": {}}


def write_trace(path: Path, kernels: list[dict], gz: bool = False) -> Path:
    if gz:
        path = path.with_suffix(path.suffix + ".gz")
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump({"traceEvents": kernels}, fh)
    else:
        path.write_text(json.dumps({"traceEvents": kernels}))
    return path


def test_family_classification() -> None:
    cases = {
        "void exl3_moe_kernel<k4>(...)": "fused_moe",
        "fm_gateup_kernel": "grouped_fat_moe",
        "flashinfer_sparse_mla_decode": "sparse_mla",
        "fused_recurrent_kda_fwd_kernel": "kda",
        "cutlass::Kernel2<cutlass_80_simt_sgemm>": "gemm",
        "some_unrelated_elementwise_kernel": "other",
    }
    for name, want in cases.items():
        assert audit.classify(name) == want, (name, audit.classify(name), want)


def test_shares_and_stop_below_floor(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "rank0.pt.trace.json",
        [
            kernel("exl3_moe_kernel<k4>", 10.0, **{"graph id": 1, "est. achieved occupancy %": 60}),
            kernel("cutlass::gemm", 990.0, **{"graph id": 1}),
        ],
    )
    report = audit.audit([trace], moe_share_floor=0.05, occ_floor=50.0)
    assert abs(report["decision"]["task29"]["fused_moe_share"] - 0.01) < 1e-9
    assert report["decision"]["task29"]["verdict"] == "STOP_SHARE_BELOW_FLOOR"
    assert report["families"]["fused_moe"]["total_us"] == 10.0
    assert report["totals"]["distinct_graphs"] == 1


def test_gap_candidate_and_no_gap(tmp_path: Path) -> None:
    low = write_trace(
        tmp_path / "low.pt.trace.json",
        [
            kernel("exl3_moe_kernel<k4>", 400.0, **{"est. achieved occupancy %": 20}),
            kernel("cutlass::gemm", 600.0),
        ],
    )
    report = audit.audit([low], moe_share_floor=0.05, occ_floor=50.0)
    assert report["decision"]["task29"]["verdict"] == "GAP_CANDIDATE_UNMEASURED"
    assert report["decision"]["task29"]["est_occupancy_pct_min"] == 20

    high = write_trace(
        tmp_path / "high.pt.trace.json",
        [
            kernel("exl3_moe_kernel<k4>", 400.0, **{"est. achieved occupancy %": 80}),
            kernel("cutlass::gemm", 600.0),
        ],
    )
    report = audit.audit([high], moe_share_floor=0.05, occ_floor=50.0)
    assert report["decision"]["task29"]["verdict"] == "STOP_NO_OCCUPANCY_GAP"


def test_sparse_mla_tactic_diversity(tmp_path: Path) -> None:
    single = write_trace(
        tmp_path / "single.pt.trace.json",
        [kernel("flashinfer_sparse_mla_decode_t256", 500.0), kernel("cutlass::gemm", 500.0)],
    )
    report = audit.audit([single], 0.05, 50.0)
    assert report["decision"]["task31"]["distinct_names"] == 1
    assert report["decision"]["task31"]["verdict"] == "STOP_NO_TACTIC_DIVERSITY"

    multi = write_trace(
        tmp_path / "multi.pt.trace.json",
        [
            kernel("flashinfer_sparse_mla_decode_t256", 300.0),
            kernel("flashinfer_sparse_mla_decode_t128", 300.0),
            kernel("cutlass::gemm", 400.0),
        ],
    )
    report = audit.audit([multi], 0.05, 50.0)
    assert report["decision"]["task31"]["distinct_names"] == 2
    assert report["decision"]["task31"]["verdict"] == "AUDIT_TACTIC_DIVERSITY"

    # a decode kernel plus its merge stage is one decode tactic, not two
    merge = write_trace(
        tmp_path / "merge.pt.trace.json",
        [
            kernel("flashinfer_sparse_mla_decode_dsv3_2_kernel", 300.0),
            kernel("flashinfer_sparse_mla_decode_dsv4_merge_kernel", 100.0),
            kernel("cutlass::gemm", 600.0),
        ],
    )
    report = audit.audit([merge], 0.05, 50.0)
    assert report["decision"]["task31"]["distinct_names"] == 2
    assert report["decision"]["task31"]["distinct_decode_names"] == 1
    assert report["decision"]["task31"]["verdict"] == "STOP_NO_TACTIC_DIVERSITY"


def test_graph_grouping_and_gzip(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "rank1.pt.trace.json",
        [
            kernel("exl3_moe_kernel<k4>", 100.0, **{"graph id": 7, "grid": [4, 1, 1]}),
            kernel("exl3_moe_kernel<k4>", 50.0, **{"graph id": 9, "grid": [8, 1, 1]}),
            kernel("cutlass::gemm", 25.0, **{"graph id": 7}),
        ],
        gz=True,
    )
    report = audit.audit([trace], 0.05, 50.0)
    graphs = {g["graph_id"]: g for g in report["graphs"]}
    assert graphs["7"]["total_us"] == 125.0
    assert graphs["9"]["total_us"] == 50.0
    fused = next(k for k in report["kernels"] if k["family"] == "fused_moe")
    assert fused["grid_volume"] == [4, 8]
    # graph ids are per-process, so they are namespaced by trace/rank
    assert fused["graph_ids"] == ["rank1:7", "rank1:9"]
    assert {g["graph_key"] for g in report["graphs"]} == {"rank1:7", "rank1:9"}


def test_rank_attribution_keeps_clock_domains_separate(tmp_path: Path) -> None:
    """Two ranks share graph-id numbers but not timelines; each trace must be
    joined on its own clock and reported separately."""
    for rank, ts, dur in (("rank0", 100.0, 40.0), ("rank1", 900.0, 20.0)):
        write_trace(
            tmp_path / f"{rank}.pt.trace.json",
            [
                annotation(f"execute_context_0({rank})_generation_4(12)", ts, 50.0),
                kernel("exl3_moe_kernel<k4>", dur, ts=ts + 5.0, **{"graph id": 1}),
            ],
        )
    report = audit.audit(sorted(tmp_path.glob("*.json")), 0.05, 50.0)
    assert {t["rank"] for t in report["traces"]} == {"rank0", "rank1"}
    assert {g["graph_key"] for g in report["graphs"]} == {"rank0:1", "rank1:1"}
    # both kernels land in the same realized-T bucket, summed across ranks
    entry = report["generations"]["execute_context_0(rank0)_generation_4(12)"]
    assert entry["tokens"] == 12 and entry["kernel_us"] == 40.0


def test_fail_closed_parsing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert audit.main(["--trace-dir", str(empty)]) == 2

    no_kernels = tmp_path / "nokern.pt.trace.json"
    no_kernels.write_text(json.dumps({"traceEvents": [{"cat": "cpu_op", "name": "x"}]}))
    assert audit.main(["--trace-dir", str(tmp_path)]) == 2


def test_truncated_trace_is_rejected(tmp_path: Path) -> None:
    truncated = tmp_path / "cut.pt.trace.json"
    truncated.write_text('{"traceEvents": [{"cat": "kernel", "name": "k", "dur": 1.0}')
    # a truncated trace must never silently yield a prefix into a verdict
    assert audit.main(["--trace", str(truncated)]) == 2
    assert list(audit.iter_trace_events(truncated, strict=False))  # diagnostic path


def test_cli_writes_json(tmp_path: Path) -> None:
    write_trace(
        tmp_path / "rank0.pt.trace.json",
        [kernel("exl3_moe_kernel<k4>", 100.0, **{"est. achieved occupancy %": 70}),
         kernel("cutlass::gemm", 900.0)],
    )
    out = tmp_path / "report.json"
    assert audit.main(["--trace-dir", str(tmp_path), "--json-out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["schema"] == 3
    assert payload["decision"]["task29"]["fused_moe_share"] == 0.1


def test_streaming_parser_handles_pretty_printed_trace(tmp_path: Path) -> None:
    path = tmp_path / "pretty.pt.trace.json"
    path.write_text(json.dumps({"traceEvents": [kernel("exl3_moe_kernel<k4>", 10.0),
                                                {"cat": "cpu_op", "name": "x"}]}, indent=1))
    events = list(audit.iter_trace_events(path))
    assert [e.get("cat") for e in events] == ["kernel", "cpu_op"]

    many = tmp_path / "many.pt.trace.json"
    many.write_text(json.dumps({"traceEvents": [kernel("k", 1.0) for _ in range(5000)]}, indent=1))
    assert len(list(audit.iter_trace_events(many))) == 5000

    truncated = tmp_path / "truncated.pt.trace.json"
    truncated.write_text('{"traceEvents": [{"cat": "kernel", "name": "k", "dur": 1.0}')
    try:
        list(audit.iter_trace_events(truncated))
    except audit.TruncatedTrace:
        pass
    else:
        raise AssertionError("a truncated trace must raise TruncatedTrace")
    # the diagnostic (non-strict) path still reads the complete prefix
    assert len(list(audit.iter_trace_events(truncated, strict=False))) == 1


def test_generation_attribution_and_roofline(tmp_path: Path) -> None:
    trace = write_trace(
        tmp_path / "rank0.pt.trace.json",
        [
            annotation("execute_context_0(0)_generation_4(12)", 100.0, 50.0),
            annotation("execute_context_0(0)_generation_4(20)", 200.0, 50.0),
            kernel("exl3_moe_kernel<k4>", 400.0, ts=110.0),
            kernel("cutlass::gemm", 10.0, ts=210.0),
        ],
    )
    report = audit.audit([trace], 0.05, 50.0)
    gens = report["generations"]
    assert gens["execute_context_0(0)_generation_4(12)"]["tokens"] == 12
    assert gens["execute_context_0(0)_generation_4(12)"]["fused_moe_share"] == 1.0
    assert gens["execute_context_0(0)_generation_4(20)"]["tokens"] == 20
    assert gens["execute_context_0(0)_generation_4(20)"]["fused_moe_share"] == 0.0

    roof = audit.Roofline(gbs=200.0, expert_bytes_per_rank=1_000_000, moe_layers=1,
                          target_t=(12,), tolerance=0.85)
    bounded = audit.audit([trace], 0.05, 50.0, roof)
    # uniform routing would need more than the node bandwidth here, but the
    # model is advisory: it must not manufacture a stop verdict
    assert bounded["roofline"]["by_t"]["12"]["implied_gbs_uniform"] > 200.0
    assert bounded["roofline"]["advisory"] is True
    assert bounded["decision"]["task29"]["verdict"] == "GAP_CANDIDATE_UNMEASURED"
    assert "STOP_ROOFLINE_BOUND" not in json.dumps(bounded)

    slow = write_trace(
        tmp_path / "slow.pt.trace.json",
        [
            annotation("execute_context_0(0)_generation_4(12)", 100.0, 50.0),
            kernel("exl3_moe_kernel<k4>", 10_000_000.0, ts=110.0),
        ],
    )
    unbounded = audit.audit([slow], 0.05, 50.0, roof)
    # even touching every expert slot cannot reach the roofline in 10 s
    assert unbounded["roofline"]["roofline_explanation_viable"] is False
    assert unbounded["decision"]["task29"]["verdict"] == "GAP_CANDIDATE_UNMEASURED"


def test_derived_occupancy_from_launch_geometry(tmp_path: Path) -> None:
    args = {
        "warps per SM": 16.0,
        "block": [512, 1, 1],
        "occupancy": {"blockLimitWarps": 3},
        "est. achieved occupancy %": 0,
    }
    assert audit._derived_occupancy(args) == round(100 * 16 / 48, 2)
    assert audit._derived_occupancy({}) is None

    trace = write_trace(
        tmp_path / "rank0.pt.trace.json",
        [kernel("exl3_moe_kernel<k4>", 400.0, **args), kernel("cutlass::gemm", 600.0)],
    )
    report = audit.audit([trace], 0.05, 50.0)
    assert report["decision"]["task29"]["derived_occupancy_pct_min"] == round(100 * 16 / 48, 2)
    assert report["decision"]["task29"]["verdict"] == "GAP_CANDIDATE_UNMEASURED"


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
    import probe_decode_profile as probe

    base, server = _serve(armed=False)
    try:
        rc = probe.main(["--base", base, "--out", str(tmp_path / "r.json"),
                         "--trace-dir", str(tmp_path)])
        assert rc == 2
    finally:
        server.shutdown()


def test_probe_unreachable_server(tmp_path: Path) -> None:
    import probe_decode_profile as probe

    rc = probe.main(["--base", "http://127.0.0.1:1", "--out", str(tmp_path / "r.json"),
                     "--trace-dir", str(tmp_path)])
    assert rc == 2


def test_window_gates_capture_acceptance_stdout() -> None:
    """The gates phase read acc.stdout without capturing it and crashed after
    acceptance had already passed 7/7; keep the capture in place."""
    src = (ROOT / "scripts" / "run_decode_profile_window.py").read_text()
    body = src[src.index("def phase_gates"):]
    body = body[: body.index("\ndef ", 1)]
    assert "capture_output=True" in body
    assert "acc.stdout" in body


def test_window_fails_closed_on_timers_and_traces() -> None:
    """Timer operations and trace collection must not pass silently."""
    src = (ROOT / "scripts" / "run_decode_profile_window.py").read_text()
    disarm = src[src.index("def phase_disarm"):]
    disarm = disarm[: disarm.index("\ndef ", 1)]
    assert "proc.returncode != 0" in disarm
    assert "timers still active after disarm" in disarm
    rearm = src[src.index("def phase_rearm"):]
    rearm = rearm[: rearm.index("\ndef ", 1)]
    assert "timers not active after rearm" in rearm
    profile = src[src.index("def phase_profile"):]
    profile = profile[: profile.index("\ndef require_traces")]
    assert "rsync of worker traces exited" in profile
    assert "require_traces(outdir" in profile
    assert "expected exactly one profiler trace" in src
    assert "too small to contain a decode window" in src


def test_window_interrupt_leaves_recoverable_state() -> None:
    """A killed window must still know the backup path, and must restore."""
    src = (ROOT / "scripts" / "run_decode_profile_window.py").read_text()
    preflight = src[src.index("def phase_preflight"):]
    preflight = preflight[: preflight.index("\ndef ", 1)]
    # backup coordinates are persisted before any later step can fail
    assert preflight.index('state.update({"backup"') < preflight.index("code, _ = curl")
    assert "save(state)" in preflight
    disarm = src[src.index("def phase_disarm"):]
    disarm = disarm[: disarm.index("\ndef ", 1)]
    # a partial disarm must still be recoverable, so intent is recorded first
    assert disarm.index('state["disarm_attempted"] = True') < disarm.index("systemctl")
    arm = src[src.index("def phase_arm"):]
    arm = arm[: arm.index("\ndef ", 1)]
    assert arm.index('state["armed_attempted"] = True') < arm.index("arm_env()")
    assert "atexit.register(emergency_restore)" in src
    assert "signal.signal(sig, _on_signal)" in src
    assert "except (Exception, KeyboardInterrupt) as exc" in src
    assert "if needs_env_restore(state):" in src
    assert "if not needs_timer_restore(state):" in src
    assert "recover_timers(state)" in src


def test_window_recovery_helpers(tmp_path: Path) -> None:
    import run_decode_profile_window as win

    assert win.needs_env_restore({}) is False
    assert win.needs_env_restore({"backup": "/x"}) is False
    assert win.needs_env_restore({"backup": "/x", "armed_attempted": True}) is True
    assert win.needs_timer_restore({}) is False
    assert win.needs_timer_restore({"disarm_attempted": True}) is True

    receipt = tmp_path / "state.json"
    win._RECEIPT = receipt
    try:
        win.save({"schema": 2, "backup": "/x"})
    finally:
        win._RECEIPT = None
    assert json.loads(receipt.read_text())["backup"] == "/x"
    assert not receipt.with_suffix(".json.tmp").exists()

    head = tmp_path / "traces" / "head"
    head.mkdir(parents=True)
    try:
        win.require_traces(tmp_path / "traces", "head")
    except RuntimeError as exc:
        assert "expected exactly one profiler trace" in str(exc)
    else:
        raise AssertionError("a rank without a trace must fail the window")

    (head / "dp0_rank0.pt.trace.json.gz").write_bytes(b"x" * 16)
    try:
        win.require_traces(tmp_path / "traces", "head")
    except RuntimeError as exc:
        assert "too small" in str(exc)
    else:
        raise AssertionError("an empty trace must fail the window")

    (head / "dp0_rank0.pt.trace.json.gz").write_bytes(b"x" * win.MIN_TRACE_BYTES)
    sizes = win.require_traces(tmp_path / "traces", "head")
    assert sizes == [{"name": "dp0_rank0.pt.trace.json.gz", "bytes": win.MIN_TRACE_BYTES}]


def test_window_recovers_timers_after_partial_disarm(tmp_path: Path) -> None:
    """Stopping one timer and failing the second must still re-enable monitoring,
    even though the profiler was never armed."""
    import run_decode_profile_window as win

    original_run, original_states = win.run, win.timer_states
    calls: list[list[str]] = []
    states = {win.TIMERS[0]: "inactive", win.TIMERS[1]: "active"}

    def fake_run(argv, timeout=600, check=True):  # noqa: ANN001
        calls.append(list(argv))
        if argv[:3] == ["systemctl", "--user", "stop"] and argv[3] == win.TIMERS[1]:
            return subprocess.CompletedProcess(argv, 1, "", "unit not found")
        return subprocess.CompletedProcess(argv, 0, "", "")

    win.run = fake_run
    win.timer_states = lambda: dict(states)
    win._RECEIPT = tmp_path / "state.json"
    try:
        state: dict = {}
        try:
            win.phase_disarm(state)
        except RuntimeError as exc:
            assert "systemctl stop" in str(exc)
        else:
            raise AssertionError("a partial disarm must fail the phase")
        assert state["disarm_attempted"] is True
        assert win.needs_timer_restore(state) is True
        assert win.needs_env_restore(state) is False
        assert json.loads((tmp_path / "state.json").read_text())["disarm_attempted"] is True

        def ok_run(argv, timeout=600, check=True):  # noqa: ANN001
            calls.append(list(argv))
            if argv[:3] == ["systemctl", "--user", "start"]:
                states[argv[3]] = "active"
            return subprocess.CompletedProcess(argv, 0, "", "")

        win.run = ok_run
        win.recover_timers(state)
        assert state["timer_restore"] == "ok"
        assert all(status == "active" for status in states.values())
        assert any(argv[:3] == ["systemctl", "--user", "start"] for argv in calls)
    finally:
        win.run, win.timer_states, win._RECEIPT = original_run, original_states, None


def test_probe_capture_gate_detects_autostop(tmp_path: Path) -> None:
    """The step floor must come from the captured trace, not from counters that
    keep counting after the profiler auto-stops on max_iterations."""
    import probe_decode_profile as probe

    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        probe.captured_engine_steps(empty, 42)
    except RuntimeError as exc:
        assert "no *.pt.trace.json.gz" in str(exc)
    else:
        raise AssertionError("a trace dir without traces must fail closed")

    short = tmp_path / "short"
    short.mkdir()
    write_trace(
        short / "rank0.pt.trace.json",
        [kernel("exl3_moe_kernel<k4>", 1.0) for _ in range(10)],
        gz=True,
    )
    got = probe.captured_engine_steps(short, 42)
    assert got["fused_moe_calls"] == 10
    assert got["engine_steps"] == round(10 / 42, 3)
    assert got["engine_steps"] < 60  # early auto-stop is visible here

    full = tmp_path / "full"
    full.mkdir()
    write_trace(
        full / "rank0.pt.trace.json",
        [kernel("exl3_moe_kernel<k4>", 1.0) for _ in range(60 * 42)],
        gz=True,
    )
    assert probe.captured_engine_steps(full, 42)["engine_steps"] == 60.0

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "rank0.pt.trace.json.gz").write_bytes(b"not gzip")
    try:
        probe.captured_engine_steps(corrupt, 42)
    except Exception:  # noqa: BLE001
        pass
    else:
        raise AssertionError("an unreadable trace must fail closed")


def test_probe_refuses_stale_traces(tmp_path: Path) -> None:
    """A leftover trace from an earlier run must never satisfy this run's floor."""
    import probe_decode_profile as probe

    old = write_trace(
        tmp_path / "old.pt.trace.json",
        [kernel("exl3_moe_kernel<k4>", 1.0) for _ in range(60 * 42)],
        gz=True,
    )
    assert [p.name for p in probe.stale_traces(tmp_path)] == [old.name]
    # captured_engine_steps alone would happily report the stale 60 steps
    assert probe.captured_engine_steps(tmp_path, 42)["engine_steps"] == 60.0

    original_request = probe.request

    def boom(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("a dirty trace dir must be rejected before any HTTP call")

    probe.request = boom
    try:
        rc = probe.main(["--base", "http://127.0.0.1:1", "--out", str(tmp_path / "r.json"),
                         "--trace-dir", str(tmp_path)])
    finally:
        probe.request = original_request
    assert rc == 2

    clean = tmp_path / "clean"
    clean.mkdir()
    assert probe.stale_traces(clean) == []


def test_probe_gates_and_labels_are_truthful() -> None:
    """The probe must require every stream past prefill, bracket the counters
    against /start_profile, and gate the capture on the trace itself."""
    src = (ROOT / "scripts" / "probe_decode_profile.py").read_text()
    assert "len(state) == args.seqs and all(v.get(\"first_ts\")" in src
    assert src.index("before = metrics()") > src.index("live, conc_samples = wait_concurrency")
    assert "engine_decode_steps_estimate" in src
    assert "drafts_delta_per_request_sum" in src
    assert '"counter_scope"' in src
    assert "probe FAILED: streams" in src
    assert "probe FAILED: /stop_profile failed" in src
    assert "--trace-dir is required outside --dry-run" in src
    assert "the trace holds only" in src
    runner = (ROOT / "scripts" / "run_decode_profile_window.py").read_text()
    assert '"--trace-dir", str(TRACE_HOST_DIR)' in runner


def test_launcher_argv_in_both_inner_scripts() -> None:
    src = START.read_text()
    assert src.count("ARGS+=(--profiler-config.profiler=torch)") == 2, (
        "the profiler argv must be added to both the head and worker inner scripts"
    )
    assert src.count('--profiler-config.torch_profiler_dir=${GLM53_PROFILE_TORCH_DIR}') == 2
    assert "GLM53_PROFILE_TORCH_DIR GLM53_PROFILE_MAX_ITERS; do" in src, (
        "the worker serve_env transport must forward both knobs"
    )
    assert src.count('-e GLM53_PROFILE_TORCH_DIR="${GLM53_PROFILE_TORCH_DIR:-}"') == 1


def test_profiler_knobs_are_not_shape_hashed() -> None:
    """ProfilerConfig.compute_hash is a constant, so the knob must never
    invalidate the JIT stamp (measured d751713988987e9331980363e24189ce both ways)."""
    src = START.read_text()
    assert "GLM53_PROFILE_TORCH_DIR" not in PROD_START, (
        "profiler knob must not enter the prod-start JIT shape hash"
    )
    assert "GLM53_PROFILE_MAX_ITERS" not in PROD_START
    # The only shape hash lives in local/prod-start.sh; the launcher must not
    # grow a second one that names these knobs.
    assert "GLM53_PROFILE_TORCH_DIR|" not in src and "|GLM53_PROFILE_TORCH_DIR" not in src


def _guard_source() -> str:
    source = START.read_text()
    # Mirror production ordering: the W41/W42 knob-defaults block always runs
    # before validate_numeric_config, so the strict-bool loop never sees an
    # UNSET knob. The profiler block must tolerate UNSET on its own (the
    # numeric-config harness slices without the defaults).
    d_begin = source.index("# LOCAL: W41/W42 knob defaults (begin)")
    d_end = source.index("# LOCAL: W41/W42 knob defaults (end)", d_begin)
    d_end += len("# LOCAL: W41/W42 knob defaults (end)")
    begin = source.index("# GLM53 numeric config guard (begin)")
    end = source.index("# GLM53 numeric config guard (end)", begin)
    end += len("# GLM53 numeric config guard (end)")
    return source[d_begin:d_end] + "\n" + source[begin:end]


def _validate_profile(extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
    script = (
        _guard_source()
        + "\nGPU_MEM_UTIL=0.85; MAX_MODEL_LEN=1000000; MAX_NUM_SEQS=4;"
        + " MAX_NUM_BATCHED_TOKENS=3584; SPEC_METHOD=dflash; DFLASH_TOKENS=7\n"
        + "validate_numeric_config || exit $?\nprintf OK\\n\n"
    )
    env = {**os.environ, "LC_ALL": "C"}
    env.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    env.update(extra)
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True, check=False, env=env)


def test_profile_knob_validation() -> None:
    ok = _validate_profile({"GLM53_PROFILE_TORCH_DIR": "/root/.cache/vllm/profiler"})
    assert ok.returncode == 0, ok.stderr
    unset = _validate_profile({})
    assert unset.returncode == 0, unset.stderr
    for bad in (
        "/tmp/prof",
        "/root/.cache/vllm/../escape",
        "/root/.cache/vllm/profiler;rm -rf /",
        "relative/path",
        "/root/.cache/vllm/",
    ):
        result = _validate_profile({"GLM53_PROFILE_TORCH_DIR": bad})
        assert result.returncode == 2, (bad, result.returncode, result.stdout)
    assert _validate_profile({"GLM53_PROFILE_MAX_ITERS": "0"}).returncode == 0
    assert _validate_profile({"GLM53_PROFILE_MAX_ITERS": "2000"}).returncode == 0
    for bad in ("-1", "1.5", "abc", "", "123456789"):
        result = _validate_profile({"GLM53_PROFILE_MAX_ITERS": bad})
        assert result.returncode == 2, (bad, result.returncode, result.stdout)


if __name__ == "__main__":
    import tempfile

    test_family_classification()
    for fn in (
        test_shares_and_stop_below_floor,
        test_gap_candidate_and_no_gap,
        test_sparse_mla_tactic_diversity,
        test_graph_grouping_and_gzip,
        test_rank_attribution_keeps_clock_domains_separate,
        test_fail_closed_parsing,
        test_truncated_trace_is_rejected,
        test_cli_writes_json,
        test_streaming_parser_handles_pretty_printed_trace,
        test_generation_attribution_and_roofline,
        test_derived_occupancy_from_launch_geometry,
        test_probe_refuses_unarmed_boot,
        test_probe_unreachable_server,
        test_window_recovery_helpers,
        test_window_recovers_timers_after_partial_disarm,
        test_probe_capture_gate_detects_autostop,
        test_probe_refuses_stale_traces,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            fn(Path(tmp))
    test_launcher_argv_in_both_inner_scripts()
    test_window_gates_capture_acceptance_stdout()
    test_window_fails_closed_on_timers_and_traces()
    test_window_interrupt_leaves_recoverable_state()
    test_probe_gates_and_labels_are_truthful()
    test_profiler_knobs_are_not_shape_hashed()
    test_profile_knob_validation()
    print("decode-profile oracle tests OK")
