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
    assert fused["graph_ids"] == ["7", "9"]


def test_fail_closed_parsing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert audit.main(["--trace-dir", str(empty)]) == 2

    no_kernels = tmp_path / "nokern.pt.trace.json"
    no_kernels.write_text(json.dumps({"traceEvents": [{"cat": "cpu_op", "name": "x"}]}))
    assert audit.main(["--trace-dir", str(tmp_path)]) == 2


def test_cli_writes_json(tmp_path: Path) -> None:
    write_trace(
        tmp_path / "rank0.pt.trace.json",
        [kernel("exl3_moe_kernel<k4>", 100.0, **{"est. achieved occupancy %": 70}),
         kernel("cutlass::gemm", 900.0)],
    )
    out = tmp_path / "report.json"
    assert audit.main(["--trace-dir", str(tmp_path), "--json-out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["schema"] == 2
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
    # the complete object before the truncation is still yielded, then EOF stops cleanly
    assert len(list(audit.iter_trace_events(truncated))) == 1


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
    # 82.7 unique experts x 1 MB in 400 us -> ~207 GB/s, past the 200 GB/s roofline
    assert bounded["decision"]["task29"]["verdict"] == "STOP_ROOFLINE_BOUND"
    assert bounded["roofline"]["roofline_bound"] is True

    slow = write_trace(
        tmp_path / "slow.pt.trace.json",
        [
            annotation("execute_context_0(0)_generation_4(12)", 100.0, 50.0),
            kernel("exl3_moe_kernel<k4>", 10_000_000.0, ts=110.0),
        ],
    )
    unbounded = audit.audit([slow], 0.05, 50.0, roof)
    assert unbounded["roofline"]["roofline_bound"] is False
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
        rc = probe.main(["--base", base, "--out", str(tmp_path / "r.json")])
        assert rc == 2
    finally:
        server.shutdown()


def test_probe_unreachable_server(tmp_path: Path) -> None:
    import probe_decode_profile as probe

    rc = probe.main(["--base", "http://127.0.0.1:1", "--out", str(tmp_path / "r.json")])
    assert rc == 2


def test_window_gates_capture_acceptance_stdout() -> None:
    """The gates phase read acc.stdout without capturing it and crashed after
    acceptance had already passed 7/7; keep the capture in place."""
    src = (ROOT / "scripts" / "run_decode_profile_window.py").read_text()
    body = src[src.index("def phase_gates"):]
    body = body[: body.index("\ndef ", 1)]
    assert "capture_output=True" in body
    assert "acc.stdout" in body


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
        test_fail_closed_parsing,
        test_cli_writes_json,
        test_streaming_parser_handles_pretty_printed_trace,
        test_generation_attribution_and_roofline,
        test_derived_occupancy_from_launch_geometry,
        test_probe_refuses_unarmed_boot,
        test_probe_unreachable_server,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            fn(Path(tmp))
    test_launcher_argv_in_both_inner_scripts()
    test_window_gates_capture_acceptance_stdout()
    test_profiler_knobs_are_not_shape_hashed()
    test_profile_knob_validation()
    print("decode-profile oracle tests OK")
