#!/usr/bin/env python3
"""Measure one shared-expert-launch arm on the cluster head.

The service restart and the environment flip stay operator-controlled. This
runner measures one phase and writes a receipt:

* the three standing decode lanes (structured, hashmap prose, hard essay), each
  with a short warmup and N measured runs at the standing 512-token cap;
* per-lane medians plus min/max and the DFlash2 acceptance ratio;
* sampled MemFree minima on both nodes for the whole phase;
* the effective container environment and whether the candidate's marker is
  present in the installed source, so "the run succeeded" is never mistaken for
  "the treatment was active".

``--compare`` reads two receipts and applies the pre-registered gate from
``docs/21-shared-experts-overlap.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "tests" / "bench_decode.py"
BASE = "http://127.0.0.1:8000"
# The head is "local": this runner executes on spark1 and spark1 cannot
# self-ssh (the Sync key is not in its own authorized_keys).
HEAD = "local"
WORKER = "nvidia@192.168.177.11"
HEAD_CONTAINER = "glm53-exl3-head"
WORKER_CONTAINER = "glm53-exl3-worker"
MARKER = "[glm53-shared-experts-early]"
SHARED_EXPERTS_PY = (
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "fused_moe/runner/shared_experts.py"
)
LANES = ("structured", "prose", "essay")
FLOORS = {"structured": 68.8, "prose": 30.0, "essay": 20.0}
PARITY = 0.97
MEMFREE_FLOOR_KIB = 2.5 * 1024 * 1024


def _ssh(host: str, command: str, timeout: int = 60) -> str:
    """Run ``command`` on ``host``; ``"local"`` runs it on this node."""
    argv = (
        ["bash", "-lc", command]
        if host == "local"
        else ["ssh", "-o", "ConnectTimeout=15", host, command]
    )
    done = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return done.stdout


def _health() -> int:
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=10) as resp:
            return resp.status
    except Exception:
        return 0


class MemWatch:
    """Sample MemFree on both nodes until stopped."""

    def __init__(self) -> None:
        self.samples: dict[str, list[int]] = {"head": [], "worker": []}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            for key, host in (("head", HEAD), ("worker", WORKER)):
                out = _ssh(
                    host,
                    "awk '/MemFree/{print $2}' /proc/meminfo",
                    timeout=30,
                ).strip()
                if out.isdigit():
                    self.samples[key].append(int(out))
            self._stop.wait(5.0)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, dict[str, int | None]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        out: dict[str, dict[str, int | None]] = {}
        for key, values in self.samples.items():
            out[key] = {
                "min_kib": min(values) if values else None,
                "max_kib": max(values) if values else None,
                "samples": len(values),
            }
        return out


def _effective_env() -> dict[str, str]:
    raw = _ssh(
        HEAD,
        f"docker inspect {HEAD_CONTAINER} "
        "--format '{{range .Config.Env}}{{println .}}{{end}}'",
    )
    wanted = (
        "GLM53_SHARED_EXPERTS_EARLY",
        "GLM53_ROUTER_ONCE",
        "GLM53_INDEXER_WORKSPACE",
    )
    return {
        k: v
        for k, v in (line.split("=", 1) for line in raw.splitlines() if "=" in line)
        if k in wanted
    }


def _marker_present(container: str, host: str) -> int:
    # -F: the marker's brackets would otherwise parse as a character class.
    out = _ssh(
        host,
        f"docker exec {container} grep -c -F -- {MARKER!r} {SHARED_EXPERTS_PY} "
        "2>/dev/null || true",
    ).strip()
    return int(out) if out.isdigit() else -1


def _run_lane(lane: str, out_dir: Path, phase: str, runs: int, cap: int) -> dict:
    target = out_dir / f"{phase}-{lane}.json"
    cmd = [
        sys.executable,
        str(BENCH),
        "--phase",
        f"{phase}-{lane}",
        "--out",
        str(target),
        "--runs",
        str(runs),
        "--max-tokens",
        str(cap),
    ]
    if lane == "structured":
        cmd.append("--structured")
    elif lane == "essay":
        cmd.append("--essay")
    print(f"[window] {phase} lane={lane} runs={runs} cap={cap}", flush=True)
    done = subprocess.run(
        cmd, capture_output=True, text=True, check=False, timeout=1800
    )
    if done.returncode != 0:
        print(done.stdout[-2000:], flush=True)
        print(done.stderr[-2000:], file=sys.stderr, flush=True)
    payload = json.loads(target.read_text()) if target.is_file() else {}
    return {
        "lane": lane,
        "returncode": done.returncode,
        "tok_s_median": payload.get("tok_s_median"),
        "tok_s_min": payload.get("tok_s_min"),
        "tok_s_max": payload.get("tok_s_max"),
        "ttft_median_s": payload.get("ttft_median_s"),
        "accept_ratio_median": payload.get("accept_ratio_median"),
        "accepted_per_step_median": payload.get("accepted_per_step_median"),
        "completion_tokens_median": payload.get("completion_tokens_median"),
        "any_nan": payload.get("any_nan"),
        "coherent": payload.get("coherent"),
        "health_code_after": payload.get("health_code_after"),
        "receipt": str(target),
    }


def measure(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    before = _health()
    if before != 200:
        print(f"[window] /health={before} — refusing to measure", file=sys.stderr)
        return 2

    env = _effective_env()
    armed = env.get("GLM53_SHARED_EXPERTS_EARLY", "<unset>")
    marks = {
        "head": _marker_present(HEAD_CONTAINER, HEAD),
        "worker": _marker_present(WORKER_CONTAINER, WORKER),
    }
    expect = 3 if armed == "1" else 0
    for node, count in marks.items():
        if count != expect:
            print(
                f"[window] {node} marker count={count}, expected {expect} for "
                f"GLM53_SHARED_EXPERTS_EARLY={armed} — refusing to measure",
                file=sys.stderr,
            )
            return 2

    watch = MemWatch()
    watch.start()
    lanes: list[dict] = []
    try:
        for lane in LANES:
            if args.warmup:
                warm = out_dir / f"{args.phase}-warmup-{lane}.json"
                warm_cmd = [
                    sys.executable,
                    str(BENCH),
                    "--phase",
                    f"{args.phase}-warmup-{lane}",
                    "--out",
                    str(warm),
                    "--runs",
                    "1",
                    "--max-tokens",
                    "32",
                    "--skip-coherence",
                ]
                if lane == "essay":
                    warm_cmd.append("--essay")
                subprocess.run(
                    warm_cmd, capture_output=True, text=True, timeout=900, check=False
                )
            lanes.append(_run_lane(lane, out_dir, args.phase, args.runs, args.cap))
    finally:
        mem = watch.stop()

    after = _health()
    receipt = {
        "phase": args.phase,
        "ts": time.time(),
        "runs_per_lane": args.runs,
        "cap": args.cap,
        "health_before": before,
        "health_after": after,
        "effective_env": env,
        "marker_counts": marks,
        "memfree": mem,
        "lanes": lanes,
    }
    path = out_dir / f"{args.phase}-receipt.json"
    path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print("wrote", path)
    return 0


def _median_lane(receipt: object, lane: str) -> float | None:
    if not isinstance(receipt, dict):
        return None
    row = _lane_rows(receipt).get(lane)
    return row.get("tok_s_median") if isinstance(row, dict) else None


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _lane_rows(receipt: dict) -> dict[str, dict]:
    """Lane rows keyed by lane; an unexpected shape yields no rows."""
    lanes = receipt.get("lanes")
    if not isinstance(lanes, list):
        return {}
    out: dict[str, dict] = {}
    for row in lanes:
        if isinstance(row, dict) and isinstance(row.get("lane"), str):
            out.setdefault(row["lane"], row)
    return out


def _receipt_failures(label: str, receipt: object, expect_markers: int) -> list[str]:
    """Structural and evidence checks for one side of the comparison."""
    problems: list[str] = []
    if not isinstance(receipt, dict):
        return [f"{label}: receipt is not an object"]
    for key in ("health_before", "health_after"):
        if receipt.get(key) != 200:
            problems.append(f"{label}: {key}={receipt.get(key)!r}, expected 200")
    rows = _lane_rows(receipt)
    for lane in LANES:
        row = rows.get(lane)
        if row is None:
            problems.append(f"{label}/{lane}: no result for this lane")
            continue
        if row.get("returncode") != 0:
            problems.append(f"{label}/{lane}: benchmark exit {row.get('returncode')!r}")
        median = row.get("tok_s_median")
        if not _positive_number(median):
            problems.append(
                f"{label}/{lane}: median {median!r} is not a finite positive number"
            )
        # Absent evidence is not clean evidence.
        if row.get("any_nan") is not False:
            problems.append(
                f"{label}/{lane}: any_nan={row.get('any_nan')!r}, expected False"
            )
        if lane in ("prose", "essay") and row.get("coherent") is not True:
            problems.append(f"{label}/{lane}: coherence not confirmed")
    extra = sorted(set(rows) - set(LANES))
    if extra:
        problems.append(f"{label}: unexpected lanes {extra}")

    mem = receipt.get("memfree")
    if not isinstance(mem, dict):
        problems.append(f"{label}: no MemFree block")
    else:
        for node in ("head", "worker"):
            entry = mem.get(node)
            if not isinstance(entry, dict):
                problems.append(f"{label}/{node}: no MemFree entry")
                continue
            samples = entry.get("samples")
            if (
                not isinstance(samples, int)
                or isinstance(samples, bool)
                or samples <= 0
            ):
                problems.append(
                    f"{label}/{node}: MemFree samples={samples!r}, expected a positive count"
                )
            low = entry.get("min_kib")
            if not isinstance(low, int) or isinstance(low, bool) or low <= 0:
                problems.append(
                    f"{label}/{node}: MemFree min {low!r} is not a positive number"
                )
            elif low < MEMFREE_FLOOR_KIB:
                problems.append(
                    f"{label}/{node}: MemFree min {low} KiB below the "
                    f"{int(MEMFREE_FLOOR_KIB)} KiB floor"
                )

    marks = receipt.get("marker_counts")
    if not isinstance(marks, dict):
        problems.append(f"{label}: no installer marker counts")
    else:
        for node in ("head", "worker"):
            count = marks.get(node)
            if count != expect_markers:
                problems.append(
                    f"{label}/{node}: installer marker count {count!r}, "
                    f"expected {expect_markers}"
                )
    return problems


def smoke_failures(smoke: object) -> list[str]:
    """Serving-correctness evidence, which is part of the gate, not an extra."""
    if not isinstance(smoke, dict):
        return ["smoke: receipt is not an object"]
    problems: list[str] = []
    if smoke.get("ok") is not True:
        problems.append(f"smoke: ok={smoke.get('ok')!r}, expected True")
    if smoke.get("failed") != 0:
        problems.append(f"smoke: failed={smoke.get('failed')!r}, expected 0")
    if smoke.get("any_nan") is not False:
        problems.append(f"smoke: any_nan={smoke.get('any_nan')!r}, expected False")
    errors = smoke.get("log_errors")
    if not isinstance(errors, dict) or any(errors.get(n) for n in ("head", "worker")):
        problems.append(f"smoke: engine/CUDA/NCCL errors recorded: {errors!r}")
    failures = smoke.get("log_failures")
    if not isinstance(failures, dict):
        problems.append("smoke: no per-rank log collection status")
    else:
        for node in ("head", "worker"):
            if failures.get(node):
                problems.append(
                    f"smoke/{node}: log collection failed: {failures[node]!r}"
                )
    return problems


def gate_failures(control: dict, armed: dict, smoke: object = None) -> list[str]:
    """Every prerequisite the pre-registered gate needs before a verdict.

    Throughput alone is not the gate. An unhealthy pair, a failed or incoherent
    lane, an unarmed container, a memory-starved node or a malformed receipt
    makes the comparison unreadable, and reporting such a probe as adopted would
    be worse than reporting nothing. The serving smoke is part of the gate too,
    so its absence is itself a failure rather than a silent omission.
    """
    problems = _receipt_failures("control", control, 0)
    problems += _receipt_failures("armed", armed, 3)
    env = armed.get("effective_env") if isinstance(armed, dict) else None
    if not isinstance(env, dict) or env.get("GLM53_SHARED_EXPERTS_EARLY") != "1":
        problems.append(
            "armed: GLM53_SHARED_EXPERTS_EARLY is not 1 in the container environment"
        )
    if smoke is None:
        problems.append("smoke: no serving-smoke receipt supplied")
    else:
        problems += smoke_failures(smoke)
    return problems


def compare_rows(control: dict, armed: dict) -> tuple[list[dict], str]:
    """Per-lane ratios and the throughput-only verdict."""
    rows: list[dict] = []
    verdict = "ADOPT"
    for lane in LANES:
        c = _median_lane(control, lane)
        a = _median_lane(armed, lane)
        if not _positive_number(c) or not _positive_number(a):
            verdict = "NEED-MORE-DATA"
            rows.append({"lane": lane, "control": c, "armed": a, "ratio": None})
            continue
        ratio = a / c
        above_parity = ratio >= PARITY
        above_floor = a >= FLOORS[lane]
        rows.append(
            {
                "lane": lane,
                "control": c,
                "armed": a,
                "delta_pct": (ratio - 1.0) * 100.0,
                "ratio": ratio,
                "parity": above_parity,
                "above_floor": above_floor,
                "floor": FLOORS[lane],
            }
        )
        if not (above_parity and above_floor):
            verdict = "REVERT"
    if any(
        r.get("any_nan") is not False
        for r in list(_lane_rows(control).values()) + list(_lane_rows(armed).values())
    ):
        verdict = "REVERT"
    return rows, verdict


def compare(args: argparse.Namespace) -> int:
    control = json.loads(Path(args.compare[0]).read_text())
    armed = json.loads(Path(args.compare[1]).read_text())
    smoke = json.loads(Path(args.smoke).read_text()) if args.smoke else None
    problems = gate_failures(control, armed, smoke)
    rows, verdict = compare_rows(control, armed)
    # An unreadable or incomplete probe is not a result, whatever the throughput
    # says. Without the serving smoke this is a partial assessment, not a gate.
    if problems:
        verdict = "PARTIAL-GATE" if args.smoke is None else "INVALID"
    out = {
        "control": args.compare[0],
        "armed": args.compare[1],
        "smoke": args.smoke,
        "parity_floor": PARITY,
        "rows": rows,
        "gate_failures": problems,
        "complete": not problems,
        "verdict": verdict,
        "memfree": {
            "control": control.get("memfree") if isinstance(control, dict) else None,
            "armed": armed.get("memfree") if isinstance(armed, dict) else None,
        },
        "marker_counts": armed.get("marker_counts")
        if isinstance(armed, dict)
        else None,
        "effective_env": armed.get("effective_env")
        if isinstance(armed, dict)
        else None,
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", help="label for this arm, e.g. control / armed")
    ap.add_argument(
        "--out-dir", default=str(ROOT / "local" / "shared-experts-20260927")
    )
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--warmup", action="store_true", default=True)
    ap.add_argument("--compare", nargs=2, metavar=("CONTROL", "ARMED"))
    ap.add_argument(
        "--smoke",
        help=(
            "serving-smoke receipt for the armed boot; it is part of the gate, so "
            "without it the comparison reports PARTIAL-GATE rather than a verdict"
        ),
    )
    ap.add_argument("--out", help="write the comparison verdict here")
    args = ap.parse_args()
    if args.compare:
        return compare(args)
    if not args.phase:
        ap.error("--phase is required unless --compare is used")
    return measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
