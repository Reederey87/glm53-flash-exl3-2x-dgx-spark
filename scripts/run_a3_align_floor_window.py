#!/usr/bin/env python3
"""Run one guarded A3 mixed-prefill align-floor arm on the cluster head.

The service restart and environment flip stay operator-controlled. This runner
validates the effective container environment, starts both-node MemFree
monitoring before the first request, runs the fixed 60k newcomer probe, and
requires a decode-floor-v3 log showing a sub-page cap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "tests" / "bench_mixed_prefill.py"
BLOCK_SIZE = 3584
BASE = "http://127.0.0.1:8000"
MIN_TRIPWIRE_KIB = 3 * 1024 * 1024
SELECTED_ENV = (
    "LONG_PREFILL_TOKEN_THRESHOLD",
    "MAX_NUM_BATCHED_TOKENS",
    "MAX_NUM_SEQS",
    "GLM53_ALIGN_FLOOR",
    "GLM53_MIXED_PREFILL_CHUNK",
    "GLM53_MIXED_PREFILL_MAX_WAIT_MS",
    "GLM53_MIXED_PREFILL_WARM_TOKENS",
    "GLM53_MIXED_PREFILL_LATE_CAP",
    "GLM53_MIXED_PREFILL_ESCALATE_MS",
    "GLM53_MIXED_PREFILL_LATE_CAP_MAX",
)


def run_text(argv: list[str], timeout: float = 30) -> str:
    completed = subprocess.run(
        argv,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout


def parse_env(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name in SELECTED_ENV:
            result[name] = value
    return result


def validate_arm(env: dict[str, str], expected_align_floor: str) -> list[str]:
    errors: list[str] = []
    required = set(SELECTED_ENV)
    missing = sorted(required - env.keys())
    if missing:
        errors.append("missing effective environment: " + ", ".join(missing))
        return errors

    try:
        lptt = int(env["LONG_PREFILL_TOKEN_THRESHOLD"])
        mnbt = int(env["MAX_NUM_BATCHED_TOKENS"])
        late_cap = int(env["GLM53_MIXED_PREFILL_LATE_CAP"])
    except ValueError as exc:
        errors.append(f"non-integer effective environment: {exc}")
        return errors

    if lptt < BLOCK_SIZE:
        errors.append(f"LPTT must be >= {BLOCK_SIZE}, got {lptt}")
    if mnbt < BLOCK_SIZE:
        errors.append(f"MNBT must be >= {BLOCK_SIZE}, got {mnbt}")
    if late_cap >= BLOCK_SIZE:
        errors.append(f"late cap must be sub-page, got {late_cap}")
    if env["GLM53_ALIGN_FLOOR"] != expected_align_floor:
        errors.append(
            "align-floor arm mismatch: expected "
            f"{expected_align_floor}, got {env['GLM53_ALIGN_FLOOR']}"
        )
    if env["GLM53_MIXED_PREFILL_CHUNK"] != "skip":
        errors.append(
            "mixed-prefill policy must remain skip, got "
            f"{env['GLM53_MIXED_PREFILL_CHUNK']!r}"
        )
    return errors


def read_memfree_kib(path: Path = Path("/proc/meminfo")) -> int:
    for line in path.read_text().splitlines():
        if line.startswith("MemFree:"):
            return int(line.split()[1])
    raise RuntimeError(f"MemFree missing from {path}")


def read_worker_memfree_kib(worker: str) -> int:
    text = run_text(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            worker,
            "grep '^MemFree:' /proc/meminfo",
        ],
        timeout=15,
    )
    return int(text.split()[1])


def matches_scheduler_request_id(log_id: str, request_ids: set[str]) -> bool:
    for request_id in request_ids:
        external_id = f"chatcmpl-{request_id}"
        if log_id == external_id or re.fullmatch(
            rf"{re.escape(external_id)}-[0-9a-f]{{8}}", log_id
        ):
            return True
    return False


def extract_subblock_caps(log_text: str, request_ids: set[str]) -> list[int]:
    caps: list[int] = []
    for line in log_text.splitlines():
        request_match = re.search(r"\breq=(\S+)", line)
        if not request_match or not matches_scheduler_request_id(
            request_match.group(1), request_ids
        ):
            continue
        if "[glm53-decode-floor-v3] late-admit" in line:
            match = re.search(r"\bcap=(\d+)\b", line)
        elif "[glm53-decode-floor-v3] late-escalate" in line:
            match = re.search(r"->(\d+)\b", line)
        else:
            continue
        if match and int(match.group(1)) < BLOCK_SIZE:
            caps.append(int(match.group(1)))
    return caps


def benchmark_request_ids(path: Path) -> set[str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("benchmark receipt has no samples")
    ids = {
        row.get("newcomer_request_id")
        for row in samples
        if isinstance(row, dict) and row.get("newcomer_request_id")
    }
    if len(ids) != len(samples):
        raise ValueError("benchmark receipt is missing newcomer request IDs")
    return ids


def read_runtime_logs(container: str, since_epoch: int) -> str:
    completed = subprocess.run(
        ["docker", "logs", "--since", str(since_epoch), container],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout + completed.stderr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MemoryMonitor:
    def __init__(self, worker: str, tripwire_kib: int, interval: float) -> None:
        self.worker = worker
        self.tripwire_kib = tripwire_kib
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self.breached = threading.Event()
        self.ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self, timeout: float = 20) -> None:
        self._thread.start()
        if not self.ready.wait(timeout):
            self.error = "initial both-node sample timed out"
            self.breached.set()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                head = read_memfree_kib()
                worker = read_worker_memfree_kib(self.worker)
                self.samples.append(
                    {"time": time.time(), "head_kib": head, "worker_kib": worker}
                )
                if min(head, worker) < self.tripwire_kib:
                    self.breached.set()
                self.ready.set()
                if self.breached.is_set():
                    return
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.breached.set()
                self.ready.set()
                return
            self._stop.wait(self.interval)


def tripwire_kib(raw: str) -> int:
    value = int(raw)
    if value < MIN_TRIPWIRE_KIB:
        raise argparse.ArgumentTypeError(
            f"tripwire must be at least {MIN_TRIPWIRE_KIB} KiB (3 GiB)"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expected-align-floor", choices=("0", "1"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--worker", default="nvidia@192.168.177.11")
    parser.add_argument("--container", default="glm53-exl3-head")
    parser.add_argument(
        "--tripwire-kib", type=tripwire_kib, default=MIN_TRIPWIRE_KIB
    )
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--incumbent-tokens", type=int, default=2400)
    parser.add_argument("--timeout", type=float, default=1800)
    return parser.parse_args()


def install_signal_handlers(interrupted: threading.Event) -> None:
    def handle_signal(_signum: int, _frame: object) -> None:
        interrupted.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_benchmark(
    args: argparse.Namespace,
    bench_out: Path,
    monitor: MemoryMonitor,
    interrupted: threading.Event,
) -> int:
    bench_cmd = [
        sys.executable,
        str(BENCH),
        "--contexts",
        "60000",
        "--samples",
        str(args.samples),
        "--incumbent-tokens",
        str(args.incumbent_tokens),
        "--timeout",
        str(args.timeout),
        "--out",
        str(bench_out),
    ]
    child_env = os.environ.copy()
    child_env["GLM53_BASE"] = BASE
    process = subprocess.Popen(bench_cmd, env=child_env)
    try:
        while process.poll() is None:
            if monitor.breached.wait(0.25) or interrupted.is_set():
                terminate_process(process)
                break
        return process.wait()
    finally:
        terminate_process(process)


def main() -> int:
    args = parse_args()
    if args.tripwire_kib < MIN_TRIPWIRE_KIB:
        raise SystemExit(
            f"tripwire must be at least {MIN_TRIPWIRE_KIB} KiB (3 GiB)"
        )
    out_path = Path(args.out)
    bench_out = out_path.with_suffix(".bench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started_epoch = int(time.time())
    interrupted = threading.Event()
    install_signal_handlers(interrupted)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "label": args.label,
        "started": time.time(),
        "expected_align_floor": args.expected_align_floor,
        "tripwire_kib": args.tripwire_kib,
        "base": BASE,
        "bench_path": str(bench_out),
        "runner_sha256": sha256(Path(__file__)),
        "bench_sha256": sha256(BENCH),
        "errors": [],
    }

    try:
        env_text = run_text(
            [
                "docker",
                "inspect",
                args.container,
                "--format",
                "{{range .Config.Env}}{{println .}}{{end}}",
            ]
        )
        effective_env = parse_env(env_text)
        receipt["effective_env"] = effective_env
        receipt["errors"].extend(
            validate_arm(effective_env, args.expected_align_floor)
        )
        if receipt["errors"]:
            return finish(receipt, out_path)

        initial_metrics = run_text(["curl", "-fsS", f"{BASE}/metrics"])
        preemptions_before = metric_total(
            initial_metrics, "num_preemptions_total"
        )
        if preemptions_before is None:
            raise ValueError("initial preemption counter missing or invalid")
        receipt["preemptions_before"] = preemptions_before

        monitor = MemoryMonitor(
            args.worker, args.tripwire_kib, args.monitor_interval
        )
        bench_rc = 1
        try:
            monitor.start()
            if not monitor.breached.is_set() and not interrupted.is_set():
                bench_rc = run_benchmark(
                    args, bench_out, monitor, interrupted
                )
        finally:
            monitor.stop()
        receipt["bench_returncode"] = bench_rc
        receipt["memory_samples"] = monitor.samples
        receipt["memory_monitor_error"] = monitor.error
        if monitor.samples:
            receipt["min_memfree_kib"] = {
                "head": min(row["head_kib"] for row in monitor.samples),
                "worker": min(row["worker_kib"] for row in monitor.samples),
            }
        if monitor.error:
            receipt["errors"].append(
                "memory monitor failed closed: " + monitor.error
            )
        if monitor.breached.is_set() and not monitor.error:
            receipt["errors"].append(
                f"MemFree crossed {args.tripwire_kib} KiB tripwire"
            )
        if interrupted.is_set():
            receipt["errors"].append("runner interrupted")
        if bench_rc != 0:
            receipt["errors"].append(f"mixed-prefill bench exited {bench_rc}")

        logs = read_runtime_logs(args.container, started_epoch)
        request_ids = benchmark_request_ids(bench_out)
        receipt["newcomer_request_ids"] = sorted(request_ids)
        caps = extract_subblock_caps(logs, request_ids)
        receipt["subblock_caps"] = caps
        receipt["decode_floor_log_lines"] = [
            line
            for line in logs.splitlines()
            if "[glm53-decode-floor-v3]" in line
        ]
        if not caps:
            receipt["errors"].append(
                "no decode-floor-v3 sub-page late cap was logged"
            )

        final_metrics = run_text(["curl", "-fsS", f"{BASE}/metrics"])
        preemptions_after = metric_total(final_metrics, "num_preemptions_total")
        if preemptions_after is None:
            raise ValueError("final preemption counter missing or invalid")
        receipt["preemptions_after"] = preemptions_after
        receipt["preemptions_delta"] = preemptions_after - preemptions_before
        if receipt["preemptions_delta"]:
            receipt["errors"].append(
                f"preemptions increased by {receipt['preemptions_delta']}"
            )
    except Exception as exc:
        receipt["errors"].append(f"{type(exc).__name__}: {exc}")
    if interrupted.is_set() and "runner interrupted" not in receipt["errors"]:
        receipt["errors"].append("runner interrupted")
    return finish(receipt, out_path)


def metric_total(text: str, name: str) -> float | None:
    values = re.findall(
        rf"^vllm:{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)$",
        text,
        re.MULTILINE,
    )
    if not values:
        return None
    parsed = [float(value) for value in values]
    if not all(math.isfinite(value) for value in parsed):
        return None
    return sum(parsed)


def finish(receipt: dict[str, Any], out_path: Path) -> int:
    receipt["finished"] = time.time()
    receipt["decision"] = "pass" if not receipt["errors"] else "abort"
    out_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["decision"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
