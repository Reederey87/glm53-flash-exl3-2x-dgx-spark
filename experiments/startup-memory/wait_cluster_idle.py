#!/usr/bin/env python3
"""Read-only, both-node readiness gate for the fixed two-Spark serving profile.

This enforces the workspace's conservative idle policy, not an estimate of
total allocatable CUDA memory. It never stops containers or reclaims memory.
Stage as scripts/wait_cluster_idle.py only in its registered service window.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import time

MIN_FREE_KIB = 90 * 1024 * 1024
RESERVE_KIB = 1048576
PROBE_SECONDS = 10
FIELDS = {"mem_free_kib", "mem_available_kib", "reserve_kib", "container_count"}
PROBE = r"""set -eu
export LC_ALL=C
docker info >/dev/null
ids=$(docker ps -aq --filter name=vllm_node)
count=$(printf '%s\n' "$ids" | awk 'NF {n++} END {print n+0}')
free=$(awk '$1 == "MemFree:" {print $2}' /proc/meminfo)
available=$(awk '$1 == "MemAvailable:" {print $2}' /proc/meminfo)
reserve=$(cat /proc/sys/vm/min_free_kbytes)
printf 'mem_free_kib=%s\nmem_available_kib=%s\nreserve_kib=%s\ncontainer_count=%s\n' "$free" "$available" "$reserve" "$count"
"""


@dataclass(frozen=True)
class Observation:
    mem_free_kib: int | None = None
    mem_available_kib: int | None = None
    reserve_kib: int | None = None
    container_count: int | None = None
    error: str = ""


def parse_probe(text: str) -> Observation:
    values = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key not in FIELDS or key in values or not re.fullmatch(r"[0-9]+", value):
            raise ValueError("malformed or duplicate readiness field")
        values[key] = int(value)
    if set(values) != FIELDS:
        raise ValueError("incomplete readiness response")
    return Observation(**values)


def reason(observation: Observation) -> str:
    if observation.error:
        return observation.error
    if any(getattr(observation, field) is None for field in FIELDS):
        return "incomplete node observation"
    if observation.container_count != 0:
        return "serving container still present"
    if observation.reserve_kib != RESERVE_KIB:
        return f"vm.min_free_kbytes differs from required {RESERVE_KIB}"
    if observation.mem_free_kib < MIN_FREE_KIB:
        return f"MemFree below {MIN_FREE_KIB} KiB"
    return ""


def all_ready(nodes: dict[str, Observation]) -> bool:
    return set(nodes) == {"head", "worker"} and all(not reason(value) for value in nodes.values())


def worker_address(value: str) -> str:
    if not re.fullmatch(r"(?:[A-Za-z_][A-Za-z0-9_-]*@)?[A-Za-z0-9][A-Za-z0-9.-]*", value):
        raise argparse.ArgumentTypeError("expected a direct SSH host or user@host")
    return value


def probe_node(node: str, worker: str, timeout: float) -> Observation:
    # Bound the remote read-only process as well as the local SSH client.
    command = ["/usr/bin/timeout", "--kill-after=1", "8", "/bin/sh", "-s"]
    if node == "worker":
        command = [
            "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=1",
            "-o", "ProxyCommand=none", "-o", "ProxyJump=none",
            worker_address(worker), *command,
        ]
    elif node != "head":
        raise ValueError("unknown cluster node")
    try:
        result = subprocess.run(command, input=PROBE, text=True, capture_output=True,
                                timeout=timeout, check=False)
        if result.returncode:
            return Observation(error=f"probe exit {result.returncode}: {result.stderr.strip()[:240]}")
        return parse_probe(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return Observation(error=f"probe failed: {type(exc).__name__}: {str(exc)[:240]}")


def wait_for_cluster(worker: str, timeout: float, poll_seconds: float, *, once=False,
                     probe=probe_node, clock=time.monotonic, sleep=time.sleep, emit=None) -> dict:
    started = clock()
    deadline = started + timeout
    attempts = []
    while True:
        # Every cycle reads BOTH nodes anew. Never combine an old head success
        # with a later worker success: all ranks must be ready in this cycle.
        nodes = {}
        for node in ("head", "worker"):
            remaining = deadline - clock()
            nodes[node] = (probe(node, worker, min(PROBE_SECONDS, remaining))
                           if remaining > 0 else Observation(error="readiness deadline reached"))
        now = clock()
        ready = now <= deadline and all_ready(nodes)
        attempt = {"elapsed_s": now - started, "ready": ready,
                   "nodes": {node: {**asdict(value), "not_ready_reason": reason(value)}
                             for node, value in nodes.items()}}
        attempts.append(attempt)
        if emit:
            emit(attempt)
        if ready or once or now >= deadline:
            return {"ready": ready, "worker": worker, "min_free_kib": MIN_FREE_KIB,
                    "required_reserve_kib": RESERVE_KIB, "attempts": attempts}
        sleep(min(poll_seconds, deadline - now))


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=worker_address, default="nvidia@192.168.177.11")
    parser.add_argument("--timeout", type=positive_int, default=600)
    parser.add_argument("--poll-seconds", type=positive_int, default=5)
    parser.add_argument("--once", action="store_true", help="one read-only readiness cycle")
    parser.add_argument("--receipt", type=Path, help="new JSON receipt path; existing files are refused")
    args = parser.parse_args()
    report = wait_for_cluster(args.worker, args.timeout, args.poll_seconds, once=args.once,
                              emit=lambda value: print(json.dumps(value, sort_keys=True), flush=True))
    if args.receipt:
        with args.receipt.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps({"ready": report["ready"], "attempts": len(report["attempts"])}))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
