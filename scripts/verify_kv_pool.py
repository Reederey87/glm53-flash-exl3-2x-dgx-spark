#!/usr/bin/env python3
"""Read the booted vLLM KV-cache geometry from the container log.

The 2x DGX Spark profile targets a 2M-token FP8 KV pool. vLLM prints the
authoritative number once at startup:

    GPU KV cache size: 2,097,152 tokens

This script parses the most recent such line from ``docker logs`` (or a log
file), prints the pool, and fails when it is below ``--min-tokens``. It is
read-only and depends only on the standard library.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass

TOKEN_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens", re.IGNORECASE)
GIB_RE = re.compile(r"GPU KV cache size:.*?\(([\d.]+)\s*GiB\)", re.IGNORECASE)
AVAILABLE_RE = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB", re.IGNORECASE)
BLOCKS_RE = re.compile(r"#\s*GPU blocks:\s*([\d,]+)", re.IGNORECASE)
BLOCK_SIZE_RE = re.compile(r"block_size[=:]\s*([\d,]+)", re.IGNORECASE)


@dataclass(frozen=True)
class KVPool:
    tokens: int
    blocks: int | None
    block_size: int | None
    gib: float | None
    line: str


def _last_int(regex: re.Pattern[str], text: str) -> int | None:
    matches = regex.findall(text)
    if not matches:
        return None
    return int(matches[-1].replace(",", ""))


def _last_float(regex: re.Pattern[str], text: str) -> float | None:
    matches = regex.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_kv_log(text: str) -> KVPool | None:
    """Return the last KV-cache geometry in *text*, or None."""
    token_matches = TOKEN_RE.findall(text)
    if not token_matches:
        return None
    tokens = int(token_matches[-1].replace(",", ""))
    line = ""
    for candidate in text.splitlines():
        if TOKEN_RE.search(candidate):
            line = candidate.strip()
    return KVPool(
        tokens=tokens,
        blocks=_last_int(BLOCKS_RE, text),
        block_size=_last_int(BLOCK_SIZE_RE, text),
        gib=_last_float(GIB_RE, text) or _last_float(AVAILABLE_RE, text),
        line=line,
    )


def read_log(container: str, log_file: str, tail: int) -> str:
    if log_file:
        with open(log_file, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    cmd = ["docker", "logs", "--tail", str(tail), container]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"docker logs failed for {container!r}: {result.stderr.strip() or result.returncode}"
        )
    return result.stdout + result.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--container", default="vllm_node", help="container name (default: vllm_node)")
    source.add_argument("--log-file", default="", help="read a saved log instead of docker logs")
    parser.add_argument("--tail", type=int, default=20000, help="log lines to inspect (default: 20000)")
    parser.add_argument("--min-tokens", type=int, default=2_000_000, help="required pool size")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    try:
        text = read_log(args.container, args.log_file, args.tail)
    except (OSError, RuntimeError) as exc:
        print(f"kv-check: {exc}", file=sys.stderr)
        return 2

    pool = parse_kv_log(text)
    if pool is None:
        print(
            "kv-check: no 'GPU KV cache size' line found. Is the server booted "
            "with --kv-cache-dtype fp8 and the recipe's --kv-cache-memory-bytes?",
            file=sys.stderr,
        )
        return 2

    ok = pool.tokens >= args.min_tokens
    payload = {
        "ok": ok,
        "min_tokens": args.min_tokens,
        "pool": asdict(pool),
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        gib = f", ~{pool.gib:.2f} GiB" if pool.gib is not None else ""
        print(
            f"KV pool: {pool.tokens:,} tokens{gib} "
            f"(target >= {args.min_tokens:,}) -> {'PASS' if ok else 'BELOW TARGET'}"
        )
    if not ok:
        print(
            "kv-check: raise --kv-cache-memory-bytes or lower --max-model-len; "
            "see docs/04-kv-cache.md",
            file=sys.stderr,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
