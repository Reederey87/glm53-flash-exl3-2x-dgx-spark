#!/usr/bin/env python3
"""Probe vLLM prefix-cache hits and cached_tokens reporting.

For each prompt length, send the same unique random token-ID prompt twice and
record, per request:

* ``usage.prompt_tokens_details.cached_tokens`` from the API response
  (``None`` when the server was not started with
  ``--enable-prompt-tokens-details``);
* the delta of ``vllm:prefix_cache_queries_total`` and
  ``vllm:prefix_cache_hits_total`` across the request;
* the resolved cache geometry from ``vllm:cache_config_info``.

This separates "the cache did not hit" from "the API did not report it" and
shows whether a hit is below the hybrid attention block.

Usage:
  python3 scripts/probe_prefix_cache.py --base http://127.0.0.1:8000 \
      --lengths 1024,4096,8192,16384
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

QUERIES = "vllm:prefix_cache_queries_total"
HITS = "vllm:prefix_cache_hits_total"


@dataclass
class Geometry:
    block_size: int | None = None
    mamba_block_size: int | None = None
    fine_grained_mamba: str | None = None
    prefix_match_unit: str | None = None
    enable_prefix_caching: str | None = None


@dataclass
class Record:
    length: int
    request: int
    prompt_tokens: int
    cached_tokens: int | None
    queries_delta: float
    hits_delta: float
    text: str


def http_json(url: str, payload: dict[str, Any] | None, api_key: str, timeout: float) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
        return response.status, json.loads(raw) if raw else None


def parse_metrics(text: str) -> tuple[dict[str, float], Geometry, dict[str, str]]:
    counters: dict[str, float] = {}
    geometry = Geometry()
    raw_labels: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("vllm:cache_config_info{"):
            labels = line.split("{", 1)[1].split("}", 1)[0]
            for part in labels.split(","):
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                value = value.strip('"')
                raw_labels[key] = value
            geometry = Geometry(
                block_size=int(raw_labels["block_size"]) if raw_labels.get("block_size", "").isdigit() else None,
                mamba_block_size=int(raw_labels["mamba_block_size"]) if raw_labels.get("mamba_block_size", "").isdigit() else None,
                fine_grained_mamba=raw_labels.get("enable_mamba_fine_grained_prefix_cache"),
                prefix_match_unit=raw_labels.get("prefix_match_unit"),
                enable_prefix_caching=raw_labels.get("enable_prefix_caching"),
            )
            continue
        try:
            head, raw_value = line.rsplit(None, 1)
            value = float(raw_value)
        except ValueError:
            continue
        name = head.split("{", 1)[0]
        counters[name] = counters.get(name, 0.0) + value
    return counters, geometry, raw_labels


def scrape(base: str, api_key: str, timeout: float) -> tuple[dict[str, float], Geometry, dict[str, str]]:
    headers = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base.rstrip('/')}/metrics", headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return parse_metrics(response.read().decode("utf-8", errors="replace"))


def make_prompt(length: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(length)]


def complete(
    base: str, model: str, prompt: list[int], api_key: str, timeout: float
) -> tuple[int, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
    }
    return http_json(f"{base.rstrip('/')}/v1/completions", payload, api_key, timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="nvidia/GLM-5.3-Flash-FP4-Local")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--lengths", default="1024,4096,8192,16384")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    lengths = [int(value) for value in args.lengths.split(",") if value.strip()]
    records: list[Record] = []
    geometry: Geometry | None = None
    raw_labels: dict[str, str] = {}

    for length in lengths:
        prompt = make_prompt(length, args.seed + length)
        for request_index in range(1, 3):
            before, _, _ = scrape(args.base, args.api_key, args.timeout)
            status, body = complete(args.base, args.model, prompt, args.api_key, args.timeout)
            after, geometry, raw_labels = scrape(args.base, args.api_key, args.timeout)
            if status != 200:
                print(f"len={length} request={request_index}: HTTP {status}: {str(body)[:200]}", file=sys.stderr)
                continue
            usage = (body or {}).get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens") if isinstance(details, dict) else None
            text = ""
            choices = (body or {}).get("choices") or []
            if choices:
                text = (choices[0].get("text") or "")[:20]
            records.append(
                Record(
                    length=length,
                    request=request_index,
                    prompt_tokens=int(usage.get("prompt_tokens") or length),
                    cached_tokens=cached,
                    queries_delta=after.get(QUERIES, 0.0) - before.get(QUERIES, 0.0),
                    hits_delta=after.get(HITS, 0.0) - before.get(HITS, 0.0),
                    text=text,
                )
            )

    payload = {
        "base": args.base,
        "model": args.model,
        "geometry": asdict(geometry) if geometry else None,
        "labels": raw_labels,
        "records": [asdict(record) for record in records],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if geometry:
            print(
                f"cache: block_size={geometry.block_size} mamba_block_size={geometry.mamba_block_size} "
                f"fine_grained_mamba={geometry.fine_grained_mamba} match_unit={geometry.prefix_match_unit} "
                f"enabled={geometry.enable_prefix_caching}"
            )
        print(f"{'len':>7} {'req':>3} {'prompt':>7} {'cached':>7} {'q_delta':>9} {'hit_delta':>9} text")
        for record in records:
            cached = "-" if record.cached_tokens is None else str(record.cached_tokens)
            print(
                f"{record.length:7d} {record.request:3d} {record.prompt_tokens:7d} {cached:>7} "
                f"{record.queries_delta:9.0f} {record.hits_delta:9.0f} {record.text!r}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
