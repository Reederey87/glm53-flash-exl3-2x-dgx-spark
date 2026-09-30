#!/usr/bin/env python3
"""Record/compare live tokenization IDs and latency without GPU work."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
import time
import urllib.request
from pathlib import Path

SIZES = (2000, 8000, 16000)
RUNS = 9


def cases():
    for repeats in SIZES:
        prefix = "<|system|>" + "Review this Python function and explain each invariant.\n" * repeats
        for turn in range(RUNS + 1):
            yield f"{repeats}-{turn}", prefix + f"<|user|>turn {turn}: 中文 👩‍💻 café é </think><|assistant|>"


def fingerprint(tokens):
    if not isinstance(tokens, list) or not tokens or any(
        type(t) is not int or not 0 <= t < 2**32 for t in tokens
    ):
        raise ValueError("invalid tokenization response")
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


def compare(baseline, candidate):
    names = [name for name, _ in cases()]
    for receipt in (baseline, candidate):
        if set(receipt) != set(names):
            raise ValueError("incomplete live-tokenizer receipt")
        for name, text in cases():
            row = receipt[name]
            if row["text_sha256"] != hashlib.sha256(text.encode()).hexdigest():
                raise ValueError("corpus drift")
            if type(row["count"]) is not int or row["count"] <= 0:
                raise ValueError("invalid token count")
            if not isinstance(row["ids_sha256"], str) or len(row["ids_sha256"]) != 64:
                raise ValueError("invalid token hash")
            elapsed = row["seconds"]
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("invalid timing")
    ratios = {}
    for name in names:
        if (baseline[name]["ids_sha256"], baseline[name]["count"]) != (
            candidate[name]["ids_sha256"], candidate[name]["count"]
        ):
            raise ValueError(f"token IDs changed at {name}")
    for size in SIZES:
        keys = [f"{size}-{turn}" for turn in range(1, RUNS + 1)]
        ratios[str(size)] = statistics.median(
            candidate[key]["seconds"] / baseline[key]["seconds"] for key in keys
        )
    return {"exact_cases": len(names), "warm_paired_ratios": ratios}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=("record", "score"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.mode == "score" and not args.baseline:
        parser.error("score requires --baseline")
    receipt = {}
    for name, text in cases():
        request = urllib.request.Request(
            args.url.rstrip("/") + "/tokenize",
            data=json.dumps({"model": "GLM-5.3-Flash-EXL3", "prompt": text,
                             "add_special_tokens": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        elapsed = time.perf_counter() - start
        token_hash = fingerprint(result["tokens"])
        if result["count"] != len(result["tokens"]):
            raise ValueError("response count disagrees with IDs")
        receipt[name] = {"text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "ids_sha256": token_hash, "count": result["count"],
                         "seconds": elapsed}
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.mode == "score":
        print(json.dumps(compare(json.loads(args.baseline.read_text()), receipt), indent=2))
    else:
        print(f"recorded {len(receipt)} cases")


if __name__ == "__main__":
    main()
