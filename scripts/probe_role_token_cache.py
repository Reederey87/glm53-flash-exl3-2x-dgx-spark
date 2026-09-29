#!/usr/bin/env python3
"""Paired CPU-only real-tokenizer gate; no GPU allocation or weight access."""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=15)
    args = parser.parse_args()
    if args.pairs < 5:
        parser.error("at least five alternating pairs are required")
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.tokenizer))
    spec = importlib.util.spec_from_file_location("role_token_cache", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = module.SegmentCache(tokenizer)
    if cache.pattern is None:
        raise RuntimeError("deployed tokenizer not qualified")
    kwargs = {"add_special_tokens": False, "truncation": True, "max_length": 1000000}
    results = []
    exact = 0
    for repeats in (2000, 8000, 16000):
        prefix = "<|system|>" + "Review this Python function and explain each invariant.\n" * repeats
        cache.tokenize(prefix + "<|user|>prime<|assistant|>", kwargs)
        control, candidate = [], []
        for i in range(args.pairs):
            text = prefix + f"<|user|>turn {i}: 中文 👩‍💻 café é </think><|assistant|>"
            outputs = {}
            for arm in (("control", "candidate") if i % 2 == 0 else ("candidate", "control")):
                started = time.perf_counter()
                outputs[arm] = (tokenizer(text, **kwargs) if arm == "control"
                                else cache.tokenize(text, kwargs))
                (control if arm == "control" else candidate).append(time.perf_counter() - started)
            if outputs["control"]["input_ids"] != outputs["candidate"]["input_ids"]:
                raise RuntimeError("token-ID mismatch")
            exact += 1
        ratios = [b / a for a, b in zip(control, candidate)]
        results.append({"prompt_tokens": len(outputs["control"]["input_ids"]),
                        "pairs": args.pairs, "control_ms": statistics.median(control) * 1000,
                        "candidate_ms": statistics.median(candidate) * 1000,
                        "paired_median_ratio": statistics.median(ratios)})
    prefix = "<|system|>" + "Concurrency invariant.\n" * 3000
    texts = [prefix + f"<|observation|>tool result {i} 中文<|assistant|>" for i in range(16)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda text: cache.tokenize(text, kwargs, "agent"), texts))
    for text, output in zip(texts, outputs):
        if output["input_ids"] != tokenizer(text, **kwargs)["input_ids"]:
            raise RuntimeError("concurrent token-ID mismatch")
        exact += 1
    passed = (cache.hits > 0 and all(row["paired_median_ratio"] <= 0.98 for row in results)
              and cache.bytes <= cache.max_bytes)
    report = {"passed": passed, "exact_cases": exact, "hits": cache.hits,
              "cache_bytes": cache.bytes, "max_bytes": cache.max_bytes, "lanes": results}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
