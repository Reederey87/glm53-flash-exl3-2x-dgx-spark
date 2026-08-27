#!/usr/bin/env python3
"""Concurrency throughput probe — stdlib only (runs on the head node via ssh).

Fires N concurrent chat completions per rep, measures wall time, reports
sum(completion_tokens)/wall as tok/s. Prints one JSON object per rep (JSONL).
"""
import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PARA = ("The history of distributed systems spans five decades, from early time-sharing "
        "mainframes and the ARPANET through client-server architectures, peer-to-peer "
        "networks, cloud computing, and modern planet-scale replicated datastores. Each era "
        "renegotiated the same fundamental trade-offs: consistency against availability, "
        "latency against throughput, simplicity against fault tolerance. ")
PROMPT = ("Read the following background text, then write a long, detailed technical essay "
          "expanding on its themes with concrete examples and historical detail.\n\n" + PARA * 8)


def one_request(url, model, max_tokens, idx):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": f"[essay request {idx}] " + PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        body = json.loads(resp.read().decode())
    dt = time.monotonic() - t0
    usage = body.get("usage") or {}
    return {"idx": idx, "wall_s": round(dt, 2),
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="glm-5.3-flash-nvfp4")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    rc = 0
    for rep in range(1, args.reps + 1):
        t0 = time.monotonic()
        errors = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(one_request, url, args.model, args.max_tokens,
                              rep * 1000 + i)
                    for i in range(args.concurrency)]
            results = []
            for f in futs:
                try:
                    results.append(f.result())
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e))
        wall = time.monotonic() - t0
        total_completion = sum(r["completion_tokens"] for r in results)
        rec = {
            "label": args.label, "concurrency": args.concurrency, "rep": rep,
            "wall_s": round(wall, 2), "completion_tokens": total_completion,
            "tok_per_s": round(total_completion / wall, 2) if wall > 0 else 0,
            "ok_requests": len(results), "errors": errors,
        }
        print(json.dumps(rec), flush=True)
        if errors or len(results) != args.concurrency:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
