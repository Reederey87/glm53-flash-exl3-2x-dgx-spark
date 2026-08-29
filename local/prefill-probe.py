# LOCAL to this cluster; not part of the vendored upstream kit.
#
# Cold-prefill TTFT probe for the dual-rail A/B (W2 follow-up).
# Sends a UNIQUE (never-cached) long prompt, streams 16 tokens, reports
# time-to-first-token and prefill throughput. Run from the Mac via the tunnel:
#
#   uv run python local/prefill-probe.py --tokens 100000 --runs 2
#
# Uniqueness matters: every run salts the prompt so the prefix cache cannot
# hit; keep other traffic off (the co-batch bug also poisons the timing).
import argparse
import json
import random
import time
import urllib.request

WORDS = (
    "ledger harbor quantum velvet orbit lattice ember cascade nickel prairie "
    "sonnet glacier turbine mosaic saffron zephyr basalt meridian copper vault"
).split()


def unique_prompt(n_tokens: int) -> str:
    rng = random.Random(time.time_ns())
    # ~1 token per word for common short words; oversize slightly, the server
    # reports the real prompt_tokens back.
    body = " ".join(rng.choice(WORDS) for _ in range(int(n_tokens * 0.95)))
    return f"[salt {rng.getrandbits(64):x}] {body}\n\nReply with the single word: done."


def one_run(base: str, model: str, n_tokens: int, timeout: int) -> dict:
    req = {
        "model": model,
        "prompt": unique_prompt(n_tokens),
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.monotonic()
    ttft = None
    usage = {}
    r = urllib.request.urlopen(
        urllib.request.Request(
            f"{base}/completions",
            data=json.dumps(req).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=timeout,
    )
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if ttft is None and chunk.get("choices") and (
            chunk["choices"][0].get("text")
        ):
            ttft = time.monotonic() - t0
        if chunk.get("usage"):
            usage = chunk["usage"]
    total = time.monotonic() - t0
    pt = usage.get("prompt_tokens", 0)
    return {
        "prompt_tokens": pt,
        "ttft_s": round(ttft, 2) if ttft else None,
        "prefill_tok_s": round(pt / ttft, 1) if ttft and pt else None,
        "total_s": round(total, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--tokens", type=int, default=100000)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    results = []
    for i in range(args.runs):
        res = one_run(args.base, args.model, args.tokens, args.timeout)
        results.append(res)
        print(f"run {i + 1}: {json.dumps(res)}", flush=True)
    rates = sorted(r["prefill_tok_s"] for r in results if r["prefill_tok_s"])
    if rates:
        print(f"median prefill tok/s: {rates[len(rates) // 2]}")


if __name__ == "__main__":
    main()
