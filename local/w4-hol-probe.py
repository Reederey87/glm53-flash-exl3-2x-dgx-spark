# LOCAL to this cluster; not part of the vendored upstream kit.
#
# W4 head-of-line probe: fire a long UNIQUE (uncacheable) prefill, then after a
# delay fire a short request and measure its TTFT — the number a tool-call
# client sees when it lands behind a long cold prefill. Run from the Mac:
#
#   uv run python local/w4-hol-probe.py --long-tokens 180000 --delay 20
import argparse
import json
import random
import threading
import time
import urllib.request

WORDS = (
    "ledger harbor quantum velvet orbit lattice ember cascade nickel prairie "
    "sonnet glacier turbine mosaic saffron zephyr basalt meridian copper vault"
).split()


def unique_text(n: int) -> str:
    rng = random.Random(time.time_ns())
    return f"[salt {rng.getrandbits(64):x}] " + " ".join(
        rng.choice(WORDS) for _ in range(int(n * 0.95))
    )


def stream_request(base, model, prompt, max_tokens, timeout, out):
    req = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.monotonic()
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
        if out.get("ttft") is None and chunk.get("choices") and chunk["choices"][0].get("text"):
            out["ttft"] = time.monotonic() - t0
        if chunk.get("usage"):
            out["usage"] = chunk["usage"]
    out["total"] = time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--long-tokens", type=int, default=180000)
    ap.add_argument("--delay", type=float, default=20.0)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    long_out, short_out = {"ttft": None}, {"ttft": None}
    long_prompt = unique_text(args.long_tokens) + "\n\nReply: done."
    short_prompt = unique_text(900) + "\n\nCount from 1 to 30, comma separated."

    t = threading.Thread(
        target=stream_request,
        args=(args.base, args.model, long_prompt, 16, args.timeout, long_out),
    )
    t.start()
    time.sleep(args.delay)
    t_short0 = time.monotonic()
    stream_request(args.base, args.model, short_prompt, 60, args.timeout, short_out)
    short_wall = time.monotonic() - t_short0
    t.join()

    lp = long_out.get("usage", {}).get("prompt_tokens", 0)
    print(json.dumps({
        "long_prompt_tokens": lp,
        "long_ttft_s": round(long_out["ttft"], 1) if long_out["ttft"] else None,
        "long_prefill_tok_s": round(lp / long_out["ttft"], 1) if long_out["ttft"] and lp else None,
        "short_sent_at_s": args.delay,
        "short_ttft_s": round(short_out["ttft"], 2) if short_out["ttft"] else None,
        "short_wall_s": round(short_wall, 2),
        "short_prompt_tokens": short_out.get("usage", {}).get("prompt_tokens", 0),
    }))


if __name__ == "__main__":
    main()
