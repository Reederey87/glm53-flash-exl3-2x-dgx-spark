#!/usr/bin/env python3
"""Content-prefix warmup: replay standing clients' stable anchors after a restart.

A restart empties the prefix cache; the first turn of every standing client then
re-reads its whole anchor (system prompt + tool defs) cold. Replaying each anchor
through the normal prefill path right after the post-/health warmup pre-warms the
real pages (field-measured elsewhere: +10-18pp hit rate, -18-45% mean TTFT on the
first post-restart turns).

Anchors: drop one file per client under ANCHOR_DIR (default local/warmup-anchors/,
override GLM53_WARMUP_ANCHOR_DIR). Each file's text is sent verbatim as a single
user message with max_tokens=1 — the completion is irrelevant, the prefill is the
warmup. Files are replayed largest-first (biggest anchors claim pool first).

Usage: uv run python local/prefix-warmup.py            (Mac, through the tunnel)
       GLM53_BASE=http://127.0.0.1:8000 python3 ...   (on spark1)
Exit 0 even when ANCHOR_DIR is empty — safe to hook after any restart.
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("GLM53_BASE", "http://127.0.0.1:8000")
MODEL = os.environ.get("GLM53_MODEL", "GLM-5.3-Flash-EXL3")
ANCHOR_DIR = os.environ.get(
    "GLM53_WARMUP_ANCHOR_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "warmup-anchors"),
)
TIMEOUT_S = int(os.environ.get("GLM53_WARMUP_TIMEOUT_S", "900"))


def replay(path: str) -> None:
    text = open(path, encoding="utf-8", errors="replace").read()
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 1,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        usage = json.load(r).get("usage", {})
    print(f"  {os.path.basename(path)}: {usage.get('prompt_tokens', '?')} tokens "
          f"warmed in {time.time() - t0:.1f}s")


def main() -> int:
    if not os.path.isdir(ANCHOR_DIR):
        print(f"prefix-warmup: no anchor dir ({ANCHOR_DIR}) — nothing to do")
        return 0
    files = [os.path.join(ANCHOR_DIR, f) for f in sorted(os.listdir(ANCHOR_DIR))
             if not f.startswith(".") and os.path.isfile(os.path.join(ANCHOR_DIR, f))]
    if not files:
        print(f"prefix-warmup: {ANCHOR_DIR} is empty — nothing to do")
        return 0
    files.sort(key=os.path.getsize, reverse=True)
    print(f"prefix-warmup: replaying {len(files)} anchor(s) from {ANCHOR_DIR}")
    failures = 0
    for f in files:
        try:
            replay(f)
        except Exception as e:  # noqa: BLE001 — warmup must never block a boot
            failures += 1
            print(f"  {os.path.basename(f)}: FAILED ({e})")
    print(f"prefix-warmup: done ({len(files) - failures}/{len(files)} ok)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
