#!/usr/bin/env python3
"""W44: live spec-accept diagnostic (no restart, no-store).

Why lifetime mean accept ~2.5–2.7 while the structured bench is 6.88–7.0.
Four short arms, same max_tokens, W42 no-store so we do not evict the
standing prefix cache:

  S0  structured  thinking off  temp 0     (bench twin)
  P0  prose       thinking off  temp 0     (bench twin)
  S1  structured  thinking on   temp 1.0   effort=max (droid path)
  P1  prose       thinking on   temp 1.0   effort=max (droid path)

Aborts if num_requests_running>0. Run from the Mac through the tunnel:

  uv run python local/w44-spec-accept-probe.py --base http://127.0.0.1:18000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request

MODEL = "GLM-5.3-Flash-EXL3"
STRUCTURED = (
    "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
)
PROSE = (
    "Write a detailed step-by-step explanation of how a hash map works, "
    "including collision handling, resizing, and time complexity. Be thorough."
)


def get(url: str, timeout: float = 15) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def metrics(base: str) -> dict:
    txt = get(f"{base}/metrics")

    def g(pat: str) -> float:
        m = re.search(pat, txt, re.M)
        return float(m.group(1)) if m else 0.0

    pos = {}
    for m in re.finditer(
        r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\} (\S+)',
        txt,
        re.M,
    ):
        pos[int(m.group(1))] = float(m.group(2))
    return {
        "running": g(r'^vllm:num_requests_running\{[^}]*\} (\S+)'),
        "waiting": g(r'^vllm:num_requests_waiting\{[^}]*\} (\S+)'),
        "kv": g(r'^vllm:kv_cache_usage_perc\{[^}]*\} (\S+)'),
        "hits": g(r'^vllm:prefix_cache_hits_total\{[^}]*\} (\S+)'),
        "queries": g(r'^vllm:prefix_cache_queries_total\{[^}]*\} (\S+)'),
        "drafts": g(r'^vllm:spec_decode_num_drafts_total\{[^}]*\} (\S+)'),
        "draft_tok": g(r'^vllm:spec_decode_num_draft_tokens_total\{[^}]*\} (\S+)'),
        "acc": g(r'^vllm:spec_decode_num_accepted_tokens_total\{[^}]*\} (\S+)'),
        "pos": pos,
    }


def spec_delta(a: dict, b: dict) -> dict:
    drafts = b["drafts"] - a["drafts"]
    dtok = b["draft_tok"] - a["draft_tok"]
    acc = b["acc"] - a["acc"]
    pos = []
    for i in range(7):
        d = b["pos"].get(i, 0) - a["pos"].get(i, 0)
        pos.append(round(d / drafts, 4) if drafts else 0.0)
    return {
        "drafts": int(drafts),
        "draft_tokens": int(dtok),
        "accepted": int(acc),
        "accept_ratio": round(acc / dtok, 4) if dtok else None,
        "accepted_per_step": round(acc / drafts, 3) if drafts else None,
        "pos": pos,
        "hit_delta": int(b["hits"] - a["hits"]),
        "query_delta": int(b["queries"] - a["queries"]),
    }


def stream_arm(
    base: str,
    prompt: str,
    *,
    thinking: bool,
    temp: float,
    effort: str | None,
    max_tokens: int,
    timeout: float,
) -> dict:
    kwargs: dict = {"enable_thinking": thinking}
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": kwargs,
        "vllm_xargs": {"skip_writing_prefix_cache": 1},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict = {}
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        http = resp.status
        buf = b""
        while True:
            piece = resp.read(256)
            if not piece:
                break
            buf += piece
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    finish = ch.get("finish_reason") or finish
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        if first is None:
                            first = time.perf_counter()
                        content.append(delta["content"])
                    if delta.get("reasoning"):
                        if first is None:
                            first = time.perf_counter()
                        reasoning.append(delta["reasoning"])
    t1 = time.perf_counter()
    ct = int(usage.get("completion_tokens") or 0)
    ttft = (first - t0) if first else None
    wall = t1 - t0
    dec = ct / (wall - ttft) if ttft is not None and wall > ttft and ct else None
    return {
        "http": http,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "wall_s": round(wall, 3),
        "tok_s": round(dec, 2) if dec is not None else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": ct,
        "finish_reason": finish,
        "content_chars": sum(len(x) for x in content),
        "reasoning_chars": sum(len(x) for x in reasoning),
        "content_head": "".join(content)[:80],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("GLM53_BASE", "http://127.0.0.1:18000"))
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    h = urllib.request.Request(f"{a.base}/health")
    with urllib.request.urlopen(h, timeout=10) as resp:
        if resp.status != 200:
            print("health not 200", resp.status)
            return 2

    m0 = metrics(a.base)
    if m0["running"] > 0 or m0["waiting"] > 0:
        print(json.dumps({"abort": "server busy", **{k: m0[k] for k in ("running", "waiting", "kv")}}))
        return 3

    arms = [
        ("S0-struct-off-t0", STRUCTURED, False, 0.0, None),
        ("P0-prose-off-t0", PROSE, False, 0.0, None),
        ("S1-struct-on-t1-max", STRUCTURED, True, 1.0, "max"),
        ("P1-prose-on-t1-max", PROSE, True, 1.0, "max"),
    ]
    rec = {
        "ts": time.time(),
        "base": a.base,
        "max_tokens": a.max_tokens,
        "pre": {k: m0[k] for k in ("running", "waiting", "kv", "hits", "queries", "drafts", "acc")},
        "arms": [],
    }
    print(
        f"{'arm':<22} {'tok/s':>7} {'ttft':>6} {'acc/step':>8} {'ratio':>6}  pos",
        flush=True,
    )
    for name, prompt, think, temp, effort in arms:
        busy = metrics(a.base)
        if busy["running"] > 0:
            rec["arms"].append({"name": name, "abort": "became busy"})
            print(f"{name:<22} ABORT busy")
            break
        before = metrics(a.base)
        try:
            run = stream_arm(
                a.base,
                prompt,
                thinking=think,
                temp=temp,
                effort=effort,
                max_tokens=a.max_tokens,
                timeout=900,
            )
        except urllib.error.HTTPError as e:
            rec["arms"].append({"name": name, "http": e.code, "err": e.read()[:300].decode("utf-8", "replace")})
            print(f"{name:<22} HTTP {e.code}")
            continue
        after = metrics(a.base)
        spec = spec_delta(before, after)
        row = {"name": name, "thinking": think, "temp": temp, "effort": effort, **run, "spec": spec}
        rec["arms"].append(row)
        pos = " ".join(f"{p:.2f}" for p in spec["pos"])
        print(
            f"{name:<22} {str(run['tok_s'] or '-'):>7} {str(run['ttft_s'] or '-'):>6} "
            f"{str(spec['accepted_per_step'] or '-'):>8} {str(spec['accept_ratio'] or '-'):>6}  {pos}",
            flush=True,
        )

    m1 = metrics(a.base)
    rec["post"] = {k: m1[k] for k in ("running", "waiting", "kv", "hits", "queries")}
    out = a.out or f"/tmp/w44-spec-accept-{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print("wrote", out)
    print(json.dumps({k: rec[k] for k in rec if k != "arms"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
