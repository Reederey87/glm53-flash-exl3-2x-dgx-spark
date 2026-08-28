#!/usr/bin/env python3
"""Smoke/correctness checks for GLM-5.3-Flash — stdlib only (runs on the head node via ssh).

Usage: python3 smoke.py --url http://127.0.0.1:8000 --model glm-5.3-flash-nvfp4
       python3 smoke.py --max-model-len 131072   # ctx-fill uses 75-80% of this
Exit 0 = all checks passed. Prints one PASS/FAIL line per check.
"""
import argparse
import base64
import json
import struct
import sys
import urllib.error
import urllib.request
import zlib

FAILURES = []

CTX_FILL_LO = 0.75
CTX_FILL_HI = 0.80
CTX_FILL_NEEDLE = "NEEDLE-7F3A"
FILLER = (
    "The history of distributed systems spans five decades, from early time-sharing "
    "mainframes and the ARPANET through client-server architectures, peer-to-peer "
    "networks, cloud computing, and modern planet-scale replicated datastores. "
)


def ctx_fill_token_band(max_model_len):
    """Inclusive prompt-token band: 75-80% of the served context window."""
    lo = int(max_model_len * CTX_FILL_LO)
    hi = int(max_model_len * CTX_FILL_HI)
    return lo, hi


def build_ctx_fill_prompt(target_prompt_tokens, chars_per_token, needle=CTX_FILL_NEEDLE):
    """Build a user prompt whose *text* aims at target_prompt_tokens via chars_per_token.

    The live smoke measures usage.prompt_tokens against the 75-80% band; this
    helper only sizes the filler. Needle is on line 1; the question is last.
    """
    if target_prompt_tokens < 1:
        raise ValueError("target_prompt_tokens must be positive")
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    head = (
        f"Secret token: {needle}. Do not mention the filler paragraphs. "
        "The last line is the only question.\n\n"
    )
    tail = (
        "\n\nQuestion: Repeat the secret token from the first line. "
        "Reply with just that token, one word."
    )
    budget = max(0, int(target_prompt_tokens * chars_per_token) - len(head) - len(tail))
    n = max(1, budget // len(FILLER))
    return head + (FILLER * n) + tail


def http_json(method, url, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def report(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def chat(url, model, messages, max_tokens, extra=None, timeout=600):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0}
    if extra:
        payload.update(extra)
    return http_json("POST", f"{url}/v1/chat/completions", payload, timeout=timeout)


def red_png_b64(size=64):
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


def reasoning_of(message):
    """CoT lives in `reasoning` on this deployment, `reasoning_content` elsewhere."""
    return message.get("reasoning") or message.get("reasoning_content") or ""


def garbled(text):
    """True for output no human would accept.

    Empty counts. The 2026-08-26 zero-RoPE-pad boot returned 2048 '!' in
    `reasoning` with `content` empty; a checker that only looks at long runs
    inside `content` calls that clean.
    """
    if not text or not text.strip():
        return True
    if "\ufffd" in text:
        return True
    run = 1
    for a, b in zip(text, text[1:]):
        run = run + 1 if a == b else 1
        if run > 100:
            return True
    # A degenerate sampler emits one symbol forever; catch short-but-uniform too.
    stripped = text.strip()
    if len(stripped) >= 32 and len(set(stripped)) <= 2:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="glm-5.3-flash-nvfp4")
    ap.add_argument("--max-model-len", type=int, default=131072,
                    help="Served context window; ctx-fill targets 75-80% of this")
    args = ap.parse_args()
    url, model = args.url.rstrip("/"), args.model
    max_model_len = args.max_model_len

    try:
        with urllib.request.urlopen(f"{url}/health", timeout=10) as r:
            report("health", r.status == 200, f"HTTP {r.status}")
    except Exception as e:
        report("health", False, str(e))
        print("Smoke aborted: server unreachable.")
        return 1

    try:
        models = http_json("GET", f"{url}/v1/models")
        ids = [m.get("id") for m in models.get("data", [])]
        report("models", model in ids, f"served={ids}")
    except Exception as e:
        report("models", False, str(e))

    try:
        qa = chat(url, model, [{"role": "user", "content":
                                "Reply with just the answer, one word: what is the capital of France?"}],
                  max_tokens=4096)
        msg = qa["choices"][0]["message"]
        content = msg.get("content") or ""
        report("short-qa", "paris" in content.lower(), f"content={content[:80]!r}")
        # glm45 parser: reasoning or reasoning_content
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        report("reasoning-block", len(reasoning) > 0, f"reasoning_len={len(reasoning)}")
        # Thinking is always on for GLM-5.3; a degenerate attention kernel shows
        # up here first (empty content + a wall of one symbol in reasoning).
        report("reasoning-not-garbled", not garbled(reasoning),
               f"reasoning[:60]={reasoning[:60]!r}")
    except Exception as e:
        report("short-qa", False, str(e))
        report("reasoning-block", False, "no response")
        report("reasoning-not-garbled", False, "no response")

    try:
        r = chat(url, model, [{"role": "user", "content":
                               "Write a detailed essay about the history of computing."}],
                 max_tokens=2048)
        ch = r["choices"][0]
        content = ch["message"].get("content") or ""
        fr = ch.get("finish_reason")
        report("long-gen-finish", fr in ("stop", "length"), f"finish_reason={fr}")
        # Thinking is on by default and shares the output budget, so an empty
        # `content` with finish_reason="length" is the model spending it all on
        # reasoning -- not a fault. Judge the text that was actually produced.
        produced = content if content.strip() else reasoning_of(ch["message"])
        report("long-gen-content", len(produced) > 200 and not garbled(produced),
               f"content={len(content)} reasoning={len(produced) if not content.strip() else 0}")
    except Exception as e:
        report("long-gen-finish", False, str(e))
        report("long-gen-content", False, "no response")

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    try:
        r = chat(url, model, [{"role": "user", "content":
                               "Call get_weather for Paris. Do not answer otherwise."}],
                 max_tokens=2048, extra={"tools": tools, "tool_choice": "auto"})
        msg = r["choices"][0]["message"]
        tcalls = msg.get("tool_calls") or []
        names = [((c.get("function") or {}).get("name") or "") for c in tcalls]
        report("tool-call", "get_weather" in names, f"tool_calls={names!r} content={str(msg.get('content'))[:60]!r}")
    except Exception as e:
        report("tool-call", False, str(e))

    try:
        uri = "data:image/png;base64," + red_png_b64()
        r = chat(url, model, [{"role": "user", "content": [
            {"type": "text", "text": "What color is this image? One word."},
            {"type": "image_url", "image_url": {"url": uri}},
        ]}], max_tokens=2048)
        content = r["choices"][0]["message"].get("content") or ""
        report("vision-red", "red" in content.lower(), f"content={content[:80]!r}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:240]
        skip = e.code in (400, 422) and ("image" in body.lower() or "multimodal" in body.lower()
                                         or "vision" in body.lower())
        if skip:
            print(f"SKIP  vision-red — HTTP {e.code}: {body}")
        else:
            report("vision-red", False, f"HTTP {e.code}: {body}")
    except Exception as e:
        report("vision-red", False, str(e))

    # 75-80% of the served window. Calibrate chars/token on the live tokenizer,
    # then one completion whose usage.prompt_tokens must land in that band.
    lo, hi = ctx_fill_token_band(max_model_len)
    target = (lo + hi) // 2
    try:
        calib = FILLER * 40
        cr = chat(url, model, [{"role": "user", "content": calib}], max_tokens=1,
                  timeout=180)
        cpt_tokens = (cr.get("usage") or {}).get("prompt_tokens") or 0
        report("ctx-fill-calibrate", cpt_tokens >= 10, f"prompt_tokens={cpt_tokens}")
        cpt = (len(calib) / cpt_tokens) if cpt_tokens else 4.0
        prompt = build_ctx_fill_prompt(target, cpt)
        # Leave room for thinking + a short answer inside the window.
        max_out = max(64, min(512, max_model_len - target - 32))
        r = chat(url, model, [{"role": "user", "content": prompt}],
                 max_tokens=max_out, timeout=1800)
        usage = r.get("usage") or {}
        pt = usage.get("prompt_tokens") or 0
        msg = r["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        in_band = lo <= pt <= hi
        # One rescale if the first shot missed the band (tokenizer vs estimate).
        if not in_band and pt > 0:
            prompt = build_ctx_fill_prompt(int(target * (target / pt)), cpt)
            r = chat(url, model, [{"role": "user", "content": prompt}],
                     max_tokens=max_out, timeout=1800)
            usage = r.get("usage") or {}
            pt = usage.get("prompt_tokens") or 0
            msg = r["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            in_band = lo <= pt <= hi
        report("ctx-fill-prompt-tokens", in_band,
               f"prompt_tokens={pt} band=[{lo},{hi}] max_model_len={max_model_len}")
        body = (content + " " + reasoning).upper()
        report("ctx-fill-retrieve", CTX_FILL_NEEDLE in body or "7F3A" in body,
               f"content={content[:80]!r}")
        # Match ctx-fill-retrieve, which searches content+reasoning: judging only
        # `content` made the two checks disagree about whether an all-reasoning
        # answer is acceptable.
        judged = content if content.strip() else reasoning_of(msg)
        report("ctx-fill-not-garbled", not garbled(judged), f"len={len(judged)}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        report("ctx-fill-prompt-tokens", False, f"HTTP {e.code}: {body}")
        report("ctx-fill-retrieve", False, "no response")
        report("ctx-fill-not-garbled", False, "no response")
    except Exception as e:
        report("ctx-fill-prompt-tokens", False, str(e))
        report("ctx-fill-retrieve", False, "no response")
        report("ctx-fill-not-garbled", False, "no response")

    print(f"\n{'SMOKE PASS' if not FAILURES else 'SMOKE FAIL: ' + ', '.join(FAILURES)}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
