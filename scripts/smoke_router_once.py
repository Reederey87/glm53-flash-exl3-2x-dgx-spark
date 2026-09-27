"""HTTP/SSE/tool/concurrency smoke; resets prefix cache on an idle endpoint.

Run with UV and --out pointing to a receipt directory. This is a correctness
smoke, not a throughput or cache-retention benchmark. Inspect engine logs too.
"""
import argparse
import concurrent.futures
import json
import os
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
OUT = Path(__file__).parent
API_KEY = os.environ.get("VLLM_API_KEY", "")
TOOLS = [{"type": "function", "function": {"name": "note_page", "description": "Record the requested page.", "parameters": {"type": "object", "properties": {"page": {"type": "string"}}, "required": ["page"]}}}]


def get(path):
    request = urllib.request.Request(BASE + path, headers=headers())
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode()


def headers():
    result = {"Content-Type": "application/json"}
    if API_KEY:
        result["Authorization"] = f"Bearer {API_KEY}"
    return result


def run(label, stream=False, tools=False, long=False):
    token = "SMOKE_" + label.upper()
    content = ("This is reference material. Cargo ballast anchor radar.\n" * 1500 if long else "")
    content += (f"Call note_page with page exactly {token}." if tools else f"Reply with exactly {token}.")
    body = {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 96, "chat_template_kwargs": {"enable_thinking": False}, "stream": stream}
    if tools:
        body.update(tools=TOOLS, tool_choice="required")
    if stream:
        body["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers())
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=240) as response:
        raw = response.read().decode()
        status = response.status
    (OUT / (label + ".response.txt")).write_text(raw)
    calls, text, finish, usage, done = {}, "", None, None, False
    if stream:
        for line in raw.splitlines():
            if line == "data: [DONE]":
                done = True
            elif line.startswith("data: "):
                event = json.loads(line[6:])
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta", {})
                    text += delta.get("content") or ""
                    for call in delta.get("tool_calls") or []:
                        acc = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                        fn = call.get("function", {})
                        acc["name"] += fn.get("name") or ""
                        acc["arguments"] += fn.get("arguments") or ""
    else:
        payload = json.loads(raw)
        choice = payload["choices"][0]
        finish, usage = choice["finish_reason"], payload.get("usage")
        text = choice["message"].get("content") or ""
        calls = {i: call["function"] for i, call in enumerate(choice["message"].get("tool_calls") or [])}
    ok = status == 200 and (not stream or done)
    if tools:
        ok = ok and finish == "tool_calls" and len(calls) == 1
        ok = ok and next(iter(calls.values()))["name"] == "note_page"
        ok = ok and json.loads(next(iter(calls.values()))["arguments"]) == {"page": token}
    else:
        ok = ok and token in text and finish == "stop"
    result = {"label": label, "ok": bool(ok), "wall_s": round(time.monotonic() - start, 3), "finish": finish, "usage": usage, "stream": stream, "tool_calls": calls, "text": text}
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    BASE, OUT = args.base.rstrip("/"), args.out
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics-before.txt").write_text(get("/metrics"))
    # No traffic may be present when flushing cache for the cold test.
    raw = get("/metrics")
    for line in raw.splitlines():
        if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
            assert float(line.rsplit(" ", 1)[1]) == 0, line
    request = urllib.request.Request(BASE + "/reset_prefix_cache", data=b"{}", method="POST", headers=headers())
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 200
    results = [run("long_tool", tools=True, long=True), run("stream_tool", stream=True, tools=True, long=True)]
    results += [run(f"bounded_{i}", long=True) for i in range(20)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run, f"concurrent_{i}", stream=bool(i % 2), tools=i < 2, long=True) for i in range(4)]
        results += [future.result() for future in futures]
    (OUT / "metrics-after.txt").write_text(get("/metrics"))
    get("/health")
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    raise SystemExit(0 if all(row["ok"] for row in results) else 1)
