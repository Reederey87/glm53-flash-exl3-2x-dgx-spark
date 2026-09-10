#!/usr/bin/env python3
"""Capture one prefill window with the in-process torch profiler (task 24 W4 gate).

Runs on the head node against the loopback API while the boot carries
``GLM53_PROFILE_TORCH_DIR``. The window is one cold, unbatched long-context
prefill:

  warmup (unprofiled, nonce A) -> reset prefix cache -> POST /start_profile
  -> one nonce-B request with ``max_tokens=1`` -> POST /stop_profile

Both prompts carry a random leading nonce, so neither can hit the APC and the
profiled request is a genuine cold prefill. ``max_tokens=1`` keeps the decode
tail to a single verify step, which is what makes the capture
prefill-dominated.

The probe fails closed. It refuses to run unless the profiler routes are
mounted (GET /start_profile -> 405), the trace directory is empty, and no
request is in flight. After the capture it re-reads the trace and requires a
trace-derived floor of ``fm_gather_kernel`` launches
(``0.5 * prompt_tokens / max_batched_tokens * moe_layers``), so an APC hit, a
partial capture, or a capture that never entered the grouped path cannot pass
as a measurement. Prefill-vs-decode is guarded by both the fused ``exl3_moe``
time share (``--decode-share-max``) and the fused/gather launch ratio
(``--fused-call-ratio-max``), because on this deployment the fused kernel also
runs during prefill and the time share alone cannot separate the paths.

The receipt records the observed prompt/completion tokens, the APC
``cached_tokens`` when the server reports it, the ``/reset_prefix_cache``
status, and the per-role kernel counts read back from the trace.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_decode_kernel_share as base  # noqa: E402
import audit_prefill_kernel_share as prefill  # noqa: E402

MODEL = os.environ.get("SERVED_MODEL_NAME", "GLM-5.3-Flash-EXL3")
API_KEY = os.environ.get("VLLM_API_KEY", "")
BASE = os.environ.get("GLM53_BASE", "http://127.0.0.1:8000")

PROMPT_SEED = (
    "Explain in thorough, well-organised prose how a modern operating system "
    "schedules threads on a multicore CPU. Cover run queues, priority, "
    "preemption, affinity, load balancing, and the interaction with memory "
    "locality. Then do the same for GPU thread-block scheduling. "
)


def headers(json_content: bool = False) -> dict[str, str]:
    out = {"Content-Type": "application/json"} if json_content else {}
    if API_KEY:
        out["Authorization"] = f"Bearer {API_KEY}"
    return out


def request(method: str, url: str, body: dict | None = None, timeout: float = 60.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, headers=headers(body is not None), method=method
    )
    return urllib.request.urlopen(req, timeout=timeout)


def metrics() -> dict[str, float]:
    with request("GET", BASE + "/metrics") as resp:
        raw = resp.read().decode("utf-8", "replace")
    out: dict[str, float] = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        try:
            out[name.split("{")[0]] = out.get(name.split("{")[0], 0.0) + float(value)
        except ValueError:
            continue
    return out


def profile_mounted() -> int:
    """GET is not a defined method on the profiler routes: 405 == mounted."""
    try:
        request("GET", BASE + "/start_profile", timeout=15.0)
        return 200
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception as exc:  # noqa: BLE001
        print(f"probe FAILED: server unreachable: {exc!r}", file=sys.stderr)
        return 0


def make_prompt(nonce: str, repeats: int) -> str:
    """Leading nonce: a shared suffix could still hit the APC prefix."""
    return f"[run {nonce}] " + PROMPT_SEED * repeats


def run_request(prompt: str, max_tokens: int, timeout: float = 3600.0) -> dict:
    """One streamed completion; returns tokens, TTFT, wall time and completion.

    vLLM sends the ``usage`` block in a final chunk whose ``choices`` is an
    empty list (OpenAI ``stream_options.include_usage`` semantics), so the
    delta lookup must tolerate an empty choices array.
    """
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.monotonic()
    first: float | None = None
    tokens = 0
    usage: dict = {}
    completed = False
    with request("POST", BASE + "/v1/chat/completions", body, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                completed = True
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0] or {}
            if choice.get("finish_reason") is not None:
                completed = True
            delta = choice.get("delta") or {}
            if delta.get("content"):
                tokens += 1
                if first is None:
                    first = time.monotonic()
    return {
        "tokens": tokens,
        "completed": completed,
        "first_token_s": round(first - started, 3) if first else None,
        "wall_s": round(time.monotonic() - started, 3),
        "usage": usage,
    }


def count_grouped(trace_dir: Path) -> dict:
    """Per-role kernel counts read back from the trace(s) in the window."""
    traces = base.find_traces(trace_dir)
    per_trace: list[dict] = []
    for path in traces:
        row = {
            "file": path.name,
            "gather_calls": 0,
            "gateup_calls": 0,
            "down_calls": 0,
            "fused_moe_calls": 0,
            "kernel_count": 0,
            "kernel_us": 0.0,
            "fused_moe_us": 0.0,
        }
        for event in base.iter_trace_events(path):
            if event.get("cat") != "kernel" or not isinstance(event.get("dur"), (int, float)):
                continue
            name = str(event.get("name", ""))
            dur = float(event["dur"])
            row["kernel_count"] += 1
            row["kernel_us"] += dur
            role = prefill.role_of(name)
            if role:
                row[f"{role}_calls"] += 1
            elif base.classify(name) == "fused_moe":
                row["fused_moe_calls"] += 1
                row["fused_moe_us"] += dur
        row["kernel_us"] = round(row["kernel_us"], 3)
        row["fused_moe_us"] = round(row["fused_moe_us"], 3)
        row["fused_moe_share"] = (
            round(row["fused_moe_us"] / row["kernel_us"], 6) if row["kernel_us"] else 0.0
        )
        per_trace.append(row)
    head = max(per_trace, key=lambda r: r["kernel_count"])
    return {"traces": per_trace, "head": head}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.environ.get("GLM53_BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--prompt-repeats", type=int, default=1100,
                    help="~60k tokens at 54.4 tokens/repeat (measured 2026-09-09: "
                         "10 repeats = 544 prompt_tokens); 4412 is ~240k")
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--warmup", action="store_true", help="one unprofiled prefill first (JIT warm)")
    ap.add_argument("--moe-layers", type=int, default=42)
    ap.add_argument("--max-batched-tokens", type=int, default=3584,
                    help="largest possible prefill chunk, for the conservative gather floor")
    ap.add_argument("--min-gather-calls", type=int, default=100)
    ap.add_argument("--decode-share-max", type=float, default=0.30,
                    help="fused exl3_moe share of kernel time; measured prefill 0.128, "
                         "measured decode 0.507 (PR #64). The fused kernel also runs "
                         "during prefill here, so this alone cannot separate the paths")
    ap.add_argument("--fused-call-ratio-max", type=float, default=2.0,
                    help="fused exl3_moe launches / fm_gather launches; measured prefill 1.03")
    ap.add_argument("--cold-cached-fraction-max", type=float, default=0.05,
                    help="fail closed when the server reports more APC hits than this")
    args = ap.parse_args(argv)

    global BASE
    BASE = args.base.rstrip("/")

    if args.moe_layers <= 0 or args.max_batched_tokens <= 0:
        print("probe FAILED: --moe-layers and --max-batched-tokens must be positive", file=sys.stderr)
        return 2
    if not args.trace_dir.is_dir():
        print(f"probe FAILED: trace dir {args.trace_dir} does not exist", file=sys.stderr)
        return 2
    try:
        stale = base.find_traces(args.trace_dir)
    except FileNotFoundError:
        stale = []
    if stale:
        print(
            f"probe FAILED: trace dir already holds {[p.name for p in stale]}; "
            "clear it first so the capture describes this run only",
            file=sys.stderr,
        )
        return 2
    code = profile_mounted()
    if code != 405:
        print(
            f"probe FAILED: GET /start_profile returned {code}, expected 405 "
            "(profiler not armed on this boot)",
            file=sys.stderr,
        )
        return 2
    running = metrics().get("vllm:num_requests_running", 0.0)
    if running > 0:
        print(f"probe FAILED: {running} requests already running", file=sys.stderr)
        return 2

    receipt: dict = {
        "schema": 1,
        "mode": "prefill",
        "prompt_repeats": args.prompt_repeats,
        "max_tokens": args.max_tokens,
        "moe_layers": args.moe_layers,
        "max_batched_tokens": args.max_batched_tokens,
        "min_gather_calls": args.min_gather_calls,
        "decode_share_max": args.decode_share_max,
        "fused_call_ratio_max": args.fused_call_ratio_max,
        "profile_dir": os.environ.get("GLM53_PROFILE_TORCH_DIR"),
        "max_iters_failsafe": os.environ.get("GLM53_PROFILE_MAX_ITERS"),
    }

    if args.warmup:
        warm = run_request(make_prompt(uuid.uuid4().hex[:8], args.prompt_repeats), args.max_tokens)
        receipt["warmup"] = warm
        print(f"warmup complete prompt_tokens={warm['usage'].get('prompt_tokens')} "
              f"wall={warm['wall_s']}s", flush=True)
        if not warm["completed"] or not warm["tokens"]:
            args.out.write_text(json.dumps(receipt, indent=1) + "\n")
            print(f"probe FAILED: warmup did not complete a generation: {warm}", file=sys.stderr)
            return 4

    reset: dict = {"status": None}
    try:
        with request("POST", BASE + "/reset_prefix_cache", {}, timeout=120.0) as resp:
            reset = {"status": resp.status, "body": resp.read().decode("utf-8", "replace")[:200]}
    except urllib.error.HTTPError as exc:
        reset = {"status": exc.code, "body": exc.read().decode("utf-8", "replace")[:200]}
    except Exception as exc:  # noqa: BLE001
        reset = {"status": None, "error": repr(exc)}
    receipt["reset_prefix_cache"] = reset
    print(f"reset_prefix_cache -> {reset['status']}", flush=True)

    prompt = make_prompt(uuid.uuid4().hex[:8], args.prompt_repeats)
    started_profiler = False
    stop_error: str | None = None
    result: dict | None = None
    try:
        with request("POST", BASE + "/start_profile", timeout=60.0) as resp:
            if resp.status != 200:
                raise RuntimeError(f"/start_profile returned {resp.status}")
        started_profiler = True
        print("profiler started; issuing cold prefill", flush=True)
        result = run_request(prompt, args.max_tokens)
    finally:
        if started_profiler:
            try:
                with request("POST", BASE + "/stop_profile", timeout=120.0) as resp:
                    print(f"profiler stopped ({resp.status})", flush=True)
            except Exception as exc:  # noqa: BLE001
                stop_error = repr(exc)
                print(f"ERROR: /stop_profile failed: {exc!r}", file=sys.stderr)
    receipt["request"] = result
    receipt["stop_profile_error"] = stop_error
    if result is None:
        args.out.write_text(json.dumps(receipt, indent=1) + "\n")
        print("probe FAILED: the profiled request did not complete", file=sys.stderr)
        return 4
    if stop_error:
        args.out.write_text(json.dumps(receipt, indent=1) + "\n")
        print(f"probe FAILED: /stop_profile failed: {stop_error}", file=sys.stderr)
        return 4

    # Kineto flushes on stop; poll until the trace parses.
    captured = None
    deadline = time.monotonic() + 240.0
    while True:
        try:
            captured = count_grouped(args.trace_dir)
            break
        except (base.TruncatedTrace, FileNotFoundError, ValueError, OSError) as exc:
            if time.monotonic() >= deadline:
                captured = {"error": repr(exc)}
                break
            time.sleep(10)
    receipt["captured"] = captured

    usage = result["usage"]
    prompt_tokens = usage.get("prompt_tokens")
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    receipt["prompt_tokens"] = prompt_tokens
    receipt["cached_tokens"] = cached_tokens
    # The floor is derived from the *reported* prompt size. Without a completed
    # generation and a positive prompt_tokens count there is no cold-prefill
    # evidence at all, so refuse rather than fall back to the fixed minimum.
    if not result.get("completed") or not result.get("tokens"):
        args.out.write_text(json.dumps(receipt, indent=1) + "\n")
        print(f"probe FAILED: the profiled request did not complete a generation: {result}",
              file=sys.stderr)
        return 3
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        args.out.write_text(json.dumps(receipt, indent=1) + "\n")
        print(
            "probe FAILED: the server reported no prompt_tokens; the capture floor "
            "cannot be established and the prefill is unverified",
            file=sys.stderr,
        )
        return 3
    gather_floor = max(
        args.min_gather_calls,
        int(0.5 * prompt_tokens / args.max_batched_tokens * args.moe_layers),
    )
    receipt["gather_floor"] = gather_floor

    head = captured.get("head") if isinstance(captured, dict) else None
    # The fused kernel also runs during prefill on this deployment (one launch
    # per layer per chunk beside the grouped triple: measured 1470 fused vs 1428
    # grouped per rank), so the time share alone cannot separate prefill from
    # decode. The launch ratio can: a capture that runs the fused path every
    # step cannot stay near 1. Recorded on both outcomes.
    fused_ratio = (
        float(head["fused_moe_calls"]) / max(1.0, float(head["gather_calls"]))
        if head is not None
        else None
    )
    if fused_ratio is not None:
        receipt["fused_call_ratio"] = round(fused_ratio, 4)
    print(
        f"prompt_tokens={prompt_tokens} cached={cached_tokens} "
        f"ttft={result['first_token_s']}s wall={result['wall_s']}s "
        f"gather={head['gather_calls'] if head else 'n/a'} "
        f"gateup={head['gateup_calls'] if head else 'n/a'} "
        f"down={head['down_calls'] if head else 'n/a'} "
        f"floor={gather_floor}"
    )
    args.out.write_text(json.dumps(receipt, indent=1) + "\n")

    if head is None or "error" in captured:
        print(f"probe FAILED: could not read the captured trace: {captured}", file=sys.stderr)
        return 3
    if isinstance(cached_tokens, int) and isinstance(prompt_tokens, int) and prompt_tokens:
        if cached_tokens / prompt_tokens > args.cold_cached_fraction_max:
            print(
                f"probe FAILED: {cached_tokens}/{prompt_tokens} prompt tokens came from the "
                "prefix cache; the prefill was not cold",
                file=sys.stderr,
            )
            return 3
    for role in ("gather", "gateup", "down"):
        calls = int(head[f"{role}_calls"])
        if calls < gather_floor:
            print(
                f"probe FAILED: only {calls} {role} launches captured, need >= {gather_floor} "
                "(APC hit, partial capture, or the grouped path was never entered)",
                file=sys.stderr,
            )
            return 3
    assert fused_ratio is not None  # head is present past this point
    if (
        float(head["fused_moe_share"]) > args.decode_share_max
        or fused_ratio > args.fused_call_ratio_max
    ):
        print(
            f"probe FAILED: fused exl3_moe share {head['fused_moe_share']:.4f} vs max "
            f"{args.decode_share_max} and fused/gather launch ratio {fused_ratio:.2f} vs max "
            f"{args.fused_call_ratio_max}; capture is not prefill-dominated",
            file=sys.stderr,
        )
        args.out.write_text(json.dumps(receipt, indent=1) + "\n")
        return 3
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
