#!/usr/bin/env python3
"""Drive a C4 decode workload and capture one torch-profiler window.

Runs on the head node against the loopback API. Preflight fails closed unless
the profiler routes are mounted (GET /start_profile -> 405; a 404 means the
boot did not carry ``GLM53_PROFILE_TORCH_DIR``). The window contains decode
steps only: the profiler starts after all four streams have emitted their first
token and stops when they finish (or ``max_iterations`` fires as a fail-safe).

Writes a JSON receipt with the spec-decode step count observed inside the
window, so the auditor's per-step denominators are traceable. The receipt also
carries the *captured* step count read back from the profiler trace: the vLLM
counters bracket ``/start_profile``..``/stop_profile`` but keep counting if the
profiler auto-stops early, so only the trace proves how much was captured.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_decode_kernel_share as audit  # noqa: E402

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


def wait_concurrency(seqs: int, timeout: float = 90.0) -> tuple[float, list[float]]:
    """Wait for ``seqs`` running requests with an empty wait queue.

    Admission is not instantaneous even with a warm prefix cache, so a single
    sample races. Returns the observed running values for the receipt.
    """
    samples: list[float] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = metrics()
        running = snap.get("vllm:num_requests_running", 0.0)
        waiting = snap.get("vllm:num_requests_waiting", 0.0)
        samples.append(running)
        if running >= seqs and waiting == 0.0:
            return running, samples
        time.sleep(0.2)
    return (samples[-1] if samples else 0.0), samples


def stream_worker(index: int, prompt: str, max_tokens: int, state: dict, barrier: threading.Barrier) -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    first = None
    tokens = 0
    try:
        with request("POST", BASE + "/v1/chat/completions", body, timeout=1800.0) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    tokens += 1
                    if first is None:
                        first = time.monotonic()
                        # Publish the first-token time immediately: the main
                        # thread gates /start_profile on every stream having
                        # cleared prefill, so it must be visible mid-stream.
                        state[index] = {"ok": True, "tokens": tokens, "first_ts": first}
                        barrier.wait(timeout=300.0)
        state[index] = {"ok": True, "tokens": tokens, "first_ts": first}
    except Exception as exc:  # noqa: BLE001
        state[index] = {"ok": False, "error": repr(exc), "tokens": tokens, "first_ts": first}
        try:
            barrier.abort()
        except Exception:  # noqa: BLE001
            pass


def stale_traces(trace_dir: Path) -> list[Path]:
    """Traces already present before this probe runs.

    The capture floor must describe *this* profiling run, so a leftover trace
    from an earlier run must never be able to satisfy it.
    """
    return sorted(
        p for p in trace_dir.rglob("*") if p.is_file() and str(p).endswith(".pt.trace.json.gz")
    )


def captured_engine_steps(trace_dir: Path, moe_layers: int) -> dict:
    """Count the decode steps that are actually present in the profiler trace.

    Every decode step launches one fused ``exl3_moe`` kernel per MoE layer, so
    ``fused_moe_calls / moe_layers`` is the step count the auditor will see.
    Each trace is a full view of its own rank, so the maximum across traces is
    the captured count (never the sum). Callers must have verified the
    directory held no traces before the run (``stale_traces``).
    """
    traces = stale_traces(trace_dir)
    if not traces:
        raise RuntimeError(f"no *.pt.trace.json.gz under {trace_dir}")
    if moe_layers <= 0:
        raise ValueError("--moe-layers must be positive")
    per_trace = []
    best = 0.0
    for path in traces:
        calls = 0
        for event in audit.iter_trace_events(path):
            if event.get("cat") != "kernel":
                continue
            if audit.classify(str(event.get("name", ""))) == "fused_moe":
                calls += 1
        steps = calls / moe_layers
        best = max(best, steps)
        per_trace.append(
            {"file": path.name, "fused_moe_calls": calls, "engine_steps": round(steps, 3)}
        )
    return {
        "moe_layers": moe_layers,
        "traces": per_trace,
        "fused_moe_calls": max(int(t["fused_moe_calls"]) for t in per_trace),
        "engine_steps": round(best, 3),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.environ.get("GLM53_BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--prompt-repeats", type=int, default=110, help="~8000 tokens at ~14 words/repeat")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument(
        "--min-steps",
        type=int,
        default=60,
        help="minimum decode steps that must be present in the captured trace",
    )
    ap.add_argument(
        "--trace-dir",
        type=Path,
        help="host directory holding the profiler trace(s); required unless --dry-run",
    )
    ap.add_argument("--moe-layers", type=int, default=42,
                    help="MoE layers per decode step, to convert fused-kernel calls to steps")
    ap.add_argument("--warmup", action="store_true", help="one unprofiled C4 burst first")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the workload only; no /start_profile or /stop_profile")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    global BASE
    BASE = args.base.rstrip("/")

    if args.moe_layers <= 0:
        print("probe FAILED: --moe-layers must be positive", file=sys.stderr)
        return 2

    if not args.dry_run:
        if args.trace_dir is None:
            print(
                "probe FAILED: --trace-dir is required outside --dry-run; the captured "
                "step count must come from the trace, not from the counters",
                file=sys.stderr,
            )
            return 2
        if not args.trace_dir.is_dir():
            print(f"probe FAILED: trace dir {args.trace_dir} does not exist", file=sys.stderr)
            return 2
        stale = stale_traces(args.trace_dir)
        if stale:
            print(
                f"probe FAILED: trace dir {args.trace_dir} already holds "
                f"{[p.name for p in stale]}; clear it first so the step floor "
                "describes this capture only",
                file=sys.stderr,
            )
            return 2

    # Preflight: profiler routes mounted? GET is not defined by the router, so a
    # mounted route answers 405 and an unarmed boot answers 404.
    if not args.dry_run:
        try:
            request("GET", BASE + "/start_profile", timeout=15.0)
            code = 200
        except urllib.error.HTTPError as exc:
            code = exc.code
        except Exception as exc:  # noqa: BLE001
            print(f"probe FAILED: server unreachable: {exc!r}", file=sys.stderr)
            return 2
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

    prompt = PROMPT_SEED * args.prompt_repeats
    if args.warmup:
        st: dict[int, dict] = {}
        bar = threading.Barrier(args.seqs)
        threads = [threading.Thread(target=stream_worker, args=(i, prompt, 64, st, bar), daemon=True) for i in range(args.seqs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=600.0)
        if not all(v.get("ok") for v in st.values()):
            print(f"probe FAILED: warmup stream error: {st}", file=sys.stderr)
            return 2
        print("warmup complete", flush=True)

    state: dict[int, dict] = {}
    barrier = threading.Barrier(args.seqs)
    threads = [
        threading.Thread(target=stream_worker, args=(i, prompt, args.max_tokens, state, barrier), daemon=True)
        for i in range(args.seqs)
    ]
    for t in threads:
        t.start()

    started = False
    stop_error: str | None = None
    before: dict[str, float] | None = None
    after: dict[str, float] | None = None
    conc_samples: list[float] = []
    try:
        # All streams are past prefill only once every worker has published its
        # first token: a running-request metric alone does not prove that.
        deadline = time.monotonic() + 900.0
        while time.monotonic() < deadline:
            if len(state) == args.seqs and all(v.get("first_ts") for v in state.values()):
                break
            if any(not v.get("ok", True) for v in state.values() if v):
                raise RuntimeError(f"stream failed before profiling: {state}")
            time.sleep(0.25)
        else:
            raise RuntimeError(
                f"timed out waiting for all {args.seqs} streams to start decoding "
                f"(published {sorted(state)})"
            )

        live, conc_samples = wait_concurrency(args.seqs)
        if live < args.seqs:
            raise RuntimeError(
                f"only {live} of {args.seqs} requests running at profile start "
                f"(samples={conc_samples})"
            )
        # Sample the counters immediately before /start_profile so the deltas
        # bracket the capture as tightly as the API allows.
        before = metrics()
        if not args.dry_run:
            with request("POST", BASE + "/start_profile", timeout=60.0) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"/start_profile returned {resp.status}")
            started = True
            print(f"profiler started with {int(live)} running requests", flush=True)
        else:
            print(f"dry-run: {int(live)} requests decoding, profiler not started", flush=True)

        for t in threads:
            t.join(timeout=1800.0)
        if any(t.is_alive() for t in threads):
            raise RuntimeError("streams did not finish inside the timeout")
    finally:
        if started:
            try:
                with request("POST", BASE + "/stop_profile", timeout=120.0) as resp:
                    print(f"profiler stopped ({resp.status})", flush=True)
            except Exception as exc:  # noqa: BLE001
                stop_error = repr(exc)
                print(f"ERROR: /stop_profile failed: {exc!r}", file=sys.stderr)
        if before is not None:
            after = metrics()

    def delta(name: str) -> float:
        if before is None or after is None:
            return 0.0
        return after.get(name, 0.0) - before.get(name, 0.0)

    drafts = delta("vllm:spec_decode_num_drafts_total")
    accepted = delta("vllm:spec_decode_num_accepted_tokens_total")
    draft_tokens = delta("vllm:spec_decode_num_draft_tokens_total")
    engine_steps = drafts / args.seqs if args.seqs else 0.0
    captured = None
    if not args.dry_run:
        # Kineto flushes the trace when the profiler stops; poll until it parses
        # so a still-growing file is never mistaken for a short capture.
        deadline = time.monotonic() + 240.0
        while True:
            try:
                captured = captured_engine_steps(args.trace_dir, args.moe_layers)
                break
            except (audit.TruncatedTrace, RuntimeError, ValueError, OSError) as exc:
                if time.monotonic() >= deadline:
                    captured = {"error": repr(exc)}
                    break
                time.sleep(10)
    receipt = {
        "schema": 3,
        "dry_run": args.dry_run,
        "seqs": args.seqs,
        "prompt_repeats": args.prompt_repeats,
        "max_tokens": args.max_tokens,
        "streams": {str(k): v for k, v in sorted(state.items())},
        "concurrency_samples": conc_samples,
        "counter_scope": (
            "vLLM counters sampled immediately before /start_profile and right "
            "after /stop_profile; they keep counting if the profiler auto-stops "
            "on max_iterations, so they are an upper bound on captured work"
        ),
        "drafts_delta_per_request_sum": drafts,
        "engine_decode_steps_estimate": round(engine_steps, 3),
        "accepted_tokens_delta": accepted,
        "draft_tokens_delta": draft_tokens,
        "acceptance": (accepted / draft_tokens) if draft_tokens else None,
        "captured": captured,
        "min_steps": args.min_steps,
        "stop_profile_error": stop_error,
        "max_iters_failsafe": os.environ.get("GLM53_PROFILE_MAX_ITERS"),
        "profile_dir": os.environ.get("GLM53_PROFILE_TORCH_DIR"),
    }
    args.out.write_text(json.dumps(receipt, indent=1) + "\n")
    print(
        f"engine_decode_steps_estimate~{engine_steps:.0f} "
        f"(per-request draft sum {drafts:.0f} over {args.seqs} streams) "
        f"captured_steps={captured.get('engine_steps') if captured else 'n/a'} "
        f"accepted={accepted:.0f} acceptance={receipt['acceptance']}"
    )
    failed = sorted(k for k, v in state.items() if not v.get("ok"))
    if failed:
        print(
            f"probe FAILED: streams {failed} did not complete: {[state[k] for k in failed]}",
            file=sys.stderr,
        )
        return 4
    if stop_error:
        print(f"probe FAILED: /stop_profile failed: {stop_error}", file=sys.stderr)
        return 4
    if args.dry_run:
        print(f"wrote {args.out}")
        return 0
    if captured is None or "error" in captured:
        print(f"probe FAILED: could not read the captured trace: {captured}", file=sys.stderr)
        return 3
    if captured["engine_steps"] < args.min_steps:
        print(
            f"probe FAILED: the trace holds only {captured['engine_steps']} engine decode "
            f"steps, need >= {args.min_steps} (the profiler may have auto-stopped on "
            f"max_iterations={os.environ.get('GLM53_PROFILE_MAX_ITERS')})",
            file=sys.stderr,
        )
        return 3
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
