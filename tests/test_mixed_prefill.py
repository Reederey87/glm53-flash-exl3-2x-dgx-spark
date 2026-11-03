from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import pytest

from test_scripts import sse_endpoint  # Reuse the real-HTTP fault fixture.


@pytest.fixture
def mixed():
    path = Path(__file__).resolve().parents[1] / "experiments/mixed-prefill/measure.py"
    spec = importlib.util.spec_from_file_location("tested_mixed_prefill", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fault", [None, "usage", "done", "short", "reset", "bool", "json", "error"])
def test_stream_uses_token_counts_and_retains_incomplete_evidence(mixed, sse_endpoint, fault):
    base, events, server = sse_endpoint
    events.extend([
        {"choices": [{"text": "", "finish_reason": None}],
         "usage": {"prompt_tokens": 2, "completion_tokens": 2}},
        {"choices": [{"text": "", "finish_reason": "length"}],
         "usage": {"prompt_tokens": 2, "completion_tokens": 4}},
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 4}}, "[DONE]",
    ])
    if fault == "usage":
        events.pop(2)
    elif fault == "done":
        events.pop()
    elif fault in ("short", "reset", "bool"):
        events[1]["usage"]["completion_tokens"] = {"short": 3, "reset": 1, "bool": True}[fault]
    elif fault == "json":
        events[1] = "not JSON"
    elif fault == "error":
        events[1] = {"error": "failed"}
    trace = mixed.Trace("decode-0", "prompt", 2, 4)
    notifications = []
    mixed.stream(base, "model", [1000, 1001], trace, time.perf_counter(), 2,
                 notify=lambda: notifications.append(len(trace.arrivals)))
    assert trace.started_s is not None and trace.ended_s is not None
    assert trace.arrivals[0][0] == 2  # Empty visible text still contains tokens.
    assert bool(trace.errors) == (fault is not None)
    if fault is None:
        assert [n for n, _ in trace.arrivals] == [2, 4]
        assert notifications == [1, 2, 2]
    assert server.calls[0][1]["stream_options"]["continuous_usage_stats"] is True


def complete_trace(mixed, role, start, arrivals):
    budget = arrivals[-1][0]
    return mixed.Trace(role, "prompt", 10, budget, started_s=start,
                       ended_s=arrivals[-1][1] + 0.01, arrivals=arrivals,
                       final_usage={"prompt_tokens": 10, "completion_tokens": budget},
                       finish_reason="length", done=True)


def test_overlap_uses_full_update_gaps_and_bundle_counts(mixed):
    short = complete_trace(mixed, "decode-0", 0, [(4, 1), (7, 3), (9, 4), (12, 8)])
    long = complete_trace(mixed, "prefill", 2, [(1, 6)])
    result = mixed.analyze([short], long)
    assert result["ok"]
    row = result["short"][0]
    assert row["observed_tokens_inside_prefill"] == 5
    assert row["observed_tokens_per_prefill_second"] == 1.25
    assert row["overlapping_update_gap_max_s"] == 4  # Full crossing gap, not clipped to 2.
    assert row["overlapping_update_gap_p95_s"] == pytest.approx(3.8)


@pytest.mark.parametrize("arrivals", [
    [(4, 1), (7, 1.5), (9, 1.8)],  # Already done before prefill submission.
    [(4, 1), (7, 7), (9, 8)],  # No progress while long request is outstanding.
    [(4, 3), (7, 4), (9, 8)],  # Not decoding before prefill submission.
    [(4, 1), (7, 3), (9, 4)],  # Decode ended before long request's first token.
    [(4, 1), (7, 3), (9, 8)],  # Only one update in the interference interval.
])
def test_queue_only_or_unproven_overlap_is_invalid(mixed, arrivals):
    result = mixed.analyze([complete_trace(mixed, "decode-0", 0, arrivals)],
                           complete_trace(mixed, "prefill", 2, [(1, 6)]))
    assert not result["ok"]
    assert "overlap" in " ".join(result["errors"])


def test_incomplete_trace_cannot_become_valid_by_erasing_errors(mixed):
    short = complete_trace(mixed, "decode-0", 0, [(4, 1), (7, 3), (9, 4), (12, 8)])
    short.done = False
    assert not mixed.analyze([short], complete_trace(mixed, "prefill", 2, [(1, 6)]))["ok"]


@pytest.mark.parametrize("before,after", [(None, 1), (1, None), (2, 1), (0, float("nan")), (False, 1)])
def test_invalid_counter_window_rejected(mixed, before, after):
    with pytest.raises(ValueError):
        mixed.counter_delta({"counter": before}, {"counter": after}, "counter")


@pytest.fixture
def interleaved_server():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self.reply({})
            with server.lock:
                body = (f"vllm:request_success_total {server.finished}\n"
                        f"vllm:prefix_cache_hits_total {server.cache_hits}\n"
                        f"vllm:num_requests_running {server.active}\n"
                        "vllm:num_requests_waiting 0\n"
                        f"vllm:spec_decode_num_accepted_tokens_total {server.short_finished * 4}\n"
                        f"vllm:spec_decode_num_draft_tokens_total {server.short_finished * 6}\n"
                        f"vllm:spec_decode_num_drafts_total {server.short_finished * 2}\n").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with server.lock:
                server.calls.append((self.path, payload, self.headers.get("Authorization")))
            if self.path == "/tokenize":
                return self.reply({"tokens": [1000 + b for b in payload["messages"][0]["content"].encode()[:16]]})
            prompt, budget = payload["prompt"], payload["max_tokens"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def emit(count, finish=None, final=False):
                event = {"choices": [] if final else [{"text": "", "finish_reason": finish}],
                         "usage": {"prompt_tokens": len(prompt), "completion_tokens": count}}
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()

            if budget > 1:
                with server.condition:
                    if server.active == 0:
                        server.prefill = threading.Event()
                        server.long_sent = threading.Event()
                        server.inside = 0
                    server.active += 1
                emit(1)
                emit(3)
                assert server.prefill.wait(3)
                for n in (5, 7):
                    emit(n)
                    with server.condition:
                        server.inside += 1
                        server.condition.notify_all()
                    time.sleep(0.01)
                assert server.long_sent.wait(3)
                time.sleep(0.025)
                emit(budget, "length")
            else:
                server.prefill.set()
                with server.condition:
                    assert server.condition.wait_for(lambda: server.inside >= 2 * server.active, 3)
                if server.mode == "queued":
                    server.long_sent.set()
                    with server.condition:
                        assert server.condition.wait_for(lambda: server.active == 0, 3)
                with server.lock:
                    if server.mode == "cachehit":
                        server.cache_hits += 1
                    elif server.mode == "foreign":
                        server.finished += 1
                time.sleep(0.025)
                emit(1, "length")
                server.long_sent.set()
            with server.condition:
                server.finished += 1
                if budget > 1:
                    server.short_finished += 1
                    server.active -= 1
                server.condition.notify_all()
            emit(budget, final=True)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.calls = []
    server.lock = threading.RLock()
    server.condition = threading.Condition(server.lock)
    server.active = server.finished = server.short_finished = server.cache_hits = 0
    server.mode = "good"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_real_http_coordinator_waits_for_decodes_and_records_max_reasoning(mixed, interleaved_server, tmp_path, monkeypatch):
    base, server = interleaved_server
    out = tmp_path / "mixed.json"
    monkeypatch.setenv("VLLM_API_KEY", "test-auth-not-for-receipts")
    status = mixed.main(["--base", base, "--out", str(out), "--decoders", "1,3",
                         "--prefill-lengths", "32", "--repetitions", "1", "--decode-tokens", "16",
                         "--ready-tokens", "2", "--ready-timeout", "3", "--timeout", "5"])
    result = json.loads(out.read_text())
    assert status == 0 and result["complete"] and result["ok"]
    assert len(result["warmups"]) == len(result["cells"]) == 2
    for cell in result["warmups"] + result["cells"]:
        assert cell["ready_barrier_passed"] and cell["ok"]
        assert cell["counter_deltas"][mixed.SUCCESS] == cell["decoders"] + 1
        assert cell["counter_deltas"][mixed.speed.CACHE_HITS] == 0
        assert all(row["observed_overlap"] for row in cell["analysis"]["short"])
        assert cell["acceptance"]["mean_acceptance_length"] == 3
    for path, payload, auth in server.calls:
        assert auth == "Bearer test-auth-not-for-receipts"
        if path == "/tokenize":
            assert payload["chat_template_kwargs"] == {"reasoning_effort": "max"}
    assert "test-auth-not-for-receipts" not in out.read_text()


@pytest.mark.parametrize("mode,reason", [("queued", "overlap"), ("cachehit", "prefix_cache_hits"),
                                        ("foreign", "request_success")])
def test_real_http_completed_streams_do_not_hide_invalid_measurement(mixed, interleaved_server, tmp_path, mode, reason):
    base, server = interleaved_server
    server.mode = mode
    out = tmp_path / "invalid.json"
    status = mixed.main(["--base", base, "--out", str(out), "--decoders", "1", "--prefill-lengths", "32",
                         "--repetitions", "1", "--decode-tokens", "16", "--ready-tokens", "2",
                         "--ready-timeout", "3", "--timeout", "5"])
    result = json.loads(out.read_text())
    assert status == 1 and not result["complete"] and not result["ok"]
    cell = result["warmups"][0]
    assert all(t["done"] and not t["errors"] for t in cell["requests"])
    assert reason in " ".join(cell["analysis"]["errors"])


def test_failed_cell_is_retained_and_stops_the_grid(mixed, tmp_path, monkeypatch):
    def fail(args, rep, count, length, budget, cell):
        cell.update(ok=False, complete=True, analysis={"errors": ["no observed overlap"]})
    monkeypatch.setattr(mixed, "measure_cell", fail)
    monkeypatch.setattr(mixed.speed, "_request_json", lambda *a: (200, {}))
    out = tmp_path / "failed.json"
    assert mixed.main(["--out", str(out)]) == 1
    receipt = json.loads(out.read_text())
    assert not receipt["complete"] and not receipt["ok"] and not receipt["cells"]
    assert receipt["warmups"][0]["analysis"]["errors"] == ["no observed overlap"]


def test_rejects_receipt_overwrite_and_queue_only_admission(mixed, tmp_path):
    out = tmp_path / "existing.json"
    out.write_text("original")
    with pytest.raises(SystemExit):
        mixed.main(["--out", str(out)])
    assert out.read_text() == "original"
    with pytest.raises(SystemExit):
        mixed.main(["--out", str(tmp_path / "new.json"), "--decoders", "4", "--max-seqs", "4"])
