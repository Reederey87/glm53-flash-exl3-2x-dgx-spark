from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tested_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"tested_{name}"] = module
    spec.loader.exec_module(module)
    return module


def test_verify_kv_pool_parses_boot_line() -> None:
    module = load_module("verify_kv_pool")
    text = "\n".join(
        [
            "INFO 10-17 06:00:00 gpu_worker.py:700] Available KV cache memory: 10.50 GiB",
            "INFO 10-17 06:00:00 gpu_worker.py:701] GPU KV cache size: 2,097,152 tokens",
            "INFO 10-17 06:00:00 gpu_worker.py:702] # GPU blocks: 8,192",
        ]
    )
    pool = module.parse_kv_log(text)
    assert pool is not None
    assert pool.tokens == 2_097_152
    assert pool.blocks == 8_192
    assert pool.gib == pytest.approx(10.50)


def test_verify_kv_pool_uses_last_boot() -> None:
    module = load_module("verify_kv_pool")
    text = "\n".join(
        [
            "GPU KV cache size: 1,000,000 tokens",
            "GPU KV cache size: 2,100,000 tokens",
        ]
    )
    assert module.parse_kv_log(text).tokens == 2_100_000


def test_verify_kv_pool_returns_none_without_line() -> None:
    module = load_module("verify_kv_pool")
    assert module.parse_kv_log("server started, no cache line") is None


def test_spec_acceptance_parses_labeled_counters() -> None:
    module = load_module("spec_acceptance")
    text = "\n".join(
        [
            "# HELP vllm:spec_decode_num_accepted_tokens_total accepted",
            'vllm:spec_decode_num_accepted_tokens_total{model_name="x"} 700.0',
            'vllm:spec_decode_num_draft_tokens_total{model_name="x"} 1000.0',
            'vllm:spec_decode_num_drafts_total{model_name="x"} 100.0',
        ]
    )
    counters = module.parse_metrics(text)
    assert counters[module.ACCEPTED] == 700.0
    assert counters[module.DRAFTED] == 1000.0
    assert counters[module.DRAFTS] == 100.0


def test_spec_acceptance_computes_deltas_and_rates() -> None:
    module = load_module("spec_acceptance")
    before = {
        module.ACCEPTED: 100.0,
        module.DRAFTED: 200.0,
        module.DRAFTS: 20.0,
    }
    after = {
        module.ACCEPTED: 800.0,
        module.DRAFTED: 1200.0,
        module.DRAFTS: 120.0,
    }
    result = module.compute(before, after, 30.0)
    assert result.accepted == 700.0
    assert result.drafted == 1000.0
    assert result.drafts == 100.0
    assert result.mean_acceptance_length == pytest.approx(8.0)
    assert result.draft_acceptance_rate == pytest.approx(0.7)
    assert result.resets == 0


def test_spec_acceptance_handles_counter_reset() -> None:
    module = load_module("spec_acceptance")
    result = module.compute(
        {module.ACCEPTED: 900.0, module.DRAFTED: 1200.0, module.DRAFTS: 120.0},
        {module.ACCEPTED: 10.0, module.DRAFTED: 20.0, module.DRAFTS: 2.0},
        1.0,
    )
    assert result.resets == 3
    assert result.accepted == 10.0


@pytest.mark.parametrize(
    "script",
    ["verify_kv_pool.py", "smoke.py", "spec_acceptance.py", "speed_matrix.py", "probe_prefix_cache.py", "tailscale_proxy.py", "validate_hf_snapshot.py"],
)
def test_scripts_have_help(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_scripts_are_stdlib_only() -> None:
    allowed = {
        "argparse",
        "dataclasses",
        "datetime",
        "json",
        "pathlib",
        "re",
        "subprocess",
        "sys",
        "time",
        "typing",
        "urllib",
        "__future__",
        "concurrent",
        "asyncio",
        "logging",
        "math",
        "random",
        "statistics",
        "os",
        "hashlib",
    }
    for path in SCRIPTS.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        imports = [
            line.split()[1].split(".")[0]
            for line in source.splitlines()
            if line.startswith("import ") or line.startswith("from ")
        ]
        for module_name in imports:
            assert module_name in allowed, f"{path.name}: unexpected import {module_name}"


def test_smoke_prefix_verdict_uses_metrics_fallback() -> None:
    module = load_module("smoke")
    check = module.prefix_cache_verdict(200, 200, 6144.0, None, False, "block=4864/mamba=256")
    assert check.status == "pass"
    assert "hits +6144" in check.detail
    assert "field absent" in check.detail

    check = module.prefix_cache_verdict(200, 200, 1024.0, 1024, True, "block=4864/mamba=256")
    assert check.status == "pass"

    check = module.prefix_cache_verdict(200, 200, 0.0, 0, True, "block=4864/mamba=256")
    assert check.status == "warn"

    check = module.prefix_cache_verdict(200, 200, 0.0, None, False, "block=4864/mamba=256")
    assert check.status == "warn"
    assert "enable-prompt-tokens-details" in check.detail


def test_smoke_parses_cache_geometry_and_hits() -> None:
    module = load_module("smoke")
    text = "\n".join(
        [
            'vllm:cache_config_info{block_size="4864",mamba_block_size="256",'
            'enable_mamba_fine_grained_prefix_cache="False"} 1.0',
            'vllm:prefix_cache_hits_total{engine="0"} 829.0',
        ]
    )
    hits, geometry = module._parse_cache_metrics(text)
    assert hits == 829.0
    assert geometry == "block=4864/mamba=256/fine_grained=False"


def test_prefix_probe_parses_cache_geometry() -> None:
    module = load_module("probe_prefix_cache")
    text = "\n".join(
        [
            'vllm:cache_config_info{block_size="4864",mamba_block_size="256",'
            'enable_mamba_fine_grained_prefix_cache="False",prefix_match_unit="None",'
            'enable_prefix_caching="True"} 1.0',
            "vllm:prefix_cache_queries_total 1870.0",
            "vllm:prefix_cache_hits_total 829.0",
        ]
    )
    counters, geometry, _ = module.parse_metrics(text)
    assert counters[module.QUERIES] == 1870.0
    assert counters[module.HITS] == 829.0
    assert geometry.block_size == 4864
    assert geometry.mamba_block_size == 256
    assert geometry.fine_grained_mamba == "False"
    assert geometry.enable_prefix_caching == "True"


def test_speed_matrix_prompt_is_unique_and_exact() -> None:
    module = load_module("speed_matrix")
    prompt = module.make_prompt(8192, module.random.Random(7))
    assert len(prompt) == 8192
    assert all(1000 <= token < 150000 for token in prompt)
    other = module.make_prompt(8192, module.random.Random(8))
    assert prompt[:64] != other[:64]


def test_speed_matrix_acceptance_from_counter_deltas() -> None:
    module = load_module("speed_matrix")
    before = {module.ACCEPTED: 100.0, module.DRAFTED: 200.0, module.DRAFTS: 20.0}
    after = {module.ACCEPTED: 800.0, module.DRAFTED: 1200.0, module.DRAFTS: 120.0}
    result = module.acceptance(before, after)
    assert result.accepted == 700.0
    assert result.mean_acceptance_length == pytest.approx(8.0)
    assert result.draft_acceptance_rate == pytest.approx(0.7)


def test_speed_matrix_acceptance_is_null_without_drafts() -> None:
    module = load_module("speed_matrix")
    result = module.acceptance({}, {})
    assert result.mean_acceptance_length is None
    assert result.draft_acceptance_rate is None


@pytest.mark.parametrize("bad", ["reset", "missing", "nan", "impossible"])
def test_speed_matrix_rejects_invalid_counter_windows(bad: str) -> None:
    module = load_module("speed_matrix")
    before = {module.ACCEPTED: 10.0, module.DRAFTED: 20.0, module.DRAFTS: 5.0}
    after = {module.ACCEPTED: 12.0, module.DRAFTED: 24.0, module.DRAFTS: 6.0}
    if bad == "reset":
        after[module.ACCEPTED] = 1.0
    elif bad == "missing":
        del after[module.DRAFTED]
    elif bad == "nan":
        after[module.DRAFTED] = float("nan")
    else:
        after[module.ACCEPTED] = 100.0
    result = module.acceptance(before, after)
    assert result.error
    assert result.mean_acceptance_length is None


@pytest.fixture
def sse_endpoint():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    events = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                body = b"{}"
            elif hasattr(self.server, "metrics"):
                body = self.server.metrics.pop(0).encode()
            else:
                count = self.server.completed
                body = (
                    f"vllm:spec_decode_num_accepted_tokens_total {count * 2}\n"
                    f"vllm:spec_decode_num_draft_tokens_total {count * 3}\n"
                    f"vllm:spec_decode_num_drafts_total {count}\n"
                ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.calls.append((self.path, payload))
            if self.path == "/tokenize":
                body = json.dumps({"tokens": self.server.tokens}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.server.completed += 1
            body = "".join(
                "data: " + (json.dumps(event) if isinstance(event, dict) else event) + "\n\n"
                for event in events
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.calls = []
    server.completed = 0
    server.tokens = [1000, 1001]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", events, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("fault", [None, "usage", "short", "done", "error", "json", "empty"])
def test_speed_matrix_requires_complete_token_usage_over_http(sse_endpoint, fault) -> None:
    """One speculative chunk can carry four tokens; partial/error streams fail."""
    module = load_module("speed_matrix")
    base, events, _server = sse_endpoint
    events.extend([
        {"choices": [{"text": "one two three four", "finish_reason": None}]},
        {"choices": [{"text": "", "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 4}},
        "[DONE]",
    ])
    if fault == "usage":
        del events[2]
    elif fault == "short":
        events[2]["usage"]["completion_tokens"] = 3
    elif fault == "done":
        events.pop()
    elif fault == "error":
        events.insert(1, {"error": {"message": "rank failed"}})
    elif fault == "json":
        events.insert(1, "{bad json")
    elif fault == "empty":
        events[0]["choices"][0]["text"] = ""
    result = module.stream_completion(base, "test", [1000, 1001], 4, "", 5)
    assert result.ok is (fault is None)
    if fault is None:
        assert result.completion_tokens == 4
    else:
        assert result.error


@pytest.mark.parametrize("hit_delta", [0, 1, -1, None])
def test_speed_matrix_invalidates_hot_or_unverifiable_prefill(sse_endpoint, hit_delta) -> None:
    module = load_module("speed_matrix")
    base, events, server = sse_endpoint
    events.extend([
        {"choices": [{"text": "ok", "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1}},
        "[DONE]",
    ])
    counters = "\n".join(f"{key} 0" for key in (module.ACCEPTED, module.DRAFTED, module.DRAFTS))
    server.metrics = [
        counters + f"\n{module.CACHE_HITS} 10\n",
        counters + (f"\n{module.CACHE_HITS} {10 + hit_delta}\n" if hit_delta is not None else "\n"),
    ]
    cell = module.run_cell(base, "test", "", "prefill", 1, 2, 4, 5, 7, 4)
    assert cell.ok == 1
    assert bool(cell.validation_errors) is (hit_delta != 0)
    assert cell.prefix_cache_hits == hit_delta
    assert cell.request_results[0].completion_tokens == 1


def test_speed_matrix_cli_warms_each_real_text_cell_and_persists_receipt(sse_endpoint, tmp_path) -> None:
    base, events, server = sse_endpoint
    events.extend([
        {"choices": [{"text": "one two three four", "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 4}},
        "[DONE]",
    ])
    out = tmp_path / "receipt.json"
    markdown_out = tmp_path / "receipt.md"
    result = subprocess.run([
        sys.executable, str(SCRIPTS / "speed_matrix.py"), "--base", base,
        "--skip-prefill", "--no-warmup", "--decode-workloads", "code",
        "--decode-tokens", "4", "--concurrencies", "1", "--repetitions", "2",
        "--out", str(out), "--markdown-out", str(markdown_out),
    ], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(out.read_text())
    assert receipt["ok"] and receipt["complete"]
    assert receipt["decode_chat_template_kwargs"] == {"reasoning_effort": "max"}
    assert receipt["decode_token_scope"] == "generated_tokens_including_reasoning"
    assert "reasoning_effort=max" in markdown_out.read_text()
    assert len(receipt["warmup_cells"]) == len(receipt["cells"]) == 2
    assert server.completed == 4
    for cell in receipt["cells"]:
        assert cell["workload"] == "code"
        assert cell["acceptance"]["drafts"] == 1
        assert cell["request_results"][0]["output_text"] == "one two three four"
    assert receipt["cells"][0]["prompt_sha256"] == receipt["cells"][1]["prompt_sha256"]
    tokenize_payload = next(payload for path, payload in server.calls if path == "/tokenize")
    assert "bounded LRU cache" in tokenize_payload["messages"][0]["content"]
    for path, payload in server.calls:
        if path == "/tokenize":
            assert payload["chat_template_kwargs"] == {"reasoning_effort": "max"}
    assert not out.with_suffix(".json.tmp").exists()


def test_speed_matrix_cli_retains_incomplete_receipt_on_bad_decode_warmup(sse_endpoint, tmp_path) -> None:
    base, events, _server = sse_endpoint
    events.append({"error": {"message": "rank failed"}})
    out = tmp_path / "receipt.json"
    result = subprocess.run([
        sys.executable, str(SCRIPTS / "speed_matrix.py"), "--base", base,
        "--skip-prefill", "--no-warmup", "--decode-workloads", "code",
        "--decode-tokens", "4", "--concurrencies", "1", "--out", str(out),
    ], capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    receipt = json.loads(out.read_text())
    assert not receipt["ok"] and not receipt["complete"]
    assert receipt["warmup_cells"][0]["errors"] == 1
    assert receipt["cells"] == []


def test_snapshot_validator_rejects_truncated_payload(tmp_path: Path) -> None:
    module = load_module("validate_hf_snapshot")
    snapshot = tmp_path / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "main").write_text("abc\n")

    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    shard = snapshot / "model-00001-of-00001.safetensors"
    shard.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": shard.name}})
    )
    report = module.validate_snapshot(tmp_path, expected_shards=1)
    assert report.shard_count == 1

    shard.write_bytes(len(header).to_bytes(8, "little") + header)
    with pytest.raises(module.SnapshotError, match="truncated"):
        module.validate_snapshot(tmp_path, expected_shards=1)
