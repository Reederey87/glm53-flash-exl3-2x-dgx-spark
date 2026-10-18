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
