from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "glm-5.3-flash-fp4-local.yaml"


def load_recipe() -> dict:
    return yaml.safe_load(RECIPE.read_text(encoding="utf-8"))


def rendered_command() -> str:
    recipe = load_recipe()
    return recipe["command"].format(**recipe["defaults"])


def test_recipe_targets_fp4_local_checkpoint() -> None:
    recipe = load_recipe()
    assert recipe["model"] == "nvidia/GLM-5.3-Flash-FP4-Local"
    assert recipe["container"] == "vllm-node-b12x"
    assert recipe["build_args"] == ["--exp-b12x"]
    assert recipe["cluster_only"] is True
    assert recipe["recipe_version"] == "1"


def test_recipe_defaults_encode_two_million_token_pool() -> None:
    recipe = load_recipe()
    defaults = recipe["defaults"]
    assert defaults["host"] == "127.0.0.1"
    assert defaults["tensor_parallel"] == 2
    assert defaults["block_size"] == 256
    assert defaults["max_model_len"] == 500000
    assert defaults["max_num_seqs"] == 4
    assert defaults["max_num_batched_tokens"] == 4096
    assert defaults["num_speculative_tokens"] == 5
    assert defaults["kv_cache_bytes"] == "11G"
    # Four 500k requests are the 2M pool.
    assert defaults["max_num_seqs"] * 500000 == 2_000_000


def test_command_contains_required_b12x_flags() -> None:
    command = rendered_command()
    required = [
        "vllm serve nvidia/GLM-5.3-Flash-FP4-Local",
        "--host 127.0.0.1",
        "--tensor-parallel-size 2",
        "--decode-context-parallel-size 1",
        "--mamba-cache-mode align",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--kv-cache-dtype fp8",
        "--kv-cache-memory-bytes 11G",
        "--quantization modelopt_mixed",
        "--attention-backend B12X",
        "--block-size 256",
        "--moe-backend b12x",
        "--linear-backend b12x",
        "--no-enable-flashinfer-autotune",
        "--load-format b12x",
        "--max-model-len 500000",
        "--max-num-seqs 4",
        "--max-num-batched-tokens 4096",
        "--reasoning-parser glm45",
        "--tool-call-parser glm47",
        "--enable-auto-tool-choice",
        "--gpu-memory-utilization 0.87",
    ]
    for flag in required:
        assert flag in command, f"missing {flag!r}"


def test_speculative_config_is_valid_json_after_formatting() -> None:
    import json

    command = rendered_command()
    match = re.search(r"--speculative-config '(\{.*?\})'", command)
    assert match, command
    spec = json.loads(match.group(1))
    assert spec == {
        "method": "mtp",
        "num_speculative_tokens": 5,
        "moe_backend": "humming",
        "attention_backend": "B12X",
    }


def test_command_has_no_unformatted_placeholders() -> None:
    command = rendered_command()
    # The speculative JSON is stored with doubled braces and must collapse.
    assert "{{" not in command
    assert "}}" not in command
    leftovers = re.findall(r"\{[a-z_]+\}", command)
    assert leftovers == []


def test_recipe_env_matches_vendor_stack() -> None:
    env = load_recipe()["env"]
    expected = {
        "CUTE_DSL_ARCH": "sm_121a",
        "SAFETENSORS_FAST_GPU": "1",
        "VLLM_ENABLE_ROCE_ALLREDUCE": "1",
        "VLLM_ROCE_ALLREDUCE_MAX_SIZE": "2MB",
        "NCCL_BUFFSIZE": "1048576",
        "NCCL_MAX_CTAS": "8",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        "VLLM_USE_AOT_COMPILE": "1",
        "VLLM_USE_MEGA_AOT_ARTIFACT": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_ENABLE_PCIE_ALLREDUCE": "0",
        "B12X_POLICY_MODE": "auto",
        "VLLM_B12X_MOE_FP4_FORCE_A16": "1",
        "B12X_W4A16_A4_PREFILL": "1",
    }
    assert env == expected


def test_recipe_has_no_exl3_surface() -> None:
    text = RECIPE.read_text(encoding="utf-8").lower()
    for forbidden in ("exl3", "exllama", "brandonmusic", "incoai", "tr3-4bpw", "dflash"):
        assert forbidden not in text, forbidden
