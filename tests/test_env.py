from __future__ import annotations

import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / "env.example"


def env_pairs() -> dict[str, str]:
    pairs: dict[str, str] = {}
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        parsed = shlex.split(value, comments=True)
        pairs[key] = parsed[0] if parsed else ""
    return pairs


def test_required_keys_are_documented() -> None:
    pairs = env_pairs()
    for key in (
        "CLUSTER_NODES",
        "ETH_IF",
        "IB_IF",
        "COPY_HOSTS",
        "MASTER_PORT",
        "CONTAINER_NAME",
        "SPARK_VLLM_REF",
        "SPARK_VLLM_DIR",
        "IMAGE_REF",
        "IMAGE_TAG",
        "PULL_IMAGE",
        "PORT",
        "HOST",
        "MAX_MODEL_LEN",
        "GPU_MEMORY_UTILIZATION",
    ):
        assert key in pairs, key


def test_launcher_pin_and_image_are_the_supported_ones() -> None:
    pairs = env_pairs()
    assert re.fullmatch(r"[0-9a-f]{40}", pairs["SPARK_VLLM_REF"])
    assert pairs["IMAGE_REF"] == "eugr/spark-vllm-b12x:latest"
    assert pairs["IMAGE_TAG"] == "vllm-node-b12x"
    assert pairs["SPARK_VLLM_REF"].startswith("0bdc9c94")


def test_no_secret_looking_values() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    assert not re.search(r"BEGIN [A-Z ]*PRIVATE KEY", text)
    pairs = env_pairs()
    assert pairs.get("HF_TOKEN", "") == ""


def test_example_nodes_are_placeholders_not_a_real_site() -> None:
    pairs = env_pairs()
    assert pairs["CLUSTER_NODES"].startswith("192.0.2.")
    assert "192.168.177." not in pairs["CLUSTER_NODES"]


def test_env_example_has_no_exl3_knobs() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8").lower()
    for forbidden in ("exl3", "exllama", "dflash", "tr3"):
        assert forbidden not in text, forbidden
