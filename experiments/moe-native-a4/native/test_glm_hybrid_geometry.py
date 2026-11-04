"""Prepared GPU case; run only inside the future idle-node native gate.

Tests the incumbent hybrid arithmetic at actual GLM TP2 width/expert geometry.
This is not native-A4 qualification, a full 4096-token prefill test, or a
completed cluster result. The pinned upstream test uses 192 live tokens.
"""
import hashlib
import importlib
from pathlib import Path
import sys


def test_glm_tp2_hybrid_prefill_matches_float64(monkeypatch):
    import torch

    assert torch.cuda.is_available(), "native gate requires CUDA; no skip"
    assert torch.cuda.get_device_capability() == (12, 1), "requires the GB10 SM121 device"
    root = Path("/opt/moe-native-a4-tests")
    source = root / "tests/experimental/b12x/moe/test_w4a16_a4_prefill.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "a00f5c683982ee8d6cdeea5dc6fa75c2cc75dde881abc8f49ead1901899acce2"
    ), "upstream test bytes differ from the reviewed pin"
    sys.path.insert(0, str(root))
    upstream = importlib.import_module("tests.experimental.b12x.moe.test_w4a16_a4_prefill")
    assert Path(upstream.__file__).resolve() == source, "unexpected test import owner"
    upstream.test_a4_prefill_matches_float64_emulation(
        terms=1,
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=288,
        top_k=8,
        swiglu_limit=10.0,
        monkeypatch=monkeypatch,
    )
