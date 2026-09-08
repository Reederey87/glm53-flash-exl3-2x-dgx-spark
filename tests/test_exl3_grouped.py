#!/usr/bin/env python3
"""CPU-only E3 grouped-tier eligibility, dispatch, and scratch accounting.

No CUDA and no vLLM import. Confirms EXL3_FAT_GROUPED=0 never inspects E3
symbols, missing symbols fail closed when grouped is requested, min(K) ≥ 128
is documented in eligibility, and production TP2 shapes predict 336 MiB/rank.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

KIT_ROOT = Path(__file__).resolve().parents[1]
OVERLAY = KIT_ROOT / "overlay" / "exl3.py"
KERNEL = KIT_ROOT / "overlay" / "exl3_fat_moe.cu"
LAUNCHER = KIT_ROOT / "start.sh"
ENV_EXAMPLE = KIT_ROOT / "env.example"
DOCKERFILE = KIT_ROOT / "Dockerfile"
LAYER = KIT_ROOT / "Dockerfile.e3-layer"
PATCHER = KIT_ROOT / "overlay" / "patch_exl3_fat_kernel.py"
BUILDER = KIT_ROOT / "overlay" / "build_exl3_fat_moe_ext.py"


def _exec_helpers():
    src = OVERLAY.read_text()
    tree = ast.parse(src)
    keep: list[str] = []
    wanted_fn = {
        "grouped_fat_enabled",
        "fat_kernel_enabled",
        "batched_fat_fallback_enabled",
        "sorted_fat_fallback_enabled",
        "configured_fat_tier",
        "grouped_fat_eligibility",
        "resolve_exl3_fat_tier",
        "grouped_scratch_bytes_for",
    }
    wanted_assign = {
        "EXL3_FAT_MOE_SYMBOLS",
        "EXL3_FAT_MOE_MIN_CAPABILITY",
        "EXL3_FAT_MOE_MIN_K",
        "EXL3_FAT_DIAG_SCHEMA",
        "EXL3_FAT_DIAG_KEYS",
        "_FAT_TIERS",
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_fn:
            keep.append(ast.get_source_segment(src, node) or "")
        elif isinstance(node, ast.Assign):
            names = [
                t.id for t in node.targets if isinstance(t, ast.Name)
            ]
            if any(name in wanted_assign for name in names):
                keep.append(ast.get_source_segment(src, node) or "")
    ns: dict = {"os": os}
    exec("\n\n".join(keep), ns, ns)
    return ns


HELPERS = _exec_helpers()


def _inner(k_in: int = 4096, mcg: bool = True, mul1: bool = False, bits: int = 4):
    return SimpleNamespace(K=bits, mcg=mcg, mul1=mul1, in_features=k_in)


def _layer(
    *,
    bits: int = 4,
    k_words: int = 64,
    hidden: int = 4096,
    inter: int = 2048,
    shared: bool = True,
    k_in: int = 4096,
    cuda: bool = True,
):
    pack = {
        "gate": _inner(k_in),
        "up": _inner(k_in),
        "down": _inner(k_in),
    }
    device = SimpleNamespace(type="cuda" if cuda else "cpu")
    tensor = SimpleNamespace(device=device)
    return SimpleNamespace(
        _exl3_bits=bits,
        _exl3_k_words=k_words,
        _exl3_inners=[pack],
        _exl3_shared_w13_suh=shared,
        _exl3_hidden_size=hidden,
        _exl3_intermediate_local=inter,
        w13_trellis=tensor,
        w13_suh=tensor,
        w13_svh=tensor,
        w2_trellis=tensor,
        w2_suh=tensor,
        w2_svh=tensor,
    )


def test_schema_keys_include_grouped() -> None:
    assert HELPERS["EXL3_FAT_DIAG_SCHEMA"] == 2
    for key in (
        "sym_fat_moe",
        "grouped_calls",
        "grouped_scratch_bytes",
        "grouped_eligible",
    ):
        assert key in HELPERS["EXL3_FAT_DIAG_KEYS"]
    assert "grouped" in HELPERS["_FAT_TIERS"]
    assert HELPERS["EXL3_FAT_MOE_MIN_K"] == 128
    src = OVERLAY.read_text()
    assert "_exl3_last_fat_fallback" in src
    assert "row_tile" in src
    assert "EXL3_FAT_GROUPED" in src
    assert src.count("EXLLAMAV3_COMMIT = ") == 1
    assert "ca13bdd83a1f4a74fd817b88f49509e0f22a9b07" in src


def test_grouped_off_is_kernel_without_e3_symbols(monkeypatch) -> None:
    monkeypatch.setenv("EXL3_FAT_GROUPED", "0")
    monkeypatch.setenv("EXL3_FAT_KERNEL", "1")
    monkeypatch.setenv("EXL3_FAT_BATCHED", "1")
    monkeypatch.setenv("EXL3_FAT_SORTED", "1")
    called = {"moe": 0}

    def boom(*_a, **_k):
        called["moe"] += 1
        raise AssertionError("E3 symbol lookup must not run when grouped is off")

    monkeypatch.setitem(HELPERS, "exl3_fat_moe_symbols", boom)
    monkeypatch.setitem(
        HELPERS, "exl3_fat_symbols", lambda * _a, **_k: (True, True, True)
    )
    # Rebind names used inside resolve via defaults / global lookup.
    resolve = HELPERS["resolve_exl3_fat_tier"]
    resolve.__globals__["exl3_fat_moe_symbols"] = boom
    resolve.__globals__["exl3_fat_symbols"] = lambda * _a, **_k: (True, True, True)
    tier, reason = resolve(True)
    assert called["moe"] == 0
    assert (tier, reason) == ("kernel", "kernel_ok")
    assert HELPERS["configured_fat_tier"]() == "kernel"


def test_grouped_on_without_symbols_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("EXL3_FAT_GROUPED", "1")
    resolve = HELPERS["resolve_exl3_fat_tier"]
    resolve.__globals__["exl3_fat_moe_symbols"] = lambda * _a, **_k: False
    with pytest.raises(RuntimeError, match="EXL3_FAT_GROUPED=1 requires"):
        resolve(True, grouped_eligible=(True, "eligible"))


def test_grouped_on_without_symbols_fails_closed_nonshared_suh(monkeypatch) -> None:
    """Missing E3 symbols fail closed even when the checkpoint would also drop SUH."""
    monkeypatch.setenv("EXL3_FAT_GROUPED", "1")
    resolve = HELPERS["resolve_exl3_fat_tier"]
    resolve.__globals__["exl3_fat_moe_symbols"] = lambda * _a, **_k: False
    with pytest.raises(RuntimeError, match="EXL3_FAT_GROUPED=1 requires"):
        resolve(False, grouped_eligible=(False, "shared_suh_absent"))


def test_grouped_off_nonshared_suh_still_skips_e3(monkeypatch) -> None:
    monkeypatch.setenv("EXL3_FAT_GROUPED", "0")
    monkeypatch.setenv("EXL3_FAT_KERNEL", "1")
    monkeypatch.setenv("EXL3_FAT_BATCHED", "1")
    monkeypatch.setenv("EXL3_FAT_SORTED", "1")
    called = {"moe": 0}

    def boom(*_a, **_k):
        called["moe"] += 1
        raise AssertionError("E3 symbol lookup must not run when grouped is off")

    resolve = HELPERS["resolve_exl3_fat_tier"]
    resolve.__globals__["exl3_fat_moe_symbols"] = boom
    resolve.__globals__["exl3_fat_symbols"] = lambda * _a, **_k: (True, True, True)
    tier, reason = resolve(False)
    assert called["moe"] == 0
    assert (tier, reason) == ("sorted", "shared_suh_absent")


def test_microbench_diag_keys_match_schema() -> None:
    bench = (KIT_ROOT / "tests" / "bench_e3_microbench.py").read_text()
    assert 'exl3_fat_diag()["grouped_scratch_bytes"]' in bench
    assert "fat_scratch_bytes" not in bench
    assert "grouped_scratch_bytes" in HELPERS["EXL3_FAT_DIAG_KEYS"]
    assert "fat_scratch_bytes" not in HELPERS["EXL3_FAT_DIAG_KEYS"]


def test_grouped_ineligible_falls_back_to_e2(monkeypatch) -> None:
    monkeypatch.setenv("EXL3_FAT_GROUPED", "1")
    resolve = HELPERS["resolve_exl3_fat_tier"]
    resolve.__globals__["exl3_fat_moe_symbols"] = lambda * _a, **_k: True
    resolve.__globals__["exl3_fat_symbols"] = lambda * _a, **_k: (True, True, True)
    tier, reason = resolve(True, grouped_eligible=(False, "k_in_96"))
    assert tier == "kernel"
    assert reason.startswith("grouped_ineligible_k_in_96:")


def test_eligibility_documents_min_k(monkeypatch) -> None:
    layer = _layer(k_in=96, hidden=256, inter=128)
    fn = HELPERS["grouped_fat_eligibility"]
    fn.__globals__["exl3_device_capability"] = lambda: (12, 1)
    ok, reason = fn(layer)
    assert (ok, reason) == (False, "k_in_96")
    src = OVERLAY.read_text()
    assert "min(K) ≥ 128" in src or "min(K) >= 128" in src
    assert "FM_STAGES" in src
    assert "k_tiles" in src


def test_eligibility_accepts_production_tp2_shape() -> None:
    layer = _layer()
    fn = HELPERS["grouped_fat_eligibility"]
    fn.__globals__["exl3_device_capability"] = lambda: (12, 1)
    ok, reason = fn(layer)
    assert (ok, reason) == (True, "eligible")


def test_grouped_scratch_bytes_336mib_at_production() -> None:
    hidden, inter, mnbt, topk = 4096, 2048, 3584, 8
    rows = mnbt * topk
    bytes_ = HELPERS["grouped_scratch_bytes_for"](hidden, inter, rows)
    assert bytes_ == 352_321_536
    assert bytes_ == 336 * 1024 * 1024
    assert hidden % 256 == 0
    assert inter % 128 == 0
    assert inter % 256 == 0


def test_kernel_intake_and_float4_atomic() -> None:
    src = KERNEL.read_text()
    assert "ptx_mma_m16n8k16" in src
    assert "cp_async" in src
    assert "FM_STAGES = 4" in src
    assert "FM_TILE_K = 32" in src
    assert "atomicAdd(reinterpret_cast<float4*>" in src
    assert "exp(-(double)" in src
    assert "__fdiv_rn" in src
    assert "tcgen05" not in src.lower()
    assert "TMEM" not in src
    builder = BUILDER.read_text()
    assert "121a" in builder
    assert "--use_fast_math" in builder
    layer = LAYER.read_text()
    assert "BASE=glm53-selfbuild:ca13bdd-v147" in layer
    assert "EXL3_SELFCHECK_GPU=0" in layer


def test_launcher_and_env_grouped_adopted() -> None:
    start = LAUNCHER.read_text()
    env = ENV_EXAMPLE.read_text()
    # Launcher default stays 0 so GHCR/old images fail closed; env.example
    # documents the adopted production arm.
    assert 'EXL3_FAT_GROUPED="${EXL3_FAT_GROUPED:-0}"' in start
    assert "EXL3_FAT_GROUPED=1" in env
    assert "EXL3_FAT_GROUPED" in start
    assert "-e EXL3_FAT_GROUPED=" in start
    assert "must be exactly 0 or 1" in start
    docker = DOCKERFILE.read_text()
    assert "COPY overlay/exl3_fat_moe.cu" in docker
    assert "exl3_fat_moe_gather" in docker
    patch = PATCHER.read_text()
    assert "exl3_fat_moe.cu" in patch
    assert "exl3_fat_moe_gather" in patch


def test_row_tile_still_short_circuits_grouped() -> None:
    src = OVERLAY.read_text()
    assert "and not use_row_tiles" in src
    assert 'layer._exl3_last_fat_fallback = "none"' in src
    assert "EXL3_MOE_ROW_TILE" in src
