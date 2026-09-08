#!/usr/bin/env python3
"""CPU-only E3 grouped-tier eligibility, dispatch, and scratch accounting.

No CUDA and no vLLM import. Confirms EXL3_FAT_GROUPED=0 never inspects E3
symbols, missing symbols fail closed when grouped is requested, min(K) ≥ 128
is documented in eligibility, and scratch is lazy grow-only (not MNBT × top-k).
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
        "grouped_scratch_capacity",
        "fused_moe_decode_skips_fat",
        "temp_rows_fused",
    }
    wanted_assign = {
        "EXL3_FAT_MOE_SYMBOLS",
        "EXL3_FAT_MOE_MIN_CAPABILITY",
        "EXL3_FAT_MOE_MIN_K",
        "EXL3_FAT_DIAG_SCHEMA",
        "EXL3_FAT_DIAG_KEYS",
        "_FAT_TIERS",
        "GROUPED_SCRATCH_MIN_ROWS",
        "TEMP_ROWS_FUSED",
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
    assert HELPERS["EXL3_FAT_DIAG_SCHEMA"] == 3
    for key in (
        "sym_fat_moe",
        "grouped_calls",
        "grouped_scratch_bytes",
        "grouped_scratch_growths",
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


def test_grouped_scratch_is_lazy_not_mnbt_topk() -> None:
    cap = HELPERS["grouped_scratch_capacity"]
    min_rows = HELPERS["GROUPED_SCRATCH_MIN_ROWS"]
    assert min_rows == 256
    assert cap(0) == 256
    assert cap(255) == 256
    assert cap(256) == 256
    assert cap(14_336) == 14_336
    assert cap(28_672) == 28_672
    assert cap(1_000, scratch_rows=28_672) == 28_672
    assert cap(40_000, scratch_rows=28_672) == 40_000
    src = OVERLAY.read_text()
    body = src[src.index("def grouped_scratch_capacity") : src.index("def grouped_scratch_bytes_for")]
    assert "MAX_NUM_BATCHED_TOKENS" not in body
    assert "EXL3_FAT_GROUPED_TOPK" not in body
    alloc = src[src.index("def _grouped_scratch(") : src.index("def _excl_cumsum")]
    assert "is_current_stream_capturing" in alloc
    assert "grouped_scratch_growths" in alloc
    hidden, inter_full, inter_tp2, mnbt, lptt, topk = 4096, 2048, 1024, 3584, 1792, 8
    full_rows = mnbt * topk
    lptt_rows = lptt * topk
    bytes_full = HELPERS["grouped_scratch_bytes_for"](hidden, inter_full, full_rows)
    assert bytes_full == 352_321_536
    assert bytes_full == 336 * 1024 * 1024
    bytes_live = HELPERS["grouped_scratch_bytes_for"](hidden, inter_tp2, full_rows)
    assert bytes_live == 293_601_280
    bytes_lptt = HELPERS["grouped_scratch_bytes_for"](hidden, inter_tp2, lptt_rows)
    assert bytes_lptt == 146_800_640
    assert hidden % 256 == 0
    assert inter_tp2 % 128 == 0


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
    py_layer = (KIT_ROOT / "Dockerfile.e3-py-layer").read_text()
    assert "BASE=glm53-selfbuild:e3-grouped" in py_layer
    assert "COPY overlay/exl3.py" in py_layer
    assert "exl3_fat_moe.cu" not in py_layer


def test_w3_zero_fill_a_pad_not_clone() -> None:
    """W3: unused A-tile rows are cp.async src-size 0, not a clone of rows-1.

    Swizzled SMEM is still fully written so ldsm4 never reads stale bytes.
    Epilogue already skips r >= rows, so outputs are bit-identical to clone.
    Cubin-only vehicle: Dockerfile.e3-cubin-layer on e3-grouped, no exl3.py.
    """
    src = KERNEL.read_text()
    load = src[src.index("auto load_stage") : src.index("for (int s = 0; s < FM_STAGES - 1;")]
    assert "rows - 1" not in load
    assert "src_row" not in load
    assert "cp_async_cg16" in load
    assert "cp_async_cg16(dst, a, 0)" in load
    assert "cp_async_pred(" not in src
    helper = src[src.index("void cp_async_cg16") : src.index("__global__ __launch_bounds__(FM_THREADS)\nvoid fm_gather_kernel")]
    assert "cp.async.cg.shared.global" in helper
    assert ", %2" in helper
    cubin_layer = (KIT_ROOT / "Dockerfile.e3-cubin-layer").read_text()
    assert "BASE=glm53-selfbuild:e3-grouped" in cubin_layer
    assert "COPY overlay/exl3_fat_moe.cu" in cubin_layer
    assert "\nCOPY overlay/exl3.py" not in cubin_layer
    assert "e3-w3-zfill" in cubin_layer
    assert "RUN CUDA_VISIBLE_DEVICES= python3" in cubin_layer
    assert "ENV CUDA_VISIBLE_DEVICES=" not in cubin_layer


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
    assert 'EXL3_FAT_SCRATCH_ROWS="${EXL3_FAT_SCRATCH_ROWS:-}"' in start
    assert "EXL3_FAT_SCRATCH_ROWS" in env
    # Empty must stay unset in the container: the adopted e3-grouped overlay
    # treats empty as 0 and skips MNBT×topk, which would contaminate image A/B.
    assert "EXL3_FAT_KERNEL EXL3_FAT_GROUPED EXL3_FAT_SCRATCH_ROWS MODEL_DIR" not in start
    assert '[ -n "${EXL3_FAT_SCRATCH_ROWS:-}" ]' in start
    assert '${EXL3_FAT_SCRATCH_ROWS:+-e EXL3_FAT_SCRATCH_ROWS="$EXL3_FAT_SCRATCH_ROWS"}' in start
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


def test_w2_decode_skip_gate_c4_unique_topk() -> None:
    """W2 hard gate: fused decode drops count > TRF with no fat fallback.

    C4 decode T = 4 seqs × 8 drafts = 32. Hottest expert count ≤ T when
    top-k is unique per token. Kernel skip is `token_count > cap`, so
    TRF=32 does not drop experts even Zipf-all-to-one. Prefill T > cap
    takes fat/grouped (helper False). No-fallback skip needs hottest >
    cap while T ≤ cap (non-unique top-k). Production TRF stays 128.
    """
    skip = HELPERS["fused_moe_decode_skips_fat"]
    assert HELPERS["TEMP_ROWS_FUSED"] == 128
    # Decode early-return in apply_exl3_fused_moe, then kernel skip.
    apply = OVERLAY.read_text()
    assert "if tokens <= cap:" in apply
    assert "Decode never reaches here" in apply
    kernel = (KIT_ROOT / "tests" / "fixtures" / "exl3-v147" / "quant" / "exl3_moe_kernel.cuh").read_text()
    assert "if (token_count > max_tokens_per_expert) continue;" in kernel
    # Production C4 unique-per-token: 4 × 8 drafts → T=32, hottest ≤ T.
    assert skip(tokens=32, hottest_expert_count=32, cap=128) is False
    assert skip(tokens=32, hottest_expert_count=32, cap=32) is False
    # Prefill T > cap takes fat/grouped instead of the decode skip.
    assert skip(tokens=1792, hottest_expert_count=1792, cap=32) is False
    assert skip(tokens=40, hottest_expert_count=40, cap=32) is False
    c4_unique = 4 * 8
    assert c4_unique == 32
    assert skip(c4_unique, c4_unique, 32) is False
    # Non-unique top-k (one expert appears twice in a token) can make
    # hottest > T; then decode T ≤ cap still skips with no fat fallback.
    assert skip(tokens=32, hottest_expert_count=33, cap=32) is True
