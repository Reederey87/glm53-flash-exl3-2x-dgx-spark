#!/usr/bin/env python3
"""W4 fused-gather is opt-in. Default overlay/ stays W3-matched.

Cluster 2026-09-08 REVERT: 60k −10.7%, no MemFree win. Production
IMAGE=e3-w3-zfill. These tests fail closed if W4 leaks into the
default Dockerfile or the W1/W3 layer recipes.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parents[1]
OVERLAY = KIT_ROOT / "overlay" / "exl3.py"
KERNEL = KIT_ROOT / "overlay" / "exl3_fat_moe.cu"
HEADER = KIT_ROOT / "overlay" / "exl3_fat_moe.cuh"
BUILDER = KIT_ROOT / "overlay" / "build_exl3_fat_moe_ext.py"
PATCHER = KIT_ROOT / "overlay" / "patch_exl3_fat_kernel.py"
W4 = KIT_ROOT / "overlay-w4"
W4_OVERLAY = W4 / "exl3.py"
W4_KERNEL = W4 / "exl3_fat_moe.cu"
W4_HEADER = W4 / "exl3_fat_moe.cuh"
W4_BUILDER = W4 / "build_exl3_fat_moe_ext.py"
W4_LAYER = KIT_ROOT / "Dockerfile.e3-w4-layer"
CUBIN_LAYER = KIT_ROOT / "Dockerfile.e3-cubin-layer"
PY_LAYER = KIT_ROOT / "Dockerfile.e3-py-layer"
DOCKERFILE = KIT_ROOT / "Dockerfile"
ENV_EXAMPLE = KIT_ROOT / "env.example"


def _exec_helpers(path: Path) -> dict:
    src = path.read_text()
    tree = ast.parse(src)
    keep: list[str] = []
    wanted_fn = {
        "grouped_scratch_bytes_for",
        "grouped_scratch_capacity",
    }
    wanted_assign = {
        "EXL3_FAT_MOE_SYMBOLS",
        "EXL3_FAT_DIAG_SCHEMA",
        "EXL3_FAT_DIAG_KEYS",
        "GROUPED_SCRATCH_MIN_ROWS",
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_fn:
            keep.append(ast.get_source_segment(src, node) or "")
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(name in wanted_assign for name in names):
                keep.append(ast.get_source_segment(src, node) or "")
    ns: dict = {"os": os}
    exec("\n\n".join(keep), ns, ns)
    return ns


def test_default_overlay_keeps_w3_gather_interface() -> None:
    overlay = OVERLAY.read_text()
    kernel = KERNEL.read_text()
    header = HEADER.read_text()
    builder = BUILDER.read_text()
    patcher = PATCHER.read_text()
    docker = DOCKERFILE.read_text()
    assert '"exl3_fat_moe_gather"' in overlay
    assert "exl3_fat_moe_gather(" in overlay
    assert "void exl3_fat_moe_gather" in kernel
    assert "void exl3_fat_moe_gather" in header
    assert "exl3_fat_moe_gather" in builder
    assert "exl3_fat_moe_gather" in patcher
    assert "exl3_fat_moe_gather" in docker
    assert "overlay-w4" not in docker
    assert "fm_mainloop_fused" not in kernel


def test_historical_layer_recipes_stay_on_overlay_not_w4() -> None:
    cubin = CUBIN_LAYER.read_text()
    py = PY_LAYER.read_text()
    assert "COPY overlay/exl3_fat_moe.cu" in cubin
    assert "overlay-w4" not in cubin
    assert "\nCOPY overlay/exl3.py" not in cubin
    assert "COPY overlay/exl3.py" in py
    assert "overlay-w4" not in py
    assert "exl3_fat_moe.cu" not in py
    # Rebuilding the cubin layer from default overlay must still expose gather.
    assert "void exl3_fat_moe_gather" in KERNEL.read_text()
    # Rebuilding the Python layer from default overlay must still call gather.
    assert "exl3_fat_moe_gather(" in OVERLAY.read_text()


def test_w4_layer_consumes_overlay_w4_only() -> None:
    w4 = W4_LAYER.read_text()
    assert "BASE=glm53-selfbuild:e3-w3-zfill" in w4
    assert "COPY overlay-w4/exl3_fat_moe.cu" in w4
    assert "COPY overlay-w4/exl3.py" in w4
    assert "COPY overlay/exl3_fat_moe.cu" not in w4
    assert "COPY overlay/exl3.py" not in w4
    assert "e3-w4-fgather" in w4
    assert "RUN CUDA_VISIBLE_DEVICES= python3" in w4
    assert "ENV CUDA_VISIBLE_DEVICES=" not in w4
    assert "assert not hasattr(m, 'exl3_fat_moe_gather')" in w4


def test_w4_sources_drop_gather_and_h13() -> None:
    src = W4_KERNEL.read_text()
    header = W4_HEADER.read_text()
    overlay = W4_OVERLAY.read_text()
    builder = W4_BUILDER.read_text()
    helpers = _exec_helpers(W4_OVERLAY)
    assert "void exl3_fat_moe_gather" not in src
    assert "void exl3_fat_moe_gather" not in header
    assert "fm_gather_kernel" not in src
    assert "fm_mainloop_fused" in src
    fused = src[src.index("fm_mainloop_fused") : src.index("void fm_stage_acc")]
    assert "__hmul2" in fused
    assert "fm_had_row" in fused
    assert "fm_smem_bytes_fused" in src
    assert "101376" in src
    assert '"exl3_fat_moe_gather"' not in overlay
    apply = overlay[overlay.index("def apply_exl3_grouped_fat") : overlay.index("def build_exl3_fused_state")]
    assert "exl3_fat_moe_gather" not in apply
    assert 'scratch["h13"]' not in apply
    assert "xh," in apply
    assert 'ptrs["gate_suh"]' in apply
    alloc = overlay[overlay.index("def _grouped_scratch(") : overlay.index("def _excl_cumsum")]
    assert '"h13"' not in alloc
    assert '"h2"' in alloc
    assert "exl3_fat_moe_gather" not in builder
    hidden, inter_tp2, rows = 4096, 1024, 28_672
    assert helpers["grouped_scratch_bytes_for"](hidden, inter_tp2, rows) == 58_720_256


def test_env_example_does_not_pin_w4_image() -> None:
    env = ENV_EXAMPLE.read_text()
    assert "IMAGE=glm53-selfbuild:e3-w4-fgather" not in env
    assert "e3-w3-zfill" in env
    assert "overlay-w4" in env
    assert "EXL3_FAT_GROUPED=1" in env
