#!/usr/bin/env python3
"""[glm53-exl3-moe-pipeline] CPU regressions for the task 42 overlay.

These run against a synthetic copy of the pinned exllamav3 v1.4.9 anchors, and
against the real deployed source tree when it is reachable (the same pattern the
other overlay tests in this kit use). They pin the properties the arm depends on:

  * the stock kernel is untouched and the variant is a *separate* symbol, so the
    two can coexist without an ODR clash;
  * the geometry actually reaches the comp unit (an overlay that silently emitted
    the stock 3/3 numbers would measure nothing);
  * arming is opt-in and the unarmed path selects the stock instance;
  * the dispatch fails closed on a geometry the variant was not built for;
  * a drifted anchor refuses rather than guessing, and a re-run is refused
    instead of double-applying.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1]
OVERLAY = KIT / "overlay" / "patch_exl3_moe_pipeline.py"


def _load_overlay():
    spec = importlib.util.spec_from_file_location("glm53_moe_pipeline_overlay", OVERLAY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OV = _load_overlay()


# --- the pinned v1.4.9 anchors, as a minimal synthetic tree -----------------

KERNEL_CUH = """#pragma once
#include "exl3_moe_common.cuh"

template<int t_bits, int MOE_TILESIZE_N, int cb>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16)
void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)
{
    exl3_gemm_kernel_inner<1, false, 1, 16, 32, 256, MOE_SH_STAGES, MOE_FRAG_STAGES>(nullptr);
}
"""

COMMON_CUH = """#pragma once
#define MOE_TILESIZE_K 32
#define MOE_TILESIZE_M 16
#define MOE_SH_STAGES 3
#define MOE_FRAG_STAGES 3
"""

INSTANCES_CUH = """#pragma once
#include "../exl3_moe_common.cuh"
typedef void (*fp_exl3_moe_kernel) (EXL3_MOE_KERNEL_ARGS);
"""

HOST_CU = """#include <cuda_fp16.h>
#include "exl3_gemm.cuh"
#include "comp_units/exl3_moe_instances.cuh"
#include "exl3_devctx.cuh"
#include <set>

std::set<void*> moe_kernel_attr_set[MAX_DEVICES] = {};

fp_exl3_moe_kernel exl3_moe_kernel_instances[] =
{
    exl3_moe_kernel_k0_n128_cb1(), exl3_moe_kernel_k0_n256_cb1(),
};

void exl3_moe()
{
    const int cb_idx = gate_mul1 ? 1 : 0;
    int N_off = 0;
    if (hidden_dim % 256 == 0 && intermediate_dim % 256 == 0) N_off = 1;
    fp_exl3_moe_kernel kernel = exl3_moe_kernel_instances[4 * K + 2 * cb_idx + N_off];
    cudaLaunchKernel((void*) kernel, grid_dim, block_dim, kernelArgs, SMEM_MAX, stream);
}
"""

BINDINGS_CPP = """#include <torch/extension.h>
#include "quant/exl3_moe.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("exl3_moe", &exl3_moe, "exl3_moe");
}
"""


def _make_tree(root: Path) -> Path:
    ext = root / "exllamav3_ext"
    quant = ext / "quant"
    (quant / "comp_units").mkdir(parents=True)
    (quant / "exl3_moe_kernel.cuh").write_text(KERNEL_CUH)
    (quant / "exl3_moe_common.cuh").write_text(COMMON_CUH)
    (quant / "comp_units" / "exl3_moe_instances.cuh").write_text(INSTANCES_CUH)
    (quant / "exl3_moe.cu").write_text(HOST_CU)
    (ext / "bindings.cpp").write_text(BINDINGS_CPP)
    return ext


def _apply(ext: Path, frag: int = 1, sh: int = 8) -> None:
    argv = sys.argv
    sys.argv = ["patch_exl3_moe_pipeline.py", str(ext), "--frag", str(frag), "--sh", str(sh)]
    try:
        assert OV.main() == 0
    finally:
        sys.argv = argv


# --- tests -----------------------------------------------------------------


def test_variant_is_a_separate_symbol_and_stock_is_untouched(tmp_path):
    ext = _make_tree(tmp_path)
    stock_before = (ext / "quant" / "exl3_moe_kernel.cuh").read_text()
    _apply(ext)

    # The stock header is byte-identical: the variant is additive, not a rewrite.
    assert (ext / "quant" / "exl3_moe_kernel.cuh").read_text() == stock_before

    variant = (ext / "quant" / "glm53_exl3_moe_pipeline_kernel.cuh").read_text()
    assert "glm53_exl3_moe_pipeline_kernel(EXL3_MOE_KERNEL_ARGS)" in variant
    # Renamed, not merely re-included: keeping the old name would be an ODR clash
    # with the stock instance at the same template signature.
    assert "void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)" not in variant


def test_geometry_reaches_the_comp_unit(tmp_path):
    ext = _make_tree(tmp_path)
    _apply(ext, frag=1, sh=8)
    unit = (ext / "quant" / "comp_units" / "glm53_exl3_moe_pipeline.cu").read_text()
    assert "#define MOE_FRAG_STAGES 1" in unit
    assert "#define MOE_SH_STAGES 8" in unit
    assert "glm53_exl3_moe_pipeline_kernel<4, 256, 1>" in unit
    assert 'return "frag1_sh8";' in unit
    # An overlay that emitted the stock numbers would make the arm inert while
    # still booting, which is the failure mode this pins against.
    assert "#define MOE_FRAG_STAGES 3" not in unit
    assert "#define MOE_SH_STAGES 3" not in unit


def test_geometry_is_configurable(tmp_path):
    ext = _make_tree(tmp_path)
    _apply(ext, frag=2, sh=4)
    unit = (ext / "quant" / "comp_units" / "glm53_exl3_moe_pipeline.cu").read_text()
    assert "#define MOE_FRAG_STAGES 2" in unit
    assert "#define MOE_SH_STAGES 4" in unit
    assert 'return "frag2_sh4";' in unit


def test_dispatch_is_opt_in_and_fails_closed(tmp_path):
    ext = _make_tree(tmp_path)
    _apply(ext)
    host = (ext / "quant" / "exl3_moe.cu").read_text()

    # Armed path is gated on the env knob, read once per process.
    assert 'getenv("GLM53_EXL3_MOE_PIPELINE")' in host
    assert "if (glm53_exl3_moe_pipeline_armed())" in host
    # The stock selection line is still the unconditional default.
    assert (
        "fp_exl3_moe_kernel kernel = exl3_moe_kernel_instances[4 * K + 2 * cb_idx + N_off];"
        in host
    )
    # Fail closed on every geometry the variant is not instantiated for.
    assert "GLM53_EXL3_MOE_PIPELINE=1 requires SM121" in host
    assert "K == 4 && N_off == 1 && cb_idx == 0" in host
    assert "refusing to serve with a silently inert arm" in host


def test_geometry_is_reported_for_audit(tmp_path):
    ext = _make_tree(tmp_path)
    _apply(ext)
    host = (ext / "quant" / "exl3_moe.cu").read_text()
    bindings = (ext / "bindings.cpp").read_text()

    # The running process must be auditable without disassembling the cubin.
    assert "[glm53-exl3-moe-pipeline] active:" in host
    assert "glm53_exl3_moe_pipeline_geometry()" in host
    assert "glm53_exl3_moe_pipeline_geometry_literal()" in host
    assert 'm.def("glm53_exl3_moe_pipeline_geometry"' in bindings
    # bindings.cpp must see the declaration, not just the definition.
    header = (ext / "quant" / "glm53_exl3_moe_pipeline.cuh").read_text()
    assert "std::string glm53_exl3_moe_pipeline_geometry();" in header
    assert "std::string glm53_exl3_moe_pipeline_geometry_literal();" in header


def test_rerun_is_refused_rather_than_double_applied(tmp_path):
    ext = _make_tree(tmp_path)
    _apply(ext)
    host_after_first = (ext / "quant" / "exl3_moe.cu").read_text()
    with pytest.raises(SystemExit, match="already present"):
        _apply(ext)
    assert (ext / "quant" / "exl3_moe.cu").read_text() == host_after_first


def test_drifted_anchor_fails_closed_and_writes_nothing(tmp_path):
    ext = _make_tree(tmp_path)
    # Break the dispatch anchor only.
    host_path = ext / "quant" / "exl3_moe.cu"
    host_path.write_text(HOST_CU.replace("4 * K + 2 * cb_idx + N_off", "4 * K + N_off"))
    before = host_path.read_text()

    with pytest.raises(SystemExit, match="exactly one anchor"):
        _apply(ext)

    # A partial application would leave a marker-bearing half-patched tree, so
    # nothing may be written at all.
    assert host_path.read_text() == before
    assert not (ext / "quant" / "glm53_exl3_moe_pipeline_kernel.cuh").exists()
    assert not (ext / "quant" / "comp_units" / "glm53_exl3_moe_pipeline.cu").exists()


def test_missing_kernel_anchor_fails_closed(tmp_path):
    ext = _make_tree(tmp_path)
    kernel = ext / "quant" / "exl3_moe_kernel.cuh"
    kernel.write_text(KERNEL_CUH.replace("void exl3_moe_kernel(", "void exl3_moe_kernel_v2("))
    with pytest.raises(SystemExit, match="exactly one anchor"):
        _apply(ext)


@pytest.mark.parametrize("frag", [0, 6, -1])
def test_out_of_range_geometry_is_rejected(tmp_path, frag):
    ext = _make_tree(tmp_path)
    with pytest.raises(SystemExit, match="--frag must be"):
        _apply(ext, frag=frag, sh=8)


@pytest.mark.parametrize("sh", [1, 17, 0])
def test_out_of_range_smem_geometry_is_rejected(tmp_path, sh):
    ext = _make_tree(tmp_path)
    with pytest.raises(SystemExit, match="--sh must be"):
        _apply(ext, frag=1, sh=sh)


def test_invalid_extension_root_is_rejected(tmp_path):
    argv = sys.argv
    sys.argv = ["patch_exl3_moe_pipeline.py", str(tmp_path / "nope")]
    try:
        with pytest.raises(SystemExit, match="invalid extension root"):
            OV.main()
    finally:
        sys.argv = argv


# --- against the real deployed tree, when it is reachable -------------------
#
# The synthetic anchors above pin the overlay's own contract; these pin that the
# contract still matches the revision the image actually builds from. They skip
# rather than fail when the checkout is absent (CI has no exllamav3 tree).

REAL_TREE = Path(os.environ.get("GLM53_EXLLAMAV3_EXT", "/nonexistent/exllamav3_ext"))


@pytest.mark.skipif(not REAL_TREE.is_dir(), reason="exllamav3 source tree not present")
def test_real_tree_still_carries_the_anchors():
    kernel = (REAL_TREE / "quant" / "exl3_moe_kernel.cuh").read_text()
    host = (REAL_TREE / "quant" / "exl3_moe.cu").read_text()
    bindings = (REAL_TREE / "bindings.cpp").read_text()
    assert kernel.count(OV.KERNEL_DEF_OLD) == 1
    assert host.count(OV.HOST_INCLUDE_OLD) == 1
    assert host.count(OV.HOST_HELPERS_ANCHOR) == 1
    assert host.count(OV.HOST_DISPATCH_OLD) == 1
    assert bindings.count(OV.BINDINGS_INCLUDE_OLD) == 1
    assert bindings.count(OV.BINDINGS_DEF_OLD) == 1
