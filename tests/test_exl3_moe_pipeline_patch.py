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
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1]
OVERLAY = KIT / "overlay" / "patch_exl3_moe_pipeline.py"
START = KIT / "start.sh"
DOCKERFILE = KIT / "Dockerfile.e3-pipeline-layer"
ENV_EXAMPLE = KIT / "env.example"


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
    const bool gated = act_function != MOE_ACT_RELU2_NOGATE;
    auto had_gather_gu_in = [&]()
    {
        const half* in_ptr = hidden_state;
        if (gated)
            had_hf_r_128_inner<true, false>
            (
                in_ptr,
                temp_state_g + 128 * warp_idx,
                exp_gate_suh + 128 * token_off,
                0.088388347648f
            );
                had_hf_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_u + 128 * warp_idx,
                    exp_up_suh + 128 * token_off,
                    0.088388347648f
                );
    };
    had_gather_gu_in();
    auto gemm_up = [&](const half* in_addr, half* out_addr, const uint16_t* trellis, const int K) {};
    if (gated)
        gemm_up(temp_state_g, temp_intermediate_g, exp_gate_trellis, K_gate);
        gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);
}
"""

COMMON_CUH = """#pragma once
#define MOE_TILESIZE_K 32
#define MOE_TILESIZE_M 16
#define MOE_SH_STAGES 3
#define MOE_FRAG_STAGES 3
#define MOE_ACT_RELU2_NOGATE 2
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
    assert "template<int t_bits, int MOE_TILESIZE_N, int cb, bool shared_input>" in variant
    assert "if constexpr (!shared_input)" in variant
    assert "gemm_up(temp_state_g, temp_intermediate_u, exp_up_trellis, K_up)" in variant


def test_geometry_reaches_the_comp_unit(tmp_path):
    ext = _make_tree(tmp_path)
    _apply(ext, frag=1, sh=8)
    unit = (ext / "quant" / "comp_units" / "glm53_exl3_moe_pipeline.cu").read_text()
    assert "#define MOE_FRAG_STAGES 1" in unit
    assert "#define MOE_SH_STAGES 8" in unit
    assert "glm53_exl3_moe_pipeline_kernel<4, 256, 1, false>" in unit
    assert "glm53_exl3_moe_pipeline_kernel<4, 256, 1, true>" in unit
    assert "glm53_exl3_moe_pipeline_kernel<4, 256, 2, false>" in unit
    assert "glm53_exl3_moe_pipeline_kernel<4, 256, 2, true>" in unit
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
    # Lever 2: reuse is a second instance, fail-closed without pipeline/proof.
    assert 'getenv("GLM53_EXL3_MOE_REUSE")' in host
    assert "GLM53_EXL3_MOE_REUSE=1 requires GLM53_EXL3_MOE_PIPELINE=1" in host
    assert "gate_ptrs_suh.data_ptr() == up_ptrs_suh.data_ptr()" in host
    assert "glm53_exl3_moe_pipeline_kernel_for(cb_idx, reuse_armed && shared_suh)" in host
    assert "MOE_ACT_RELU2_NOGATE" in host


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


def test_missing_hadamard_anchor_fails_closed(tmp_path):
    ext = _make_tree(tmp_path)
    kernel = ext / "quant" / "exl3_moe_kernel.cuh"
    kernel.write_text(KERNEL_CUH.replace("temp_state_u + 128 * warp_idx", "temp_state_x + 128 * warp_idx"))
    with pytest.raises(SystemExit, match="exactly one anchor"):
        _apply(ext)


def test_missing_up_gemm_anchor_fails_closed(tmp_path):
    ext = _make_tree(tmp_path)
    kernel = ext / "quant" / "exl3_moe_kernel.cuh"
    kernel.write_text(
        KERNEL_CUH.replace(
            "gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);",
            "gemm_up(temp_state_x, temp_intermediate_u, exp_up_trellis, K_up);",
        )
    )
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
    assert kernel.count(OV.KERNEL_TEMPLATE_OLD) == 1
    assert kernel.count(OV.HAD_UP_OLD) == 1
    assert kernel.count(OV.GEMM_UP_OLD) == 1
    assert host.count(OV.HOST_INCLUDE_OLD) == 1
    assert host.count(OV.HOST_HELPERS_ANCHOR) == 1
    assert host.count(OV.HOST_DISPATCH_OLD) == 1
    assert bindings.count(OV.BINDINGS_INCLUDE_OLD) == 1
    assert bindings.count(OV.BINDINGS_DEF_OLD) == 1


# --- launcher wiring: the knob is fail-closed end to end --------------------
#
# The overlay's own TORCH_CHECK covers only a process that actually reaches the
# patched dispatcher. Two configurations reach stock code instead, and an arm
# that boots armed while running stock is uninterpretable: an image that carries
# no variant, and an armed knob with the fused expert path disabled. Both are
# refused by the launcher, so they are exercised against the real
# `validate_numeric_config` and the real `ensure_image` rather than a lifted
# fragment.

STUB_TOOLS = ("docker", "ssh", "scp", "rsync", "curl", "ip", "nvidia-smi")

# The launcher asks the image two questions: its identity key (the GLM53KEY
# format) and the task-42 capability label. The stubs answer both, so head and
# worker identity can be varied independently.
_DOCKER_STUB = """#!/usr/bin/env bash
case "$*" in
  *GLM53KEY*)              printf 'GLM53KEY %s\\n' "$GLM53_STUB_HEAD_KEY" ;;
  *glm53.task42.reuse*)    printf '%s' "$GLM53_STUB_REUSE_LABEL" ;;
  *glm53.task42.pipeline*) printf '%s' "$GLM53_STUB_PIPELINE_LABEL" ;;
esac
exit 0
"""

_SSH_STUB = """#!/usr/bin/env bash
case "$*" in
  *GLM53KEY*)              printf 'GLM53KEY %s\\n' "$GLM53_STUB_WORKER_KEY" ;;
  *glm53.task42.reuse*)    printf '%s' "$GLM53_STUB_REUSE_LABEL" ;;
  *glm53.task42.pipeline*) printf '%s' "$GLM53_STUB_PIPELINE_LABEL" ;;
esac
exit 0
"""

_PLAIN_STUB = "#!/usr/bin/env bash\nexit 0\n"


class Launcher:
    """Throwaway copy of start.sh with the host tools stubbed out."""

    def __init__(
        self,
        tmp: Path,
        pipeline_label: str = "",
        reuse_label: str = "",
        head_key: str = "key-head",
        worker_key: str = "key-head",
    ) -> None:
        self.repo = tmp / "repo"
        self.repo.mkdir()
        shutil.copy2(START, self.repo / "start.sh")
        shutil.copy2(ENV_EXAMPLE, self.repo / "env.example")
        text = START.read_text()
        assert text.rstrip().endswith('\nmain "$@"'), 'start.sh must end with main "$@"'
        # Drop the dispatcher so the prologue can be driven directly.
        (self.repo / "start.fn.sh").write_text(text.rstrip()[: -len('main "$@"')] + '"$@"\n')
        (self.repo / ".env").write_text("")
        home = tmp / "home"
        home.mkdir()
        self.bin = tmp / "bin"
        self.bin.mkdir()
        for tool in STUB_TOOLS:
            p = self.bin / tool
            p.write_text(
                _DOCKER_STUB if tool == "docker"
                else _SSH_STUB if tool == "ssh"
                else _PLAIN_STUB
            )
            p.chmod(0o755)
        self.env = {
            "PATH": f"{self.bin}{os.pathsep}/usr/bin:/bin",
            "HOME": str(home),
            "USER": "t42-launcher",
            "LC_ALL": "C",
            "TERM": "dumb",
            "GLM53_STUB_PIPELINE_LABEL": pipeline_label,
            "GLM53_STUB_REUSE_LABEL": reuse_label,
            "GLM53_STUB_HEAD_KEY": head_key,
            "GLM53_STUB_WORKER_KEY": worker_key,
        }

    def run(self, body: str, **overrides: str) -> subprocess.CompletedProcess[str]:
        env = dict(self.env)
        env.update(overrides)
        return subprocess.run(
            ["bash", "./start.fn.sh", "eval", body],
            cwd=self.repo, capture_output=True, text=True, env=env,
        )


STRICT = [
    ({}, 0, ""),
    ({"GLM53_EXL3_MOE_PIPELINE": "0"}, 0, ""),
    ({"GLM53_EXL3_MOE_PIPELINE": "1"}, 0, ""),
    # `:-` used to coerce these two to 0; the W41/W42 contract says "" is a value.
    ({"GLM53_EXL3_MOE_PIPELINE": ""}, 2, "must be exactly 0 or 1"),
    ({"GLM53_EXL3_MOE_PIPELINE": "2"}, 2, "must be exactly 0 or 1"),
    ({"GLM53_EXL3_MOE_PIPELINE": "yes"}, 2, "must be exactly 0 or 1"),
    ({"GLM53_EXL3_MOE_PIPELINE": "1", "EXL3_FUSED_MOE": "0"}, 2, "requires EXL3_FUSED_MOE=1"),
    ({"GLM53_EXL3_MOE_PIPELINE": "0", "EXL3_FUSED_MOE": "0"}, 0, ""),
    ({"GLM53_EXL3_MOE_PIPELINE": "", "EXL3_FUSED_MOE": "0"}, 2, "must be exactly 0 or 1"),
    ({"GLM53_EXL3_MOE_REUSE": "0"}, 0, ""),
    ({"GLM53_EXL3_MOE_REUSE": "1", "GLM53_EXL3_MOE_PIPELINE": "1"}, 0, ""),
    ({"GLM53_EXL3_MOE_REUSE": ""}, 2, "must be exactly 0 or 1"),
    ({"GLM53_EXL3_MOE_REUSE": "2"}, 2, "must be exactly 0 or 1"),
    ({"GLM53_EXL3_MOE_REUSE": "1"}, 2, "requires GLM53_EXL3_MOE_PIPELINE=1"),
    ({"GLM53_EXL3_MOE_REUSE": "1", "GLM53_EXL3_MOE_PIPELINE": "0"}, 2, "requires GLM53_EXL3_MOE_PIPELINE=1"),
    ({"GLM53_EXL3_MOE_REUSE": "1", "GLM53_EXL3_MOE_PIPELINE": "1", "EXL3_FUSED_MOE": "0"}, 2, "requires EXL3_FUSED_MOE=1"),
]


@pytest.mark.parametrize("caller,rc,want_err", STRICT)
def test_knob_is_strict_and_the_fused_path_is_required(tmp_path, caller, rc, want_err):
    launcher = Launcher(tmp_path)
    r = launcher.run('validate_numeric_config; printf "rc=%s\\n" "$?"', **caller)
    assert r.returncode == rc, (r.returncode, r.stdout, r.stderr)
    if want_err:
        assert want_err in r.stderr, r.stderr
    else:
        assert "rc=0" in r.stdout, r.stdout


def test_armed_knob_is_refused_when_the_image_carries_no_variant(tmp_path):
    launcher = Launcher(tmp_path, pipeline_label="")
    r = launcher.run('ensure_image; printf "rc=%s\\n" "$?"', GLM53_EXL3_MOE_PIPELINE="1")
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    assert "carries no register-cut kernel" in r.stderr
    assert "glm53.task42.pipeline absent" in r.stderr


def test_armed_knob_passes_when_the_image_carries_the_variant(tmp_path):
    launcher = Launcher(tmp_path, pipeline_label="1x8")
    r = launcher.run('ensure_image; printf "rc=%s\\n" "$?"', GLM53_EXL3_MOE_PIPELINE="1")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "pipeline geometry 1x8" in r.stdout
    assert "on both nodes" in r.stdout


def test_reuse_passes_when_the_image_carries_the_shared_input_variant(tmp_path):
    launcher = Launcher(tmp_path, pipeline_label="1x8", reuse_label="1")
    r = launcher.run(
        'ensure_image; printf "rc=%s\\n" "$?"',
        GLM53_EXL3_MOE_PIPELINE="1",
        GLM53_EXL3_MOE_REUSE="1",
    )
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "gate/up Hadamard reuse armed" in r.stdout


@pytest.mark.parametrize("skip_ship", [None, "1"])
def test_armed_knob_is_refused_on_a_heterogeneous_pair(tmp_path, skip_ship):
    """The kernel is compiled into the image, so a mismatched pair arms one node
    and not the other. `ensure_image` tolerates an unmatched worker under
    SKIP_SHIP=1 and after a failed post-ship key comparison, so the armed path
    has to refuse the pair explicitly rather than measure it."""
    launcher = Launcher(
        tmp_path, pipeline_label="1x8", head_key="key-f1s8", worker_key="key-v149"
    )
    overrides = {"GLM53_EXL3_MOE_PIPELINE": "1"}
    if skip_ship:
        overrides["SKIP_SHIP"] = skip_ship
    r = launcher.run('ensure_image; printf "rc=%s\\n" "$?"', **overrides)
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    assert "requires the same image on both nodes" in r.stderr
    assert "head=key-f1s8 worker=key-v149" in r.stderr


def test_unarmed_boot_still_tolerates_a_heterogeneous_pair(tmp_path):
    """Stock behaviour: the existing warn-and-continue path must be preserved."""
    launcher = Launcher(
        tmp_path, pipeline_label="", head_key="key-a", worker_key="key-b"
    )
    r = launcher.run('ensure_image; printf "rc=%s\\n" "$?"', GLM53_EXL3_MOE_PIPELINE="0")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "requires the same image on both nodes" not in r.stderr


def test_unarmed_boot_is_unaffected_by_a_missing_variant(tmp_path):
    """The capability check must not turn stock images into a boot failure."""
    launcher = Launcher(tmp_path, pipeline_label="")
    r = launcher.run('ensure_image; printf "rc=%s\\n" "$?"', GLM53_EXL3_MOE_PIPELINE="0")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "carries no register-cut kernel" not in r.stderr


def test_knob_uses_an_unset_only_default_and_joins_the_strict_bool_loop():
    src = START.read_text()
    assert 'GLM53_EXL3_MOE_PIPELINE="${GLM53_EXL3_MOE_PIPELINE-0}"' in src
    assert 'GLM53_EXL3_MOE_PIPELINE="${GLM53_EXL3_MOE_PIPELINE:-0}"' not in src
    assert 'GLM53_EXL3_MOE_REUSE="${GLM53_EXL3_MOE_REUSE-0}"' in src
    assert 'GLM53_EXL3_MOE_REUSE="${GLM53_EXL3_MOE_REUSE:-0}"' not in src
    assert (
        "for _v in GLM53_KV_CAPACITY_LOG GLM53_APC_NO_STORE "
        "GLM53_EXL3_MOE_PIPELINE GLM53_EXL3_MOE_REUSE; do"
        in src
    )
    assert '-e "GLM53_EXL3_MOE_PIPELINE=$GLM53_EXL3_MOE_PIPELINE"' in src
    assert '-e "GLM53_EXL3_MOE_REUSE=$GLM53_EXL3_MOE_REUSE"' in src


def test_dockerfile_stamps_the_label_the_launcher_reads():
    dockerfile = DOCKERFILE.read_text()
    assert (
        "glm53.task42.pipeline=${GLM53_EXL3_MOE_PIPELINE_FRAG}x${GLM53_EXL3_MOE_PIPELINE_SH}"
        in dockerfile
    )
    assert "glm53.task42.reuse=1" in dockerfile
    # The reader and the stamp have to name the same label.
    src = START.read_text()
    assert "glm53.task42.pipeline" in src
    assert "glm53.task42.reuse" in src
    assert "COPY overlay/exl3.py" in dockerfile


def test_env_example_documents_the_knob_and_its_refusals():
    env = ENV_EXAMPLE.read_text()
    assert "GLM53_EXL3_MOE_PIPELINE=0" in env
    assert "Rollback: GLM53_EXL3_MOE_PIPELINE=0" in env
    # The documented signature must be the shipped one, not the compile probe's.
    assert "STACK:32 B with 9 STL / 4 LDL" in env
    assert "STACK:40 B" not in env
    assert "GLM53_EXL3_MOE_REUSE=0" in env
    assert "Rollback: GLM53_EXL3_MOE_REUSE=0" in env


def test_caller_export_wins_over_dotenv_including_empty(tmp_path):
    """Setness-aware caller-wins: an explicitly EMPTY caller export is a value.

    The generic `[ -n ]` replay skips empty caller values, so the strict knobs
    carry a setness-aware exception. This knob is a strict bool under the same
    rule, so `GLM53_EXL3_MOE_PIPELINE= ./start.sh validate` must not silently
    fall back to the `.env` value and pass.
    """
    launcher = Launcher(tmp_path)
    (launcher.repo / ".env").write_text("GLM53_EXL3_MOE_PIPELINE=0\n")
    body = 'printf "%s\\n" "${GLM53_EXL3_MOE_PIPELINE-<unset>}"'

    assert launcher.run(body).stdout.strip() == "0"          # .env only
    assert launcher.run(body, GLM53_EXL3_MOE_PIPELINE="1").stdout.strip() == "1"
    # Empty caller value beats .env, and is then rejected by the validator.
    assert launcher.run(body, GLM53_EXL3_MOE_PIPELINE="").stdout.strip() == ""
    r = launcher.run('validate_numeric_config; printf "rc=%s\\n" "$?"',
                     GLM53_EXL3_MOE_PIPELINE="")
    assert r.returncode == 2
    assert "must be exactly 0 or 1" in r.stderr


def test_reuse_caller_export_wins_over_dotenv_including_empty(tmp_path):
    launcher = Launcher(tmp_path)
    (launcher.repo / ".env").write_text("GLM53_EXL3_MOE_REUSE=0\n")
    body = 'printf "%s\\n" "${GLM53_EXL3_MOE_REUSE-<unset>}"'
    assert launcher.run(body).stdout.strip() == "0"
    assert launcher.run(body, GLM53_EXL3_MOE_REUSE="1").stdout.strip() == "1"
    assert launcher.run(body, GLM53_EXL3_MOE_REUSE="").stdout.strip() == ""
    r = launcher.run('validate_numeric_config; printf "rc=%s\\n" "$?"',
                     GLM53_EXL3_MOE_REUSE="")
    assert r.returncode == 2
    assert "must be exactly 0 or 1" in r.stderr


def test_reuse_is_refused_when_the_image_carries_no_variant(tmp_path):
    launcher = Launcher(tmp_path, pipeline_label="")
    r = launcher.run(
        'ensure_image; printf "rc=%s\\n" "$?"',
        GLM53_EXL3_MOE_PIPELINE="1",
        GLM53_EXL3_MOE_REUSE="1",
    )
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    assert "carries no register-cut kernel" in r.stderr


def test_reuse_is_refused_on_a_pipeline_image_without_the_skip(tmp_path):
    """A pre-lever-2 pipeline cubin still compiles two Hadamards."""
    launcher = Launcher(tmp_path, pipeline_label="1x8", reuse_label="")
    r = launcher.run(
        'ensure_image; printf "rc=%s\\n" "$?"',
        GLM53_EXL3_MOE_PIPELINE="1",
        GLM53_EXL3_MOE_REUSE="1",
    )
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    assert "carries no gate/up Hadamard-reuse kernel" in r.stderr
    assert "glm53.task42.reuse absent" in r.stderr


# --- the parity comparator must fail closed --------------------------------
#
# cmp.py is a receipt tool, but it is used as a gate, so it has to exit non-zero
# rather than print a verdict and return 0. The NaN case is the one that bit:
# `max(0.0, nan)` is 0.0, so a non-finite tensor used to report PARITY OK.
# These need torch, so they skip on a host without it (the kit's pattern for
# image-only checks) and run inside the image.

CMP = KIT / "local" / "task42-receipts-20260920" / "cmp.py"


def _torch():
    return pytest.importorskip("torch", reason="cmp.py needs torch; run inside the image")


def _write_arm(out: Path, arm: str, tensor, case: dict) -> None:
    (out / f"parity-{arm}.json").write_text(
        json.dumps(
            {"arm": arm, "tokens": str(case["tokens"]), "cap": 32, "cases": [case]}
        )
    )
    tensor_name = f"parity-{arm}.json.t{case['tokens']}.s{case['skew']}.pt"
    _torch().save(tensor, out / tensor_name)


def _cmp_case(tokens: int = 2, skew: float = 1.0, shape: tuple = (2, 4)) -> dict:
    return {"tokens": tokens, "skew": skew, "path": "none", "shape": list(shape)}


def _run_cmp(out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CMP), str(out)], capture_output=True, text=True
    )


def test_cmp_accepts_identical_outputs(tmp_path):
    torch = _torch()
    out = tmp_path / "out"
    out.mkdir()
    case = _cmp_case()
    t = torch.ones(2, 4, dtype=torch.float32)
    _write_arm(out, "ctrl", t, case)
    _write_arm(out, "var", t.clone(), case)
    r = _run_cmp(out)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "VERDICT: PARITY OK" in r.stdout
    assert "bit-exact: 1" in r.stdout


def test_cmp_rejects_nonfinite_output(tmp_path):
    """`max(0.0, nan)` is 0.0, so this must be checked explicitly, not inferred."""
    torch = _torch()
    out = tmp_path / "out"
    out.mkdir()
    case = _cmp_case()
    t = torch.ones(2, 4, dtype=torch.float32)
    bad = t.clone()
    bad[0, 0] = float("nan")
    _write_arm(out, "ctrl", t, case)
    _write_arm(out, "var", bad, case)
    r = _run_cmp(out)
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    assert "non-finite" in r.stderr
    assert "PARITY OK" not in r.stdout


def test_cmp_rejects_a_tolerance_failure(tmp_path):
    torch = _torch()
    out = tmp_path / "out"
    out.mkdir()
    case = _cmp_case()
    t = torch.ones(2, 4, dtype=torch.float32)
    _write_arm(out, "ctrl", t, case)
    _write_arm(out, "var", t + 1.0, case)
    r = _run_cmp(out)
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    assert "tolerance" in r.stderr
    assert "PARITY OK" not in r.stdout


def test_cmp_rejects_mismatched_case_lists(tmp_path):
    """A short or reordered list must not read as a pass via a truncating zip."""
    torch = _torch()
    out = tmp_path / "out"
    out.mkdir()
    _write_arm(out, "ctrl", torch.ones(2, 4), _cmp_case(tokens=2, skew=1.0))
    _write_arm(out, "var", torch.ones(2, 4), _cmp_case(tokens=3, skew=1.0))
    r = _run_cmp(out)
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    assert "case lists differ" in r.stderr


def test_cmp_rejects_a_missing_tensor(tmp_path):
    torch = _torch()
    out = tmp_path / "out"
    out.mkdir()
    case = _cmp_case()
    _write_arm(out, "ctrl", torch.ones(2, 4), case)
    (out / "parity-var.json").write_text(
        json.dumps({"arm": "1", "tokens": "2", "cap": 32, "cases": [case]})
    )
    r = _run_cmp(out)
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    assert "missing tensor" in r.stderr
