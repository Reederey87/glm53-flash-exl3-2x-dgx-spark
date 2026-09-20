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


class Launcher:
    """Throwaway copy of start.sh with the host tools stubbed out."""

    def __init__(self, tmp: Path, pipeline_label: str = "") -> None:
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
                f'#!/usr/bin/env bash\nprintf "%s" "{pipeline_label}"\nexit 0\n'
                if tool == "docker"
                else "#!/usr/bin/env bash\nexit 0\n"
            )
            p.chmod(0o755)
        self.env = {
            "PATH": f"{self.bin}{os.pathsep}/usr/bin:/bin",
            "HOME": str(home),
            "USER": "t42-launcher",
            "LC_ALL": "C",
            "TERM": "dumb",
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
    assert (
        "for _v in GLM53_KV_CAPACITY_LOG GLM53_APC_NO_STORE GLM53_EXL3_MOE_PIPELINE; do"
        in src
    )
    assert '-e "GLM53_EXL3_MOE_PIPELINE=$GLM53_EXL3_MOE_PIPELINE"' in src


def test_dockerfile_stamps_the_label_the_launcher_reads():
    dockerfile = DOCKERFILE.read_text()
    assert (
        "glm53.task42.pipeline=${GLM53_EXL3_MOE_PIPELINE_FRAG}x${GLM53_EXL3_MOE_PIPELINE_SH}"
        in dockerfile
    )
    # The reader and the stamp have to name the same label.
    assert "glm53.task42.pipeline" in START.read_text()


def test_env_example_documents_the_knob_and_its_refusals():
    env = ENV_EXAMPLE.read_text()
    assert "GLM53_EXL3_MOE_PIPELINE=0" in env
    assert "Rollback: GLM53_EXL3_MOE_PIPELINE=0" in env
    # The documented signature must be the shipped one, not the compile probe's.
    assert "STACK:32 B with 9 STL / 4 LDL" in env
    assert "STACK:40 B" not in env


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
