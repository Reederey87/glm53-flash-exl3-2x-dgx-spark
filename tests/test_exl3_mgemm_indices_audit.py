#!/usr/bin/env python3
"""CPU tests for the task 37 unenforced 128-slot v_indices audit.

Every case runs against a synthetic ExLlamaV3 tree so the suite is hermetic and
does not need the 100+ GiB checkout or a GPU.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_exl3_mgemm_indices.py"
SPEC = importlib.util.spec_from_file_location("exl3_mgemm_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

KERNEL_HEADER = """\
#define MAX_INDICES 128

__device__ int64_t v_indices[128];
__device__ half v_weights[128];
__device__ int bszm_sync;

__global__ __launch_bounds__(256)
void exl3_gemm_kernel(int x)
{
    int y = x;
}

__global__ __launch_bounds__(256)
void exl3_mgemm_kernel(int bszm)
{
    if (min_index >= 0)
    {
        for (int i = 0; i < bszm; ++i)
        {
            v_indices[i] = -1;
            v_weights[i] = __float2half(0.0f);
        }
    }
}
"""

KERNEL_HEADER_ORPHAN = KERNEL_HEADER.replace(
    "void exl3_gemm_kernel(int x)\n{\n    int y = x;\n}",
    "void exl3_gemm_kernel(int x)\n{\n    v_indices[x] = 0;\n}",
)

LINEAR_CPP = """\
void BC_LinearEXL3::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)
{
    if (x.numel() == x.size(-1))
    {
        exl3_gemm_gr(x, trellis, y, suh, xh, svh, -1, mcg, mul1, 0, graph);
    }
    else
    {
        exl3_gemm(x, trellis, y, suh, xh_, svh, -1, mcg, mul1, 0);
    }
}

void BC_LinearEXL3::run(const at::Tensor& x, at::Tensor& y)
{
    run_gr(x, y, nullptr);
}
"""

LINEAR_CPP_MGEMM = LINEAR_CPP.replace(
    "        exl3_gemm(x, trellis, y, suh, xh_, svh, -1, mcg, mul1, 0);",
    "        exl3_gemm(x, trellis, y, suh, xh_, svh, -1, mcg, mul1, 0);\n"
    "        exl3_mgemm_gr(x, trellis, y, suh, xh_, svh, {}, {}, 0, 0, 0, 0, 8, 0, graph);",
)

GEMM_CU = """\
int exl3_mgemm_gr(const at::Tensor& A)
{
    TORCH_CHECK(num_tokens == 1 && min_index < 0 && !weights,
                "exl3_mgemm: per-matrix widths incompatible with multi-token");
    return 0;
}
"""

SERVING_PY = """\
from ...model.config import Config
from .exl3_lib.quantize import q


class LinearEXL3:
    def forward(self, x):
        return q(x)
"""

SERVING_PY_MGEMM = SERVING_PY + """\

def bad(x):
    return ext.exl3_mgemm(x, x, x, x, x, x, None, None, 4, 0, 1, 1, -1, -1, 0, 1)
"""

NATIVE_PY = """\
def native(x):
    return ext.exl3_mgemm(x, x, x, x, x, x, None, None, 4, 0, 1, 1, 0, 8, 0, 1)
"""

BINDINGS_CPP = """\
void bind(py::module& m)
{
    m.def("exl3_gemm", &exl3_gemm, "exl3_gemm");
    m.def("exl3_mgemm", &exl3_mgemm, "exl3_mgemm");
    m.def("exl3_moe", &exl3_moe, "exl3_moe");
}
"""

OVERLAY_OK = """\
SYMBOL = "exllamav3_ext.exl3_moe"
MODULE = "exllamav3.modules.quant.exl3"
"""

OVERLAY_MGEMM = """\
SYMBOL = "exllamav3_ext.exl3_mgemm"
MODULE = "exllamav3.modules.quant.exl3"
"""

OVERLAY_NO_SYMBOL = """\
MODULE = "exllamav3.modules.quant.exl3"
"""


def build_tree(
    tmp_path: Path,
    *,
    kernel: str = KERNEL_HEADER,
    linear: str = LINEAR_CPP,
    gemm: str = GEMM_CU,
    serving: str = SERVING_PY,
) -> Path:
    root = tmp_path / "exllamav3"
    pkg = root / "exllamav3"
    (pkg / "exllamav3_ext/quant").mkdir(parents=True)
    (pkg / "exllamav3_ext/libtorch").mkdir(parents=True)
    (pkg / "model").mkdir(parents=True)
    (pkg / "util").mkdir(parents=True)
    (pkg / "modules/quant/exl3_lib").mkdir(parents=True)

    (pkg / "exllamav3_ext/quant/exl3_gemm_kernel.cuh").write_text(kernel)
    (pkg / "exllamav3_ext/quant/exl3_gemm.cu").write_text(gemm)
    (pkg / "exllamav3_ext/libtorch/linear.cpp").write_text(linear)
    (pkg / "exllamav3_ext/bindings.cpp").write_text(BINDINGS_CPP)

    for rel in (
        "__init__.py",
        "model/__init__.py",
        "util/__init__.py",
        "modules/__init__.py",
        "modules/quant/__init__.py",
        "modules/quant/exl3_lib/__init__.py",
    ):
        (pkg / rel).write_text("")
    (pkg / "model/config.py").write_text("Config = object\n")
    (pkg / "util/helpers.py").write_text("q = None\n")
    (pkg / "modules/quant/exl3_lib/quantize.py").write_text("q = None\n")
    (pkg / "modules/quant/exl3.py").write_text(serving)
    (pkg / "modules/native.py").write_text(NATIVE_PY)
    return root


def run(tmp_path: Path, shapes=None, **kwargs):
    defaults = {
        "kernel": KERNEL_HEADER,
        "linear": LINEAR_CPP,
        "gemm": GEMM_CU,
        "serving": SERVING_PY,
    }
    defaults.update(kwargs)
    root = build_tree(tmp_path, **defaults)
    return MODULE.audit(
        root,
        root / "exllamav3",
        "exllamav3.modules.quant.exl3",
        shapes or MODULE.worst_case_slots(8, 4, 7),
    )


def test_scratch_writes_are_confined_to_the_mgemm_kernel(tmp_path):
    scratch = MODULE.kernel_scratch(KERNEL_HEADER)
    assert scratch["max_indices"] == 128
    assert scratch["writer_kernels"] == ["exl3_mgemm_kernel"]
    assert scratch["writes_outside_mgemm_kernel"] == []
    assert sorted(scratch["arrays"]) == ["v_indices", "v_weights"]


def test_orphan_writer_is_reported(tmp_path):
    scratch = MODULE.kernel_scratch(KERNEL_HEADER_ORPHAN)
    assert scratch["writes_outside_mgemm_kernel"], "orphan write must be surfaced"
    assert "exl3_gemm_kernel" in scratch["writer_kernels"]


def test_missing_declaration_aborts():
    with pytest.raises(MODULE.Abort):
        MODULE.kernel_scratch(KERNEL_HEADER.replace("#define MAX_INDICES 128", ""))


def test_missing_scratch_arrays_abort():
    with pytest.raises(MODULE.Abort):
        MODULE.kernel_scratch(KERNEL_HEADER.replace("__device__ half v_weights[128];", ""))


def test_serving_path_calls_gemm_and_never_mgemm():
    path = MODULE.serving_path(LINEAR_CPP)
    assert path["calls_exl3_gemm_gr"] is True
    assert path["calls_exl3_gemm"] is True
    assert path["calls_exl3_mgemm"] is False


def test_serving_path_detects_mgemm_regression():
    path = MODULE.serving_path(LINEAR_CPP_MGEMM)
    assert path["calls_exl3_mgemm"] is True


def test_serving_path_aborts_without_the_gemm_call():
    with pytest.raises(MODULE.Abort):
        MODULE.serving_path(LINEAR_CPP.replace("exl3_gemm_gr(", "something_else("))


def test_gemm_guards_present():
    guards = MODULE.gemm_guards(GEMM_CU)
    assert guards["sliced_mode_requires_min_index_negative"] is True
    assert guards["per_matrix_widths_require_min_index_negative"] is True


def test_gemm_guards_abort_on_missing_anchor():
    with pytest.raises(MODULE.Abort):
        MODULE.gemm_guards(GEMM_CU.replace("min_index < 0", "min_index >= 0"))


def test_import_closure_follows_relative_imports(tmp_path):
    root = build_tree(tmp_path)
    closure = MODULE.import_closure(root / "exllamav3", ["exllamav3.modules.quant.exl3"])
    assert "exllamav3.model.config" in closure, "relative `...model.config` must resolve"
    assert "exllamav3.modules.quant.exl3_lib.quantize" in closure
    assert "exllamav3.modules.native" not in closure, "unimported native module must stay out"


def test_call_sites_enumerated(tmp_path):
    root = build_tree(tmp_path)
    sites = MODULE.mgemm_call_sites(root / "exllamav3")
    assert "exllamav3.modules.native" in sites
    assert sites["exllamav3.modules.native"] == [2]


def test_call_site_names_match_closure_namespace(tmp_path):
    """Guards the fail-open bug: keys and closure must share one namespace."""
    root = build_tree(tmp_path, serving=SERVING_PY_MGEMM)
    sites = MODULE.mgemm_call_sites(root / "exllamav3")
    closure = MODULE.import_closure(
        root / "exllamav3", ["exllamav3.modules.quant.exl3"]
    )
    assert "exllamav3.modules.quant.exl3" in sites
    assert "exllamav3.modules.quant.exl3" in closure


def test_verdict_not_reachable(tmp_path):
    report = run(tmp_path)
    assert report["verdict"] == "NOT_REACHABLE"
    assert report["decisive_reachable"] == []
    assert report["binding_reaches_mgemm"] == []
    assert report["serving_path"]["calls_exl3_mgemm"] is False


def test_verdict_reachable_overflow(tmp_path):
    """32 concurrent sequences x top_k 8 = 256 slots, above MAX_INDICES."""
    report = run(
        tmp_path,
        shapes=MODULE.worst_case_slots(8, 32, 7),
        serving=SERVING_PY_MGEMM,
    )
    assert report["verdict"] == "REACHABLE_OVERFLOW"
    assert report["decisive_reachable"] == ["exllamav3.modules.quant.exl3"]


def test_verdict_reachable_ok_when_shapes_fit(tmp_path):
    report = run(tmp_path, serving=SERVING_PY_MGEMM)
    assert report["verdict"] == "REACHABLE_OK"


def test_closure_is_advisory_only(tmp_path):
    """A native module in the closure is context, not a reachability verdict."""
    root = build_tree(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "patch_x.py").write_text(
        OVERLAY_OK + 'NATIVE = "exllamav3.modules.native"\n'
    )
    report = MODULE.audit(
        root,
        root / "exllamav3",
        "exllamav3.modules.quant.exl3",
        MODULE.worst_case_slots(8, 32, 7),
        overlay,
    )
    assert "exllamav3.modules.native" in report["advisory_closure_call_site_modules"]
    assert report["verdict"] == "NOT_REACHABLE"
    assert report["decisive_reachable"] == []


def test_orphan_writer_forces_abort(tmp_path):
    report = run(tmp_path, kernel=KERNEL_HEADER_ORPHAN)
    assert report["verdict"] == "ABORT"


def test_worst_case_slots_is_batch_times_topk():
    shapes = MODULE.worst_case_slots(top_k=8, max_num_seqs=4, draft_tokens=7)
    assert shapes["num_tokens_1_slots_batch_x_topk"] == 32
    assert shapes["num_tokens_gt_1_slots_bszm"] == 32


def test_missing_source_aborts(tmp_path):
    root = build_tree(tmp_path)
    (root / "exllamav3/exllamav3_ext/libtorch/linear.cpp").unlink()
    with pytest.raises(MODULE.Abort):
        MODULE.audit(
            root,
            root / "exllamav3",
            "exllamav3.modules.quant.exl3",
            MODULE.worst_case_slots(8, 4, 7),
        )


def test_cli_exit_codes(tmp_path):
    root = build_tree(tmp_path)
    ok = subprocess.run(
        [sys.executable, str(SCRIPT), "--exl3-root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["verdict"] == "NOT_REACHABLE"

    bad = subprocess.run(
        [sys.executable, str(SCRIPT), "--exl3-root", str(tmp_path / "nope")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 1
    assert json.loads(bad.stdout)["verdict"] == "ABORT"

    overflow_root = build_tree(tmp_path / "overflow", serving=SERVING_PY_MGEMM)
    overflow = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--exl3-root",
            str(overflow_root),
            "--max-num-seqs",
            "32",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert overflow.returncode == 2
    assert json.loads(overflow.stdout)["verdict"] == "REACHABLE_OVERFLOW"


def test_overlay_dir_seeds_closure(tmp_path):
    root = build_tree(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "patch_x.py").write_text(
        OVERLAY_OK + 'NATIVE = "exllamav3.modules.native"\n'
    )
    report = MODULE.audit(
        root,
        root / "exllamav3",
        "exllamav3.modules.quant.exl3",
        MODULE.worst_case_slots(8, 32, 7),
        overlay,
    )
    assert "exllamav3.modules.native" in report["advisory_closure_call_site_modules"], (
        "an overlay that names a native mgemm module must appear in the advisory list"
    )


def test_overlay_calling_mgemm_entry_point_is_reachable(tmp_path):
    root = build_tree(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "patch_x.py").write_text(OVERLAY_MGEMM)
    report = MODULE.audit(
        root,
        root / "exllamav3",
        "exllamav3.modules.quant.exl3",
        MODULE.worst_case_slots(8, 32, 7),
        overlay,
    )
    assert report["verdict"] == "REACHABLE_OVERFLOW"
    assert report["binding_reaches_mgemm"] == ["exl3_mgemm"]


def test_binding_check_is_clean_for_the_real_overlay_symbols(tmp_path):
    root = build_tree(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "patch_x.py").write_text(OVERLAY_OK)
    report = MODULE.audit(
        root,
        root / "exllamav3",
        "exllamav3.modules.quant.exl3",
        MODULE.worst_case_slots(8, 32, 7),
        overlay,
    )
    assert report["verdict"] == "NOT_REACHABLE"
    assert report["binding_reaches_mgemm"] == []
    assert report["overlay_ext_symbols"] == ["exl3_moe"]
    assert report["mgemm_entry_points"] == ["exl3_mgemm"]


def test_overlay_without_ext_symbols_aborts(tmp_path):
    root = build_tree(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "patch_x.py").write_text(OVERLAY_NO_SYMBOL)
    with pytest.raises(MODULE.Abort):
        MODULE.audit(
            root,
            root / "exllamav3",
            "exllamav3.modules.quant.exl3",
            MODULE.worst_case_slots(8, 4, 7),
            overlay,
        )


def test_bindings_without_mgemm_entry_aborts(tmp_path):
    root = build_tree(tmp_path)
    bindings = root / "exllamav3/exllamav3_ext/bindings.cpp"
    bindings.write_text(BINDINGS_CPP.replace('m.def("exl3_mgemm"', 'm.def("exl3_other"'))
    with pytest.raises(MODULE.Abort):
        MODULE.audit(
            root,
            root / "exllamav3",
            "exllamav3.modules.quant.exl3",
            MODULE.worst_case_slots(8, 4, 7),
        )


def test_stub_namespace_is_not_descended(tmp_path):
    """`exllamav3.modules` is stubbed at runtime, so modules/__init__ never runs."""
    root = build_tree(tmp_path)
    pkg = root / "exllamav3"
    (pkg / "modules/__init__.py").write_text("from . import native\n")
    closure = MODULE.import_closure(
        pkg, ["exllamav3.modules.quant.exl3", "exllamav3.modules"]
    )
    assert "exllamav3.modules" in closure
    assert "exllamav3.modules.native" not in closure, (
        "the synthetic stub namespace must not pull in the native model modules"
    )
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "patch_x.py").write_text(
        OVERLAY_OK + 'MODULE = "exllamav3.modules"\n'
    )
    report = MODULE.audit(
        root,
        pkg,
        "exllamav3.modules.quant.exl3",
        MODULE.worst_case_slots(8, 32, 7),
        overlay,
    )
    assert report["verdict"] == "NOT_REACHABLE"


def test_overlay_dir_missing_aborts(tmp_path):
    root = build_tree(tmp_path)
    with pytest.raises(MODULE.Abort):
        MODULE.audit(
            root,
            root / "exllamav3",
            "exllamav3.modules.quant.exl3",
            MODULE.worst_case_slots(8, 4, 7),
            tmp_path / "absent",
        )
