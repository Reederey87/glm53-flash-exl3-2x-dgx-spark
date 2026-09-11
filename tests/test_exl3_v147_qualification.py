"""Host fixtures for the ExLlamaV3 image-qualification blockers.

Task 16 introduced these for the v1.4.7 pin (aarch64 CPU-MoE stub, native
ticket-scheduler skip, NullConfig namespace, plus the existing fat-kernel
binding anchors); task 35 carried them to v1.4.9, which adds the
`exl3_moe_cpu_has_avx512_bw` probe. No torch, no CUDA.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = KIT_ROOT / "tests" / "fixtures" / "exl3-v147"
AARCH64 = KIT_ROOT / "overlay" / "patch_exl3_ext_aarch64.py"
TICKET = KIT_ROOT / "overlay" / "patch_exl3_ticket_scheduler.py"
FAT = KIT_ROOT / "overlay" / "patch_exl3_fat_kernel.py"
NAMESPACE = KIT_ROOT / "overlay" / "exl3_namespace.py"
TICKET_DIR = KIT_ROOT / "overlay" / "exl3-ticket"
PRISTINE = TICKET_DIR / "pristine"
PATCHED = TICKET_DIR / "patched"
FILES = (
    "exl3_devctx.cu",
    "exl3_devctx.cuh",
    "exl3_moe.cu",
    "exl3_moe.cuh",
    "exl3_moe_common.cuh",
    "exl3_moe_kernel.cuh",
)
PIN = "5be886578ec80324c2c715269387be2058724b6e"
PIN_VERSION = "1.4.9"

# The CPU-MoE pybind surface as registered at v1.4.9. `has_avx512_bw` is the
# symbol v1.4.9 adds; `set_memops`/`worker_run` live in cpu/moe_handoff.cu,
# every other name is defined by the aarch64 stub that replaces moe_mul1.cpp.
V149_CPU_BINDINGS = """\
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "cpu/moe_mul1.h"
#include "cpu/moe_handoff.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("exl3_moe_cpu_set_prof", &exl3_moe_cpu_set_prof, "exl3_moe_cpu_set_prof");
    m.def("exl3_moe_cpu_make_layer", &exl3_moe_cpu_make_layer, "exl3_moe_cpu_make_layer");
    m.def("exl3_moe_cpu_free_layer", &exl3_moe_cpu_free_layer, "exl3_moe_cpu_free_layer");
    m.def("exl3_moe_cpu_forward", &exl3_moe_cpu_forward, "exl3_moe_cpu_forward");
    m.def("exl3_moe_cpu_forward_raw", &exl3_moe_cpu_forward_raw, "exl3_moe_cpu_forward_raw");
    m.def("exl3_moe_cpu_stage_experts", &exl3_moe_cpu_stage_experts, "exl3_moe_cpu_stage_experts");
    m.def("exl3_moe_cpu_pool_stress", &exl3_moe_cpu_pool_stress, "exl3_moe_cpu_pool_stress");
    m.def("exl3_moe_cpu_set_memops", &exl3_moe_cpu_set_memops, "exl3_moe_cpu_set_memops");
    m.def("exl3_moe_cpu_worker_run", &exl3_moe_cpu_worker_run, "exl3_moe_cpu_worker_run");
    m.def("exl3_moe_cpu_has_avx2", &exl3_moe_cpu_has_avx2, "exl3_moe_cpu_has_avx2");
    m.def("exl3_moe_cpu_has_avx512_bw", &exl3_moe_cpu_has_avx512_bw, "exl3_moe_cpu_has_avx512_bw");
    m.def("exl3_moe_cpu_has_avx512_vnni", &exl3_moe_cpu_has_avx512_vnni, "exl3_moe_cpu_has_avx512_vnni");
    m.def("exl3_moe_cpu_has_avx512_vbmi", &exl3_moe_cpu_has_avx512_vbmi, "exl3_moe_cpu_has_avx512_vbmi");
}
"""

V149_CPU_SYMBOL_COUNT = 13

HANDOFF_SYMBOLS = (
    "void exl3_moe_cpu_set_memops() {}\n"
    "void exl3_moe_cpu_worker_run() {}\n"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script: Path, *args: str):
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
    )


def _ext_with_quant(tmp_path: Path, source: Path) -> Path:
    ext = tmp_path / "ext"
    (ext / "quant").mkdir(parents=True)
    (ext / "cpu").mkdir(parents=True)
    (ext / "parallel").mkdir(parents=True)
    for name in FILES:
        shutil.copyfile(source / name, ext / "quant" / name)
    return ext


def _native_tree(tmp_path: Path) -> Path:
    return _ext_with_quant(tmp_path, FIXTURES / "quant")


def test_pin_constants_match_v147():
    overlay = (KIT_ROOT / "overlay" / "exl3.py").read_text()
    dockerfile = (KIT_ROOT / "Dockerfile").read_text()
    assert f'EXLLAMAV3_COMMIT = "{PIN}"' in overlay
    assert f'EXLLAMAV3_VERSION = "{PIN_VERSION}"' in overlay
    assert f"ARG EXLLAMAV3_COMMIT={PIN}" in dockerfile
    assert dockerfile.split("ARG EXLLAMAV3_COMMIT=")[1].startswith(PIN)
    assert "COPY overlay/exl3_namespace.py" in dockerfile
    assert "Do not copy published x86_64 wheels" in dockerfile
    assert "GLM53_AARCH64_CPU_MOE_STUB" in dockerfile
    assert "b'__builtin_ia32_pause'" in dockerfile
    assert "GLM53_AARCH64_CPU_PAUSE_STUB" in dockerfile
    assert "is_f16c_supported" in dockerfile
    # The version assert is parameterized so the pin can move without editing
    # the fail-closed check itself; keep it in lockstep with the commit ARG.
    assert f"ARG EXLLAMAV3_VERSION={PIN_VERSION}" in dockerfile
    assert "'${EXLLAMAV3_VERSION}' in ver" in dockerfile
    assert "python3 /opt/glm53/patch_exl3_ticket_scheduler.py" in dockerfile
    fat_pos = dockerfile.index("patch_exl3_fat_kernel.py /tmp/exllamav3")
    ticket_pos = dockerfile.index("patch_exl3_ticket_scheduler.py /tmp/exllamav3")
    aarch_pos = dockerfile.index("patch_exl3_ext_aarch64.py /tmp/exllamav3")
    assert aarch_pos < fat_pos < ticket_pos


def test_aarch64_stubs_x86_cpu_moe(tmp_path):
    ext = tmp_path / "ext"
    (ext / "cpu").mkdir(parents=True)
    (ext / "parallel").mkdir(parents=True)
    shutil.copyfile(FIXTURES / "moe_mul1.cpp", ext / "cpu" / "moe_mul1.cpp")
    shutil.copyfile(FIXTURES / "moe_handoff.cu", ext / "cpu" / "moe_handoff.cu")
    shutil.copyfile(FIXTURES / "all_reduce_cpu.cu", ext / "parallel" / "all_reduce_cpu.cu")
    # leftover AVX sources that the installer must also overwrite
    (ext / "avx2_target.cpp").write_text('#include <immintrin.h>\n')
    (ext / "avx512_target.cpp").write_text('#include <immintrin.h>\n')
    (ext / "parallel" / "all_reduce_cpu_avx2.cpp").write_text('#include <immintrin.h>\n')
    (ext / "parallel" / "all_reduce_cpu_avx512.cpp").write_text('#include <immintrin.h>\n')
    r = _run(AARCH64, str(ext))
    assert r.returncode == 0, r.stderr + r.stdout
    assert "cpu_moe=stubbed" in r.stdout
    assert "cpu/moe_handoff.cu" in r.stdout
    assert "parallel/all_reduce_cpu.cu" in r.stdout
    stub = (ext / "cpu" / "moe_mul1.cpp").read_text()
    assert "GLM53_AARCH64_CPU_MOE_STUB" in stub
    assert "<immintrin.h>" not in stub
    assert "exl3_moe_cpu_make_layer" in stub
    assert "exl3_moe_cpu_forward_raw" in stub
    # v1.4.9 adds this probe; bindings.cpp registers it and the stub is its
    # only definition site, so it must survive the wholesale replacement.
    assert "bool exl3_moe_cpu_has_avx512_bw() { return false; }" in stub
    leftover = [
        p for p in ext.rglob("*")
        if p.suffix in {".c", ".cpp", ".h", ".hpp", ".cu", ".cuh"}
        and (
            b"immintrin.h" in p.read_bytes()
            or b"__builtin_ia32_pause" in p.read_bytes()
            or b"_mm_pause" in p.read_bytes()
            or b'__attribute__((target("avx' in p.read_bytes()
            or b"__builtin_cpu_supports" in p.read_bytes()
        )
    ]
    assert leftover == []
    handoff = (ext / "cpu" / "moe_handoff.cu").read_text()
    assert "GLM53_AARCH64_CPU_PAUSE_STUB" in handoff
    assert "__builtin_ia32_pause" not in handoff
    assert "_mm_pause" not in handoff
    assert "is_f16c_supported" in (ext / "avx2_target.h").read_text()
    assert "bool is_f16c_supported() { return false; }" in (
        ext / "avx2_target.cpp"
    ).read_text()
    r2 = _run(AARCH64, str(ext))
    assert r2.returncode == 0, r2.stderr + r2.stdout
    assert "cpu_moe=already" in r2.stdout


def test_aarch64_allows_missing_cpu_moe_on_old_pin(tmp_path):
    ext = tmp_path / "ext"
    (ext / "parallel").mkdir(parents=True)
    r = _run(AARCH64, str(ext))
    assert r.returncode == 0, r.stderr + r.stdout
    assert "cpu_moe=absent" in r.stdout
    assert not (ext / "cpu" / "moe_mul1.cpp").exists()


def test_aarch64_refuses_unknown_cpu_moe(tmp_path):
    ext = tmp_path / "ext"
    (ext / "cpu").mkdir(parents=True)
    (ext / "parallel").mkdir(parents=True)
    (ext / "cpu" / "moe_mul1.cpp").write_text("// neither x86 nor stub\nvoid foo() {}\n")
    r = _run(AARCH64, str(ext))
    assert r.returncode != 0
    assert "refusing" in (r.stderr + r.stdout)


def test_aarch64_contract_accepts_v149_cpu_bindings(tmp_path):
    """Every symbol v1.4.9 registers must have a definition site in the tree."""
    ext = tmp_path / "ext"
    (ext / "cpu").mkdir(parents=True)
    (ext / "parallel").mkdir(parents=True)
    shutil.copyfile(FIXTURES / "moe_mul1.cpp", ext / "cpu" / "moe_mul1.cpp")
    (ext / "cpu" / "moe_handoff.cu").write_text(HANDOFF_SYMBOLS)
    (ext / "bindings.cpp").write_text(V149_CPU_BINDINGS)
    r = _run(AARCH64, str(ext))
    assert r.returncode == 0, r.stderr + r.stdout
    assert f"cpu_pybind_symbols={V149_CPU_SYMBOL_COUNT}" in r.stdout


def test_aarch64_contract_rejects_legacy_stub_without_v149_probe(tmp_path):
    """Negative control: a v1.4.7-era stub lacks has_avx512_bw and must be refused.

    Without this check the miss only surfaces as an undefined symbol inside a
    long aarch64 image build.
    """
    ext = tmp_path / "ext"
    (ext / "cpu").mkdir(parents=True)
    (ext / "parallel").mkdir(parents=True)
    shutil.copyfile(FIXTURES / "moe_mul1.cpp", ext / "cpu" / "moe_mul1.cpp")
    r = _run(AARCH64, str(ext))
    assert r.returncode == 0, r.stderr + r.stdout
    stub_path = ext / "cpu" / "moe_mul1.cpp"
    legacy = stub_path.read_text().replace(
        "bool exl3_moe_cpu_has_avx512_bw() { return false; }", ""
    )
    assert "exl3_moe_cpu_has_avx512_bw" not in legacy
    stub_path.write_text(legacy)
    (ext / "cpu" / "moe_handoff.cu").write_text(HANDOFF_SYMBOLS)
    (ext / "bindings.cpp").write_text(V149_CPU_BINDINGS)
    r2 = _run(AARCH64, str(ext))
    assert r2.returncode != 0
    assert "exl3_moe_cpu_has_avx512_bw" in (r2.stderr + r2.stdout)


def test_aarch64_contract_skips_synthetic_tree_without_bindings(tmp_path):
    """Fixture trees have no bindings.cpp; the guard is inapplicable, not failed."""
    ext = tmp_path / "ext"
    (ext / "cpu").mkdir(parents=True)
    (ext / "parallel").mkdir(parents=True)
    shutil.copyfile(FIXTURES / "moe_mul1.cpp", ext / "cpu" / "moe_mul1.cpp")
    r = _run(AARCH64, str(ext))
    assert r.returncode == 0, r.stderr + r.stdout
    assert "skipping the pybind CPU-symbol contract check" in r.stdout


def test_ticket_installer_skips_native_v147(tmp_path):
    ext = _native_tree(tmp_path)
    before = {n: (ext / "quant" / n).read_bytes() for n in FILES}
    r = _run(TICKET, str(ext))
    assert r.returncode == 0, r.stderr + r.stdout
    assert "native=1" in r.stdout
    assert "patched=0" in r.stdout
    after = {n: (ext / "quant" / n).read_bytes() for n in FILES}
    assert after == before


def test_ticket_installer_still_patches_c5d9_pristine(tmp_path):
    ext = _ext_with_quant(tmp_path, PRISTINE)
    r = _run(TICKET, str(ext))
    assert r.returncode == 0, r.stderr + r.stdout
    assert "patched=6, already=0" in r.stdout
    for name in FILES:
        assert (ext / "quant" / name).read_bytes() == (PATCHED / name).read_bytes()


def test_ticket_installer_refuses_mixed_native_and_pristine(tmp_path):
    ext = _native_tree(tmp_path)
    shutil.copyfile(PRISTINE / "exl3_moe.cu", ext / "quant" / "exl3_moe.cu")
    before = {n: (ext / "quant" / n).read_bytes() for n in FILES}
    r = _run(TICKET, str(ext))
    assert r.returncode != 0
    assert "mixed native/c5d9c657" in (r.stderr + r.stdout)
    after = {n: (ext / "quant" / n).read_bytes() for n in FILES}
    assert after == before


def test_fat_kernel_anchors_on_v147_bindings(tmp_path):
    ext = tmp_path / "ext"
    (ext / "quant").mkdir(parents=True)
    shutil.copyfile(FIXTURES / "bindings.cpp", ext / "bindings.cpp")
    source = KIT_ROOT / "overlay"
    r = _run(FAT, str(ext), str(source))
    assert r.returncode == 0, r.stderr + r.stdout
    text = (ext / "bindings.cpp").read_text()
    assert text.count('#include "quant/exl3_moe.cuh"') == 1
    assert text.count('#include "quant/exl3_fat_gemm.cuh"') == 1
    assert text.count('#include "quant/exl3_fat_moe.cuh"') == 1
    assert 'm.def("exl3_fat_gemm", &exl3_fat_gemm, "exl3_fat_gemm");' in text
    assert 'm.def("exl3_fat_gemm_scatter", &exl3_fat_gemm_scatter, "exl3_fat_gemm_scatter");' in text
    assert 'm.def("exl3_fat_moe_gather", &exl3_fat_moe_gather, "exl3_fat_moe_gather");' in text
    assert (ext / "quant" / "exl3_fat_gemm.cu").is_file()
    assert (ext / "quant" / "exl3_fat_gemm.cuh").is_file()
    assert (ext / "quant" / "exl3_fat_moe.cu").is_file()
    assert (ext / "quant" / "exl3_fat_moe.cuh").is_file()
    # Prefix anchors stay unique after the additive insert, so a second
    # apply is not the drift case. Re-applying would duplicate the fat
    # entries; the build runs the installer once on a fresh pin tarball.


def test_nullconfig_inferparams_injected_for_v147_linearexl3(tmp_path):
    ns = _load(NAMESPACE, "glm53_exl3_namespace")
    pkg = tmp_path / "exllamav3"
    (pkg / "model").mkdir(parents=True)
    (pkg / "modules").mkdir(parents=True)
    fake_modules: dict = {}
    config = ns.inject_config_stub(pkg, fake_modules)
    assert config.NullConfig is ns.NullConfig
    assert config.InferParams is ns.InferParams
    assert fake_modules["exllamav3.model.config"] is config

    g: dict = {
        "__name__": "linear_exl3_init",
        "Config": object,
        "NullConfig": ns.NullConfig,
    }
    exec((FIXTURES / "linear_exl3_init.py").read_text().split("from ...")[0], g)
    # Drive the same config=None contract LinearEXL3 uses in v1.4.7.
    cfg = ns.NullConfig()
    assert cfg.infer_params.no_reconstruct is False
    assert isinstance(cfg.infer_params, ns.InferParams)

    def construct(config=None):
        if config is None:
            from_mod = fake_modules["exllamav3.model.config"]
            config = from_mod.NullConfig()
        return config.infer_params.no_reconstruct

    assert construct(None) is False
    forced = ns.NullConfig()
    forced.infer_params.no_reconstruct = True
    assert construct(forced) is True


def test_old_config_only_stub_would_not_construct():
    class ConfigOnly:
        pass

    def construct(config=None):
        if config is None:
            try:
                from nowhere import NullConfig  # noqa: F401
            except ImportError as exc:
                raise AttributeError("config.infer_params") from exc
        return config.infer_params.no_reconstruct

    try:
        construct(None)
    except AttributeError as exc:
        assert "infer_params" in str(exc)
    else:
        raise AssertionError("old Config-only stub must not construct LinearEXL3")
