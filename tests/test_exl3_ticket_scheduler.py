"""Byte-exact three-state installer tests for the exllamav3 ticket-scheduler overlay.

The installer (overlay/patch_exl3_ticket_scheduler.py) applies upstream
exllamav3 d5e4361 onto the historical c5d9c657 ext source at image build
time from vendored pristine/patched byte sets. v1.4.7 already contains the
ticket scheduler, so a native tree is a skip, not drift. These tests
exercise the installer logic host-side with fixture extension roots — no
torch, no CUDA.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

KIT_ROOT = Path(__file__).resolve().parent.parent
PATCHER = KIT_ROOT / "overlay" / "patch_exl3_ticket_scheduler.py"
TICKET_DIR = KIT_ROOT / "overlay" / "exl3-ticket"
PRISTINE = TICKET_DIR / "pristine"
PATCHED = TICKET_DIR / "patched"
NATIVE_V147 = KIT_ROOT / "tests" / "fixtures" / "exl3-v147" / "quant"
FILES = (
    "exl3_devctx.cu",
    "exl3_devctx.cuh",
    "exl3_moe.cu",
    "exl3_moe.cuh",
    "exl3_moe_common.cuh",
    "exl3_moe_kernel.cuh",
)


def _run(ext_root: Path, env_extra: dict[str, str] | None = None):
    env = dict(os.environ)
    env.pop("GLM53_EXL3_TICKET_SCHEDULER", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(PATCHER), str(ext_root)],
        capture_output=True,
        text=True,
        env=env,
    )


def _make_ext(tmp_path: Path, source_dir: Path) -> Path:
    ext = tmp_path / "ext"
    (ext / "quant").mkdir(parents=True)
    for name in FILES:
        shutil.copyfile(source_dir / name, ext / "quant" / name)
    return ext


def _read_all(ext: Path) -> dict[str, bytes]:
    return {n: (ext / "quant" / n).read_bytes() for n in FILES}


def test_vendored_sets_exist_and_differ():
    for name in FILES:
        p = (PRISTINE / name).read_bytes()
        q = (PATCHED / name).read_bytes()
        assert p and q and p != q, name


def test_semantic_markers():
    kernel = (PATCHED / "exl3_moe_kernel.cuh").read_text()
    assert "atomicAdd(&sched[0], 1)" in kernel
    assert "sched = locks + MOE_SCHED_OFFSET" in kernel
    assert "expert_idx_assign++ != ticket" in kernel
    assert "expert_idx_assign++ % concurrency" not in kernel
    devctx_h = (PATCHED / "exl3_devctx.cuh").read_text()
    assert "MOE_SCHED_INTS" in devctx_h and "MOE_MAX_GROUPS" in devctx_h
    moe_h = (PATCHED / "exl3_moe.cuh").read_text()
    assert "num_active" in moe_h
    pristine_kernel = (PRISTINE / "exl3_moe_kernel.cuh").read_text()
    assert "MOE_SCHED_OFFSET" not in pristine_kernel


def test_pristine_root_gets_fully_patched(tmp_path):
    ext = _make_ext(tmp_path, PRISTINE)
    r = _run(ext)
    assert r.returncode == 0, r.stderr
    assert "patched=6, already=0" in r.stdout
    assert _read_all(ext) == _read_all_of(PATCHED)


def test_idempotent_rerun_is_noop(tmp_path):
    ext = _make_ext(tmp_path, PATCHED)
    before = _read_all(ext)
    r = _run(ext)
    assert r.returncode == 0, r.stderr
    assert "patched=0, already=6" in r.stdout
    assert _read_all(ext) == before


def test_drift_fails_closed_and_writes_nothing(tmp_path):
    ext = _make_ext(tmp_path, PRISTINE)
    drifted = ext / "quant" / "exl3_moe.cu"
    drifted.write_bytes(drifted.read_bytes() + b"\n// drift\n")
    before = _read_all(ext)
    r = _run(ext)
    assert r.returncode != 0
    assert "anchor drifted" in (r.stderr + r.stdout)
    assert _read_all(ext) == before, "installer must not write on drift"


def test_optout_env_leaves_pristine(tmp_path):
    ext = _make_ext(tmp_path, PRISTINE)
    before = _read_all(ext)
    r = _run(ext, {"GLM53_EXL3_TICKET_SCHEDULER": "0"})
    assert r.returncode == 0, r.stderr
    assert _read_all(ext) == before


def test_missing_quant_dir_fails(tmp_path):
    ext = tmp_path / "empty-ext"
    ext.mkdir()
    r = _run(ext)
    assert r.returncode != 0
    assert "invalid extension root" in (r.stderr + r.stdout)


def test_missing_bindings_is_not_our_concern(tmp_path):
    # The fat-kernel installer owns bindings.cpp; this patcher only replaces
    # quant files and must not require or touch bindings.
    ext = _make_ext(tmp_path, PRISTINE)
    assert not (ext / "bindings.cpp").exists()
    r = _run(ext)
    assert r.returncode == 0, r.stderr


def test_mixed_state_refused_before_any_write(tmp_path):
    # One file already patched, five pristine: the installer must refuse on
    # the pre-scan and leave every file untouched.
    ext = _make_ext(tmp_path, PRISTINE)
    shutil.copyfile(PATCHED / "exl3_devctx.cuh", ext / "quant" / "exl3_devctx.cuh")
    before = _read_all(ext)
    r = _run(ext)
    assert r.returncode != 0
    assert "mixed pristine/patched state" in (r.stderr + r.stdout)
    assert _read_all(ext) == before


def test_native_v147_tree_is_skipped_not_drift(tmp_path):
    ext = _make_ext(tmp_path, NATIVE_V147)
    before = _read_all(ext)
    r = _run(ext)
    assert r.returncode == 0, r.stderr
    assert "native=1" in r.stdout
    assert "patched=0" in r.stdout
    assert "already=0" in r.stdout
    assert _read_all(ext) == before
    assert (ext / "quant" / "exl3_moe.cuh").read_bytes() == (
        PATCHED / "exl3_moe.cuh"
    ).read_bytes()


def test_native_v147_hashes_match_vendored_pin():
    spec = importlib.util.spec_from_file_location("ticket_installer", PATCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.NATIVE_V147_SHA256) == set(FILES)
    for name in FILES:
        data = (NATIVE_V147 / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == module.NATIVE_V147_SHA256[name]
    shared = (NATIVE_V147 / "exl3_moe.cuh").read_bytes()
    assert shared == (PATCHED / "exl3_moe.cuh").read_bytes()
    assert shared != (PRISTINE / "exl3_moe.cuh").read_bytes()
    for name in FILES:
        if name == "exl3_moe.cuh":
            continue
        assert (NATIVE_V147 / name).read_bytes() != (PATCHED / name).read_bytes()
        assert (NATIVE_V147 / name).read_bytes() != (PRISTINE / name).read_bytes()


def test_dockerfile_wiring():
    """Static guard: the build must declare + forward the opt-out, copy the
    vendored sets, run the installer after the fat-kernel step, and assert
    num_active via the pybind-safe __doc__ check (inspect.signature raises
    ValueError on pybind11 builtins). v1.4.7 is the current pin; the
    installer remains for the historical c5d9c657 rollback pin."""
    dockerfile = (KIT_ROOT / "Dockerfile").read_text()
    assert "ARG GLM53_EXL3_TICKET_SCHEDULER=1" in dockerfile
    assert "ENV GLM53_EXL3_TICKET_SCHEDULER=${GLM53_EXL3_TICKET_SCHEDULER}" in dockerfile
    assert "COPY overlay/exl3-ticket/ /opt/glm53/exl3-ticket/" in dockerfile
    assert (
        "python3 /opt/glm53/patch_exl3_ticket_scheduler.py" in dockerfile
    ), "installer not wired into the build"
    fat_pos = dockerfile.index("patch_exl3_fat_kernel.py /tmp/exllamav3")
    ticket_pos = dockerfile.index("patch_exl3_ticket_scheduler.py /tmp/exllamav3")
    assert ticket_pos > fat_pos, "ticket-scheduler step must follow the fat-kernel step"
    assert "inspect.signature(exllamav3_ext.exl3_moe)" not in dockerfile
    assert "'num_active' in doc or 'arg29' in doc" in dockerfile
    assert 'if [ "${GLM53_EXL3_TICKET_SCHEDULER}" = "1" ]' in dockerfile
    assert "ARG EXLLAMAV3_COMMIT=5be886578ec80324c2c715269387be2058724b6e" in dockerfile
    assert "v1.4.9, like v1.4.7, already contains the d5e4361 ticket scheduler" in dockerfile


def _read_all_of(source_dir: Path) -> dict[str, bytes]:
    return {n: (source_dir / n).read_bytes() for n in FILES}
