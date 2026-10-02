#!/usr/bin/env python3
"""The 50 ms recheck must not treat the 5000 ms assignment as already patched."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "overlay" / "patch_shm_recheck.py"


def _load():
    spec = importlib.util.spec_from_file_location("patch_shm_recheck", PATCH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rewrite_is_not_a_prefix_of_the_upstream_constant() -> None:
    mod = _load()
    src = (
        "# cap\n"
        "SHM_READER_RECHECK_INTERVAL_MS = 5000\n"
        "wait_ms = SHM_READER_RECHECK_INTERVAL_MS\n"
    )
    out = mod.rewrite(src)
    assert "SHM_READER_RECHECK_INTERVAL_MS = 50\n" in out
    assert "SHM_READER_RECHECK_INTERVAL_MS = 5000\n" not in out
    assert out.count("SHM_READER_RECHECK_INTERVAL_MS") == 2
    assert mod.rewrite(out) == out


def test_anchor_drift_fails_closed() -> None:
    mod = _load()
    for src in ("SHM_READER_RECHECK_INTERVAL_MS = 5000\n" * 2, "no constant here\n"):
        try:
            mod.rewrite(src)
        except ValueError:
            continue
        raise AssertionError(f"anchor drift was accepted: {src!r}")


def test_start_wires_the_patch_six_ways() -> None:
    start = (ROOT / "start.sh").read_text(encoding="utf-8")
    host = "SHM_RECHECK_PATCH_HOST"
    fname = "patch_shm_recheck.py"
    assert (ROOT / "overlay" / fname).is_file()
    assert f'{host}="${{{host}:-$SCRIPT_DIR/overlay/{fname}}}"' in start
    assert f'[ -f "${host}" ] || die "${host} missing"' in start
    assert f'scp -q -o BatchMode=yes "${host}" "${{WORKER_SSH}}:/tmp/{fname}"' in start
    assert f"-v '/tmp/{fname}:/opt/glm53/{fname}:ro'" in start
    assert f'-v "${host}:/opt/glm53/{fname}:ro"' in start
    assert start.count(f"python3 -S /opt/glm53/{fname}") == 2
    assert 'GLM53_SHM_RECHECK_50MS="${GLM53_SHM_RECHECK_50MS:-0}"' in start
    assert '-e "GLM53_SHM_RECHECK_50MS=$GLM53_SHM_RECHECK_50MS"' in start


def test_image_layer_bakes_the_same_constant() -> None:
    df = (ROOT / "Dockerfile.shm-recheck-layer").read_text(encoding="utf-8")
    assert "ARG BASE=glm53-selfbuild:e3-armc-guards" in df
    assert "COPY overlay/patch_shm_recheck.py /opt/glm53/patch_shm_recheck.py" in df
    assert "GLM53_SHM_RECHECK_50MS=1 python3 -S /opt/glm53/patch_shm_recheck.py" in df
    assert "ENV GLM53_SHM_RECHECK_50MS" not in df
    assert "grep -Fqx 'SHM_READER_RECHECK_INTERVAL_MS = 50'" in df
    assert "grep -Fqx 'SHM_READER_RECHECK_INTERVAL_MS = 5000'" in df
    assert "glm53.shm.recheck=50" in df
    assert "busy_loop_s =" not in df
    assert "0.002" not in df


if __name__ == "__main__":
    test_rewrite_is_not_a_prefix_of_the_upstream_constant()
    test_anchor_drift_fails_closed()
    test_start_wires_the_patch_six_ways()
    test_image_layer_bakes_the_same_constant()
    print("shm recheck guards OK")
