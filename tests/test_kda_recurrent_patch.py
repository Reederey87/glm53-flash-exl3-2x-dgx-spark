#!/usr/bin/env python3
"""Apply overlay/patch_kda_recurrent.py against the live FLA launch fixture.

Hard gates:
  * default-off leaves the stock warps=1 / stages=3 / BV-cap=8 block
  * armed knobs rewrite only that unique launch
  * helpers refuse illegal values
  * installer is idempotent
  * FlashInfer fused_kda_decode is not imported
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = HERE.parent / "overlay" / "patch_kda_recurrent.py"
FIXTURE = HERE / "fixtures" / "kda-recurrent-fla.py"
STOCK = (
    "    BK, BV = next_power_of_2(K), min(next_power_of_2(V), 8)\n"
    "    NK, NV = cdiv(K, BK), cdiv(V, BV)\n"
    "    assert NK == 1, \"NK > 1 is not supported yet\"\n"
    "    num_stages = 3\n"
    "    num_warps = 1\n"
)


def _helpers(text: str) -> dict:
    start = text.index("def _glm53_kda_rec_int")
    end = text.index("def fused_recurrent_kda_fwd")
    ns: dict = {"os": os}
    exec(compile("import os\n" + text[start:end], "<kda-helpers>", "exec"), ns)
    return ns


def _run(tmp: Path, env: dict[str, str] | None = None, glm_present: bool = False) -> subprocess.CompletedProcess[str]:
    fla = tmp / "kda.py"
    shutil.copy2(FIXTURE, fla)
    glm = tmp / "kernels.py"
    if glm_present:
        shutil.copy2(FIXTURE, glm)
    full_env = {**os.environ, "GLM53_FLA_KDA_PY": str(fla), "PYTHONPATH": ""}
    if glm_present:
        full_env["GLM53_GLM_KDA_KERNELS_PY"] = str(glm)
    else:
        full_env["GLM53_GLM_KDA_KERNELS_PY"] = str(tmp / "missing-kernels.py")
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-S", str(PATCH)],
        cwd=tmp,
        env=full_env,
        text=True,
        capture_output=True,
        check=False,
    ), fla, glm


def test_default_off_leaves_stock() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        result, fla, _ = _run(tmp)
        assert result.returncode == 0, result.stderr
        assert "knobs unset" in result.stdout
        text = fla.read_text()
        assert STOCK in text
        assert "[glm53-kda-recurrent]" not in text
        assert "fused_kda_decode" not in text


def test_armed_rewrites_launch_and_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        result, fla, _ = _run(
            tmp,
            env={"GLM53_KDA_REC_WARPS": "2", "GLM53_KDA_REC_STAGES": "3", "GLM53_KDA_REC_BV_CAP": "8"},
        )
        assert result.returncode == 0, result.stderr
        text = fla.read_text()
        assert STOCK not in text
        assert "num_warps = _glm53_kda_num_warps()" in text
        assert "min(next_power_of_2(V), _glm53_kda_bv_cap())" in text
        assert "fused_kda_decode" not in text
        helpers = _helpers(text)
        os.environ["GLM53_KDA_REC_WARPS"] = "2"
        os.environ["GLM53_KDA_REC_STAGES"] = "3"
        os.environ["GLM53_KDA_REC_BV_CAP"] = "8"
        try:
            assert helpers["_glm53_kda_num_warps"]() == 2
            assert helpers["_glm53_kda_num_stages"]() == 3
            assert helpers["_glm53_kda_bv_cap"]() == 8
        finally:
            os.environ.pop("GLM53_KDA_REC_WARPS", None)
            os.environ.pop("GLM53_KDA_REC_STAGES", None)
            os.environ.pop("GLM53_KDA_REC_BV_CAP", None)
        again, fla2, _ = _run(
            tmp,
            env={"GLM53_KDA_REC_WARPS": "2"},
        )
        # Second apply uses the already-patched file via GLM53_FLA_KDA_PY on a
        # fresh copy; re-copy then apply twice on the same file.
        patched = fla.read_text()
        fla.write_text(patched)
        result2 = subprocess.run(
            [sys.executable, "-S", str(PATCH)],
            cwd=tmp,
            env={
                **os.environ,
                "GLM53_FLA_KDA_PY": str(fla),
                "GLM53_GLM_KDA_KERNELS_PY": str(tmp / "missing-kernels.py"),
                "GLM53_KDA_REC_WARPS": "2",
            },
            text=True,
            capture_output=True,
            check=False,
        )
        assert result2.returncode == 0, result2.stderr
        assert "already" in result2.stdout
        assert fla.read_text() == patched
        del again, fla2


def test_helpers_refuse_illegal_values() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        result, fla, _ = _run(tmp, env={"GLM53_KDA_REC_WARPS": "3"})
        assert result.returncode == 0, result.stderr
        helpers = _helpers(fla.read_text())
        os.environ["GLM53_KDA_REC_WARPS"] = "3"
        try:
            raised = False
            try:
                helpers["_glm53_kda_num_warps"]()
            except SystemExit as exc:
                raised = True
                assert "GLM53_KDA_REC_WARPS" in str(exc)
            assert raised
        finally:
            os.environ.pop("GLM53_KDA_REC_WARPS", None)


def test_armed_missing_targets_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        env = {
            **os.environ,
            "GLM53_FLA_KDA_PY": str(tmp / "no-fla.py"),
            "GLM53_GLM_KDA_KERNELS_PY": str(tmp / "no-glm.py"),
            "GLM53_KDA_REC_WARPS": "2",
        }
        result = subprocess.run(
            [sys.executable, "-S", str(PATCH)],
            cwd=tmp,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 1, (result.returncode, result.stdout, result.stderr)
        assert "no fused_recurrent_kda_fwd" in (result.stderr + result.stdout)


def test_drifted_anchor_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        fla = tmp / "kda.py"
        text = FIXTURE.read_text().replace("num_warps = 1", "num_warps = 4", 1)
        fla.write_text(text)
        result = subprocess.run(
            [sys.executable, "-S", str(PATCH)],
            cwd=tmp,
            env={
                **os.environ,
                "GLM53_FLA_KDA_PY": str(fla),
                "GLM53_GLM_KDA_KERNELS_PY": str(tmp / "missing-kernels.py"),
                "GLM53_KDA_REC_WARPS": "2",
            },
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 1, (result.returncode, result.stdout, result.stderr)
        assert "launch anchor found 0 times" in (result.stderr + result.stdout)


if __name__ == "__main__":
    test_default_off_leaves_stock()
    test_armed_rewrites_launch_and_is_idempotent()
    test_helpers_refuse_illegal_values()
    test_armed_missing_targets_fail_closed()
    test_drifted_anchor_fails_closed()
    print("kda-recurrent overlay OK")
