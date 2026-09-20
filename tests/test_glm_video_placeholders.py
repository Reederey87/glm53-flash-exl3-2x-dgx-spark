#!/usr/bin/env python3
"""Host-runnable installer tests for the video overlay / kpool preflight."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (
        HERE / "patch_glm_video_placeholders.py",
        ROOT / "overlay" / "patch_glm_video_placeholders.py",
    )
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
from patch_glm_video_placeholders import (  # noqa: E402
    KPOOL_NEW,
    KPOOL_OLD,
    PTH_BODY,
    prepare_kpool,
)


PINNED_KPOOL = f'''def maybe_persistent_topk(select_k):
    {KPOOL_OLD}
        return "persistent"
    return "decode"
'''


def _run_patch(sitepackages: Path, kpool: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_SITEPACKAGES"] = str(sitepackages)
    env["GLM53_KPOOL_PY"] = str(kpool)
    env["GLM53_VIDEO_PATCH_PY"] = str(sitepackages / "glm53_video_patch.py")
    env["GLM53_VIDEO_PTH"] = str(sitepackages / "glm53_video.pth")
    return subprocess.run(
        [sys.executable, str(PATCH)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_prepare_then_install() -> None:
    patched, action = prepare_kpool(PINNED_KPOOL)
    assert action == "patched"
    assert KPOOL_NEW in patched
    assert KPOOL_OLD not in patched
    again, again_action = prepare_kpool(patched)
    assert again_action == "already present"
    assert again == patched

    with tempfile.TemporaryDirectory() as raw:
        site = Path(raw) / "site-packages"
        kpool_dir = site / "vllm/model_executor/layers"
        kpool_dir.mkdir(parents=True)
        kpool = kpool_dir / "sparse_attn_indexer_kpool.py"
        kpool.write_text(PINNED_KPOOL)
        first = _run_patch(site, kpool)
        assert first.returncode == 0, first.stderr + first.stdout
        assert (site / "glm53_video_patch.py").is_file()
        assert (site / "glm53_video.pth").read_text() == PTH_BODY
        assert KPOOL_NEW in kpool.read_text()
        assert "overlay install ok aligned=True" in first.stderr
        second = _run_patch(site, kpool)
        assert second.returncode == 0, second.stderr + second.stdout
        assert "already disabled in kpool" in second.stderr


def test_kpool_drift_leaves_hook_and_pth_untouched() -> None:
    with tempfile.TemporaryDirectory() as raw:
        site = Path(raw) / "site-packages"
        kpool_dir = site / "vllm/model_executor/layers"
        kpool_dir.mkdir(parents=True)
        kpool = kpool_dir / "sparse_attn_indexer_kpool.py"
        drifted = PINNED_KPOOL.replace(
            "select_k in (512, 1024, 2048)",
            "select_k in (256, 512, 1024)",
            1,
        )
        kpool.write_text(drifted)
        hook = site / "glm53_video_patch.py"
        pth = site / "glm53_video.pth"
        result = _run_patch(site, kpool)
        assert result.returncode != 0
        assert "preflight failed" in result.stderr
        assert kpool.read_text() == drifted
        assert not hook.exists()
        assert not pth.exists()


def test_recipe_wiring() -> None:
    dockerfile = ROOT / "Dockerfile"
    if not dockerfile.is_file():
        return
    image = dockerfile.read_text()
    assert "COPY overlay/patch_glm_video_placeholders.py" in image
    assert "COPY tests/test_glm_video_placeholders.py" in image


if __name__ == "__main__":
    test_prepare_then_install()
    test_kpool_drift_leaves_hook_and_pth_untouched()
    test_recipe_wiring()
    print("glm53 video overlay OK")
