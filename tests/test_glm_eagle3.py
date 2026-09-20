#!/usr/bin/env python3
"""Host-runnable installer tests for the glm5next EAGLE3 aux-hidden overlay."""
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
        HERE / "patch_glm_eagle3.py",
        ROOT / "overlay" / "patch_glm_eagle3.py",
    )
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
from patch_glm_eagle3 import (  # noqa: E402
    EDITS,
    leftover_old_counts,
    prepare,
    verified_state,
)

ACTIVE_OLD = EDITS[2][0]
ACTIVE_NEW = EDITS[2][1]

PINNED_FIXTURE = (
    "from torch import nn\n"
    f"{EDITS[0][0]}"
    "from vllm.model_executor.models.utils import AutoWeightsLoader\n"
    f"{EDITS[1][0]}"
    "    def __init__(self):\n"
    f"{ACTIVE_OLD}"
    "        self.norm = None\n"
    "    def forward(self, positions, hidden_states, residual, post, comb):\n"
    f"{EDITS[3][0]}"
    "        if False:\n"
    "            return residual\n"
    f"{EDITS[4][0]}"
    f"{EDITS[5][0]}"
    "    pass\n"
    f"{EDITS[6][0]}"
    "    pass\n"
)


def _run_patch(target: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_GLM5NEXT_MODEL_PY"] = str(target)
    return subprocess.run(
        [sys.executable, str(PATCH)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_overlapping_active_layers_anchor() -> None:
    leftovers = leftover_old_counts()
    assert leftovers[2] == 1, leftovers
    assert leftovers == [0, 0, 1, 0, 0, 0, 0]
    assert ACTIVE_OLD in ACTIVE_NEW
    assert ACTIVE_OLD != ACTIVE_NEW


def test_fresh_install_and_idempotency() -> None:
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "model.py"
        target.write_text(PINNED_FIXTURE)
        first = _run_patch(target)
        assert first.returncode == 0, first.stderr + first.stdout
        text = target.read_text()
        assert verified_state(text)
        assert "class Glm5NextModel(nn.Module, EagleModelMixin):" in text
        assert "aux_hidden_state_layers" in text
        assert text.count(ACTIVE_OLD) == 1
        assert text.count(ACTIVE_NEW) == 1
        second = _run_patch(target)
        assert second.returncode == 0, second.stderr + second.stdout
        assert "already present" in second.stdout
        assert target.read_text() == text
        again, action = prepare(text)
        assert action == "already present"
        assert again == text


def test_fail_closed_on_drift() -> None:
    drifted = PINNED_FIXTURE.replace(
        "for layer in self._active_layers:",
        "for layer in self.layers:",
        1,
    )
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "model.py"
        target.write_text(drifted)
        result = _run_patch(target)
        assert result.returncode != 0
        assert "preflight failed" in result.stderr
        assert target.read_text() == drifted


def test_recipe_wiring() -> None:
    dockerfile = ROOT / "Dockerfile"
    if not dockerfile.is_file():
        return
    image = dockerfile.read_text()
    assert "COPY overlay/patch_glm_eagle3.py" in image
    assert "RUN python3 /opt/glm53/patch_glm_eagle3.py" in image
    assert "COPY tests/test_glm_eagle3.py" in image


if __name__ == "__main__":
    test_overlapping_active_layers_anchor()
    test_fresh_install_and_idempotency()
    test_fail_closed_on_drift()
    test_recipe_wiring()
    print("glm5next EAGLE3 overlay OK")
