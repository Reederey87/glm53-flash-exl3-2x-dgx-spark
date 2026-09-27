"""Installer for the discarded-router-GEMM skip."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "overlay" / "patch_router_once.py"

OLD = """\
        # The router is always external (self.gate); main's MoERunner expects
        # pre-computed router_logits, so compute them here unconditionally.
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
"""

UPSTREAM = """\
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )
"""


def _run(tmp: Path, flag: str, text: str | None) -> subprocess.CompletedProcess[str]:
    model = tmp / "model.py"
    if text is not None:
        model.write_text("class M:\n    def forward(self):\n" + text)
    env = os.environ.copy()
    env["GLM53_ROUTER_ONCE"] = flag
    env["GLM53_GLM5NEXT_MODEL_PY"] = str(model)
    return subprocess.run(
        [sys.executable, str(INSTALLER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_flag_off_leaves_a_missing_file(tmp_path: Path) -> None:
    result = _run(tmp_path, "0", None)
    assert result.returncode == 0
    assert "unchanged" in result.stdout
    assert not (tmp_path / "model.py").exists()


def test_flag_on_patches_once_and_keeps_the_unowned_gate(tmp_path: Path) -> None:
    result = _run(tmp_path, "1", OLD)
    assert result.returncode == 0, result.stderr
    text = (tmp_path / "model.py").read_text()
    assert text.count("[glm53-router-once]") == 1
    assert "getattr(self.experts, \"gate\", None) is self.gate" in text
    assert "router_logits, _ = self.gate(hidden_states)" in text
    again = _run(tmp_path, "1", None)
    assert again.returncode == 0, again.stderr
    assert (tmp_path / "model.py").read_text() == text


def test_upstream_placeholder_is_left_alone(tmp_path: Path) -> None:
    result = _run(tmp_path, "1", UPSTREAM)
    assert result.returncode == 0, result.stderr
    assert "already owns" in result.stdout
    assert "[glm53-router-once]" not in (tmp_path / "model.py").read_text()


def test_drift_writes_nothing(tmp_path: Path) -> None:
    original = "class M:\n    def forward(self):\n        return hidden_states\n"
    (tmp_path / "model.py").write_text(original)
    result = _run(tmp_path, "1", None)
    assert result.returncode == 1
    assert "anchor drift" in result.stderr
    assert (tmp_path / "model.py").read_text() == original


def test_garbage_flag_is_rejected(tmp_path: Path) -> None:
    result = _run(tmp_path, "yes", OLD)
    assert result.returncode == 1
    assert "exactly 0 or 1" in result.stderr
    assert "self.gate(hidden_states)" in (tmp_path / "model.py").read_text()


@pytest.mark.parametrize("owned", [True, False])
def test_forward_preserves_routing_and_computes_gate_once(tmp_path, owned):
    """An internal gate replaces the placeholder; an external gate needs logits."""
    from types import SimpleNamespace

    class Gate:
        calls = 0

        def __call__(self, value):
            self.calls += 1
            return value * 3, None

    class Runner:
        def __init__(self, gate):
            self.gate = gate if owned else None

        def __call__(self, *, hidden_states, router_logits):
            if self.gate is not None:
                router_logits, _ = self.gate(hidden_states)
            return router_logits + hidden_states

    result = _run(tmp_path, "1", OLD)
    assert result.returncode == 0, result.stderr
    patched = (tmp_path / "model.py").read_text().split("    def forward(self):\n", 1)[1]
    outputs = []
    counts = []
    for body in (OLD, patched):
        scope = {}
        exec("def forward(self, hidden_states):\n" + body + "        return final_hidden_states\n", scope)
        gate = Gate()
        model = SimpleNamespace(gate=gate, experts=Runner(gate))
        outputs.append(scope["forward"](model, 7))
        counts.append(gate.calls)
    assert outputs == [28, 28]
    assert counts == ([2, 1] if owned else [1, 1])


@pytest.mark.parametrize("body", [OLD + OLD, OLD + "        # [glm53-router-once]\n"])
def test_partial_or_duplicate_anchor_fails_without_writing(tmp_path, body):
    result = _run(tmp_path, "1", body)
    assert result.returncode == 1
    assert (tmp_path / "model.py").read_text() == "class M:\n    def forward(self):\n" + body
