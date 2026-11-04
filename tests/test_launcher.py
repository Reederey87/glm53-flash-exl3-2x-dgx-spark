from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHELL_SCRIPTS = ["launch.sh", "stop.sh", "status.sh", "download.sh"]


def test_shell_scripts_parse() -> None:
    for name in SHELL_SCRIPTS:
        result = subprocess.run(
            ["bash", "-n", str(ROOT / name)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"


def test_shell_scripts_are_executable() -> None:
    for name in SHELL_SCRIPTS:
        assert os.access(ROOT / name, os.X_OK), name


def test_launcher_help_lists_operations() -> None:
    result = subprocess.run(
        [str(ROOT / "launch.sh"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for word in ("setup", "down", "status", "logs", "smoke", "kv", "preflight", "render"):
        assert word in result.stdout, word


def test_launcher_does_not_build_an_image() -> None:
    text = (ROOT / "launch.sh").read_text(encoding="utf-8")
    assert "docker build" not in text
    assert "Dockerfile" not in text


def test_launcher_pins_the_upstream_repository() -> None:
    text = (ROOT / "launch.sh").read_text(encoding="utf-8")
    assert "https://github.com/eugr/spark-vllm-docker.git" in text
    assert "SPARK_VLLM_REF" in text
    assert "run-recipe.sh" in text
