from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEMD = ROOT / "systemd"


def test_boot_unit_is_enabled_by_default_target() -> None:
    text = (SYSTEMD / "vllm-glm53fp4.service").read_text(encoding="utf-8")
    assert "WantedBy=default.target" in text
    assert "ExecStart=/home/nvidia/glm53-fp4-local/launch.sh" in text
    assert "PULL_IMAGE=0" in text
    assert "ExecStartPost=" in text
    assert "vllm-glm53fp4-watchdog" not in text


def test_boot_unit_waits_for_memory_and_worker() -> None:
    text = (SYSTEMD / "vllm-glm53fp4.service").read_text(encoding="utf-8")
    assert "MemAvailable" in text
    assert "104857600" in text
    assert "192.168.177.11" in text
    assert "ExecStopPost=" in text
    assert "ExecStartPre=-/home/nvidia/glm53-fp4-local/launch.sh down" in text


def test_watchdog_timer_is_on_boot_and_minute() -> None:
    text = (SYSTEMD / "vllm-glm53fp4-watchdog.timer").read_text(encoding="utf-8")
    assert "OnBootSec=15min" in text
    assert "OnUnitActiveSec=1min" in text
    assert "Unit=vllm-glm53fp4-watchdog.service" in text
    assert "WantedBy=timers.target" in text


def test_watchdog_service_runs_repo_script() -> None:
    text = (SYSTEMD / "vllm-glm53fp4-watchdog.service").read_text(encoding="utf-8")
    assert "ExecStart=/home/nvidia/glm53-fp4-local/scripts/watchdog.sh" in text
    assert "TimeoutStartSec=1200" in text


def test_watchdog_script_parses_and_is_executable() -> None:
    path = ROOT / "scripts" / "watchdog.sh"
    assert os.access(path, os.X_OK)
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_boot_watchdog_receipt_is_present() -> None:
    receipt = ROOT / "docs" / "receipts" / "2026-10-17-systemd-boot-watchdog.md"
    assert receipt.is_file()
    text = receipt.read_text(encoding="utf-8")
    assert "2,078,358" in text
    assert "vllm-dsv41-two-sparks.service" in text
    assert "No physical reboot" in text


def test_systemd_doc_names_the_previous_unit() -> None:
    text = (ROOT / "docs" / "09-systemd.md").read_text(encoding="utf-8")
    assert "systemctl --user disable --now vllm-dsv41-two-sparks.service" in text
    assert "Linger=yes" in text
    assert "3 in 6 hours" in text
