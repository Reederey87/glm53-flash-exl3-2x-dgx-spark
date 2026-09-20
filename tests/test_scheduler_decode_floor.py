#!/usr/bin/env python3
"""Host-runnable installer tests for the mixed-prefill decode-floor overlay."""
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
        HERE / "patch_scheduler_decode_floor.py",
        ROOT / "overlay" / "patch_scheduler_decode_floor.py",
    )
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
from patch_scheduler_decode_floor import (  # noqa: E402
    IMPORT_NEW,
    IMPORT_OLD,
    MARK,
    MARK_V31,
    RUNNING_NEW,
    RUNNING_OLD,
    WAITING_NEW,
    WAITING_OLD,
    validate_v31,
)

INSTALLED = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)

PINNED_FIXTURE = f'''{IMPORT_OLD}from collections import defaultdict, deque
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class Scheduler:
    def schedule(self):
        request = None
        token_budget = 0
        input_budget = 0
        draft_slots = 0
        num_new_tokens = 0
        if True:
{RUNNING_OLD}            num_new_tokens = min(num_new_tokens, 1)

    def _schedule_waiting(self, request, num_computed_tokens, num_new_tokens, request_queue, step_skipped_waiting):
        while True:
            if True:
                if True:
{WAITING_OLD}                    break
'''


def _run_patch(target: Path, extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_SCHEDULER_PY"] = str(target)
    env["GLM53_MIXED_PREFILL_CHUNK"] = "skip"
    if extra:
        env.update(extra)
    return subprocess.run(
        [sys.executable, str(PATCH)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_fixture() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "scheduler.py"
        dst.write_text(PINNED_FIXTURE)
        first = _run_patch(dst)
        assert first.returncode == 0, first.stderr + first.stdout
        text = dst.read_text()
        validate_v31(text)
        assert MARK in text
        assert MARK_V31 in text
        assert "def _glm53_mixed_prefill_policy(" in text
        assert IMPORT_NEW in text
        assert RUNNING_NEW in text
        assert WAITING_NEW in text
        second = _run_patch(dst)
        assert second.returncode == 0, second.stderr + second.stdout
        assert "already present" in second.stdout
        assert dst.read_text() == text


def test_fail_closed() -> None:
    drifted = PINNED_FIXTURE.replace(
        "long_prefill_token_threshold",
        "long_prefill_limit",
        1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "scheduler.py"
        dst.write_text(drifted)
        result = _run_patch(dst)
        assert result.returncode != 0
        assert dst.read_text() == drifted


def test_installed_copy_if_present() -> None:
    src = Path(os.environ.get("GLM53_SCHEDULER_PY_SRC", INSTALLED))
    if not src.is_file():
        return
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "scheduler.py"
        dst.write_text(src.read_text())
        result = _run_patch(dst)
        assert result.returncode == 0, result.stderr
        validate_v31(dst.read_text())


def test_recipe_wiring() -> None:
    dockerfile = ROOT / "Dockerfile"
    start = ROOT / "start.sh"
    if not dockerfile.is_file() or not start.is_file():
        return
    image = dockerfile.read_text()
    launcher = start.read_text()
    assert "COPY overlay/patch_scheduler_decode_floor.py" in image
    assert "RUN python3 /opt/glm53/patch_scheduler_decode_floor.py" in image
    assert "patch_scheduler_decode_floor.py" in launcher


def main() -> int:
    test_fixture()
    test_fail_closed()
    test_installed_copy_if_present()
    test_recipe_wiring()
    print("scheduler decode-floor patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
