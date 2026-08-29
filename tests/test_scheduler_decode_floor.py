#!/usr/bin/env python3
"""Apply overlay/patch_scheduler_decode_floor.py to a copy of scheduler.py."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = next(
    p
    for p in (
        HERE / "patch_scheduler_decode_floor.py",
        HERE.parent / "overlay" / "patch_scheduler_decode_floor.py",
    )
    if p.is_file()
)
SRC = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)


def main() -> int:
    if not PATCH.is_file():
        raise SystemExit(f"missing {PATCH}")
    src = Path(os.environ.get("GLM53_SCHEDULER_PY_SRC", SRC))
    if not src.is_file():
        # Host unit test: copy from a live container if present.
        raise SystemExit(f"missing scheduler.py at {src}")
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "scheduler.py"
        shutil.copyfile(src, dst)
        env = os.environ.copy()
        env["GLM53_SCHEDULER_PY"] = str(dst)
        env["GLM53_MIXED_PREFILL_CHUNK"] = "skip"
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        text = dst.read_text()
        assert "[glm53-decode-floor]" in text
        assert text.count("[glm53-decode-floor]") == 2
        assert "def _glm53_mixed_prefill_policy(" in text
        # idempotent
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        assert dst.read_text().count("[glm53-decode-floor]") == 2
    print("scheduler decode-floor patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
