#!/usr/bin/env python3
"""Host-runnable installer tests for the hybrid APC overlay."""
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
        HERE / "patch_hybrid_prefix_hit.py",
        ROOT / "overlay" / "patch_hybrid_prefix_hit.py",
    )
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
from patch_hybrid_prefix_hit import (  # noqa: E402
    EAGLE_NEW,
    EAGLE_OLD,
    LOG_NEW,
    LOG_OLD,
    MARK,
    MIN_NEW,
    MIN_OLD,
    verified_state,
)

INSTALLED = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py"
)

# Exact vLLM 487ecf187 / glm53-flash image fragments the overlay matches.
PINNED_FIXTURE = f'''from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    SlidingWindowSpec,
)

logger = init_logger(__name__)


def _validate_prefix_cache_retention_interval(
    retention_interval,
    scheduler_block_size,
    kv_cache_config,
):
    if retention_interval is None:
        return


class HybridKVCacheCoordinator:
    def __init__(self, kv_cache_config, use_eagle=False):
        self.eagle_group_ids = set()
{EAGLE_OLD}        self.single_type_managers = ()
        self.attention_groups = []
{LOG_OLD}

    def find_longest_cache_hit(self, spec, group_ids, hit_blocks, drop_eagle_block, idx, eagle_verified, curr_hit_length, _new_hit_length, hit_blocks_by_group, hit_length_by_group, longest_hit_length):
        while True:
            if True:
{MIN_OLD}                break
        return curr_hit_length, longest_hit_length
'''


def _run_patch(target: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_KV_COORDINATOR_PY"] = str(target)
    return subprocess.run(
        [sys.executable, str(PATCH)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_fixture() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "kv_cache_coordinator.py"
        dst.write_text(PINNED_FIXTURE)
        first = _run_patch(dst)
        assert first.returncode == 0, first.stderr + first.stdout
        text = dst.read_text()
        assert verified_state(text)
        assert MARK in text
        assert text.count(MARK) >= 3
        assert "def _glm53_is_draft_swa_spec(" in text
        assert "swa_ids or set(" in text
        assert EAGLE_NEW in text
        assert MIN_NEW in text
        assert LOG_NEW in text
        second = _run_patch(dst)
        assert second.returncode == 0, second.stderr + second.stdout
        assert "already present" in second.stdout
        assert dst.read_text() == text


def test_fail_closed() -> None:
    drifted = PINNED_FIXTURE.replace(
        "if use_eagle and not self.eagle_group_ids:",
        "if use_eagle and self.eagle_group_ids is None:",
        1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "kv_cache_coordinator.py"
        dst.write_text(drifted)
        result = _run_patch(dst)
        assert result.returncode != 0
        assert dst.read_text() == drifted


def test_installed_copy_if_present() -> None:
    src = Path(os.environ.get("GLM53_KV_COORDINATOR_PY_SRC", INSTALLED))
    if not src.is_file():
        return
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "kv_cache_coordinator.py"
        dst.write_text(src.read_text())
        result = _run_patch(dst)
        assert result.returncode == 0, result.stderr
        assert verified_state(dst.read_text())


def test_recipe_wiring() -> None:
    dockerfile = ROOT / "Dockerfile"
    start = ROOT / "start.sh"
    if not dockerfile.is_file() or not start.is_file():
        return
    image = dockerfile.read_text()
    launcher = start.read_text()
    assert "COPY overlay/patch_hybrid_prefix_hit.py" in image
    assert "RUN python3 /opt/glm53/patch_hybrid_prefix_hit.py" in image
    assert "patch_hybrid_prefix_hit.py" in launcher


def main() -> int:
    test_fixture()
    test_fail_closed()
    test_installed_copy_if_present()
    test_recipe_wiring()
    print("hybrid prefix-hit patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
