#!/usr/bin/env python3
"""Unit tests for overlay/patch_fine_grained_apc.py against synthetic source.

Covers: disabled = no-op exit 0; enabled + missing target = exit 1 (fail closed);
first application rewrites both anchors and parses; re-application is a no-op;
a marker-bearing partial file is refused; a drifted anchor is refused.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "overlay" / "patch_fine_grained_apc.py"

SYNTHETIC = '''import logging
logger = logging.getLogger(__name__)

class HybridKVCacheCoordinator:
    def __init__(self, hash_block_size):
        self.single_type_managers = []
        self.enable_partial_hash_hits = True
        if self.enable_partial_hash_hits:
            unsupported_partial_hit_managers = {
                type(manager).__name__
                for manager in self.single_type_managers
                if not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
            }
            if unsupported_partial_hit_managers:
                self.enable_partial_hash_hits = False
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self):
        pass
'''


def run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    e = os.environ.copy()
    e.update(env)
    return subprocess.run([sys.executable, str(OVERLAY)], env=e, capture_output=True, text=True)


def test_disabled_is_noop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "kv.py"
        target.write_text(SYNTHETIC)
        r = run({"GLM53_FINE_GRAINED_APC": "0", "GLM53_FGAPC_TARGET": str(target)})
        assert r.returncode == 0 and "skipping" in r.stdout
        assert target.read_text() == SYNTHETIC


def test_enabled_missing_target_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        r = run({"GLM53_FINE_GRAINED_APC": "1", "GLM53_FGAPC_TARGET": str(Path(tmp) / "absent.py")})
        assert r.returncode == 1 and "FATAL: target not found" in r.stderr


def test_apply_then_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "kv.py"
        target.write_text(SYNTHETIC)
        env = {"GLM53_FINE_GRAINED_APC": "1", "GLM53_FGAPC_TARGET": str(target)}
        r = run(env)
        assert r.returncode == 0, r.stderr
        text = target.read_text()
        assert 'and type(manager).__name__ != "KpoolTailManager"' in text
        assert '"[glm53-fgapc] partial_hash=%s hash_block=%s managers=%s"' in text
        assert text.count("[glm53-fgapc]") >= 2
        compile(text, "kv.py", "exec")
        assert not list(Path(tmp).glob("*.tmp"))
        r2 = run(env)
        assert r2.returncode == 0 and "already patched" in r2.stdout
        assert target.read_text() == text


def test_partial_marker_is_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "kv.py"
        target.write_text(SYNTHETIC.replace("import logging", "# [glm53-fgapc] stray marker\nimport logging"))
        r = run({"GLM53_FINE_GRAINED_APC": "1", "GLM53_FGAPC_TARGET": str(target)})
        assert r.returncode != 0 and "incomplete" in (r.stderr + r.stdout)


def test_drifted_anchor_is_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "kv.py"
        target.write_text(SYNTHETIC.replace("and manager.block_size != hash_block_size", "and manager.block_size > hash_block_size"))
        r = run({"GLM53_FINE_GRAINED_APC": "1", "GLM53_FGAPC_TARGET": str(target)})
        assert r.returncode != 0 and "veto block not found" in (r.stderr + r.stdout)
        assert "[glm53-fgapc]" not in target.read_text()


if __name__ == "__main__":
    for t in (test_disabled_is_noop, test_enabled_missing_target_fails_closed,
              test_apply_then_idempotent, test_partial_marker_is_refused, test_drifted_anchor_is_refused):
        t()
    print("fine-grained APC overlay tests OK")
