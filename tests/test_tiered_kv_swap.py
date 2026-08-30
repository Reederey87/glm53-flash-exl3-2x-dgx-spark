#!/usr/bin/env python3
"""Unit tests for Tiered NVMe Swapping & KV Pruning integration."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_DIR = REPO_ROOT / "overlay"
START_SCRIPT = REPO_ROOT / "start.sh"


class TestTieredKVSwap(unittest.TestCase):
    def test_overlay_script_exists_and_runs(self):
        patch_file = OVERLAY_DIR / "patch_kv_sparse_prune.py"
        self.assertTrue(patch_file.exists(), "patch_kv_sparse_prune.py should exist in overlay/")

        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["GLM53_ENABLE_KV_PRUNING"] = "1"
        env["GLM53_KV_COMPRESSION_RATIO"] = "0.30"
        res = subprocess.run(
            [sys.executable, str(patch_file)],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Overlay execution failed: {res.stderr}")
        self.assertIn("Applied dynamic KV cache pruning hooks", res.stdout)

    def test_start_script_contains_swap_and_prune_flags(self):
        self.assertTrue(START_SCRIPT.exists(), "start.sh must exist")
        content = START_SCRIPT.read_text()
        self.assertIn("SWAP_SPACE_GB", content)
        self.assertIn("--swap-space", content)
        self.assertIn("patch_kv_sparse_prune.py", content)
        self.assertIn("GLM53_ENABLE_KV_PRUNING", content)


if __name__ == "__main__":
    unittest.main()
