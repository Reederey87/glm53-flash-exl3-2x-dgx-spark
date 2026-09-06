#!/usr/bin/env python3
"""CPU tests for the DFlash2 verification-length compatibility audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_dflash_verification_length.py"
SPEC = importlib.util.spec_from_file_location("dflash_length_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "dflash_config": {"block_size": 8, "selector_top_k": 16},
}
BASE_SPECULATOR = """
self.num_query_per_req = 1 + self.num_speculative_steps
max_num_sampled_tokens = self.max_num_reqs * self.num_speculative_steps
num_query_tokens = num_reqs * self.num_query_per_req
"""
DFLASH2_SPECULATOR = """
self.num_speculative_steps,
num_steps=self.num_speculative_steps
num_sample = num_reqs * self.num_speculative_steps
"""
DFLASH2_MODEL = "block_size=1 + speculative_config.num_speculative_tokens"
SPECULATIVE_CONFIG = """
if self.method != "dspark" and self.enable_adaptive_verification:
    raise ValueError("Adaptive verification only supported with DSpark")
"""


def run_audit(**overrides):
    values = {
        "config": CONFIG,
        "base_speculator": BASE_SPECULATOR,
        "dflash2_speculator": DFLASH2_SPECULATOR,
        "dflash2_model": DFLASH2_MODEL,
        "speculative_config": SPECULATIVE_CONFIG,
        "candidates": [4, 3],
    }
    values.update(overrides)
    return MODULE.audit(**values)


def test_legacy_dflash2_path_parks_verification_only_arm() -> None:
    report = run_audit()
    assert report["decision"] == "park"
    assert report["supported_verification_only_arm"] is False
    assert report["native_block_size"] == 8
    assert report["native_num_speculative_tokens"] == 7
    assert all(report["checks"].values())


def test_source_drift_fails_closed() -> None:
    report = run_audit(dflash2_model="block_size=8")
    assert report["decision"] == "unknown"
    assert report["supported_verification_only_arm"] is False
    assert not report["checks"][
        "grouped_conv_block_coupled_to_num_speculative_tokens"
    ]
