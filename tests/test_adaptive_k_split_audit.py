#!/usr/bin/env python3
"""CPU tests for the exact-image adaptive-k draft/verify split audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_adaptive_k_split.py"
OVERLAY = (ROOT / "overlay/patch_adaptive_k.py").read_text()
SPEC = importlib.util.spec_from_file_location("adaptive_k_split_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SCHEDULER = """
scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
num_spec_tokens_to_schedule = self.num_spec_tokens
"""
V2_RUNNER = """
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_draft_tokens_per_req = np.fromiter(
            (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
"""
DFLASH = """
        self.num_query_per_req = 1 + self.num_speculative_steps
        max_num_sampled_tokens = self.max_num_reqs * self.num_speculative_steps
        num_query_tokens = num_reqs * self.num_query_per_req
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
"""
DFLASH2 = """
        self.num_speculative_steps,
        num_steps=self.num_speculative_steps
"""
MODEL = "block_size=1 + speculative_config.num_speculative_tokens"
SPEC_CFG = 'Currently only supported for method="dspark".'
LLM_BASE = "self.num_speculative_tokens = num_speculative_tokens"


def run_audit(**overrides):
    values = {
        "scheduler": SCHEDULER,
        "v2_model_runner": V2_RUNNER,
        "dflash_speculator": DFLASH,
        "dflash2_speculator": DFLASH2,
        "dflash2_model": MODEL,
        "speculative_config": SPEC_CFG,
        "overlay": OVERLAY,
        "llm_base_proposer": LLM_BASE,
    }
    values.update(overrides)
    return MODULE.audit(**values)


def test_live_contract_proceeds() -> None:
    report = run_audit()
    assert report["decision"] == "proceed"
    assert report["supported_verification_only_arm"] is True
    assert report["native_draft_query_rows"] == 8
    assert report["target_verify_query_lens"] == [3, 5, 8]
    assert all(report["checks"].values())


def test_kit139_draft_hook_parks() -> None:
    poisoned = OVERLAY.replace(
        "num_spec_tokens_to_schedule = self.num_spec_tokens",
        "num_spec_tokens_to_schedule = _GLM53_ADAPTIVE_K.batch_k(\n"
        "                num_spec_tokens_to_schedule, reqs, self.requests)\n"
        "        leftover = self.num_spec_tokens",
        1,
    )
    report = run_audit(overlay=poisoned)
    assert report["decision"] == "park"
    assert report["supported_verification_only_arm"] is False
    assert not report["checks"]["overlay_omits_draft_batch_k_hook"]


def test_coupled_verify_length_parks() -> None:
    report = run_audit(v2_model_runner="num_draft_tokens = self.num_speculative_steps")
    assert report["decision"] == "park"
    assert not report["checks"]["target_verify_follows_len_spec_token_ids"]
