#!/usr/bin/env python3
"""Prove the exact-image draft / verify split before any adaptive-k restart.

Task 25 parks if target verification query length cannot follow
``len(spec_token_ids)`` independently of the eight-row DFlash draft.
This audit is the hard gate. Exit 0 only when the split is proven and the
overlay does not write ``batch_k()`` into ``num_spec_tokens_to_schedule``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _contains_all(source: str, anchors: tuple[str, ...]) -> bool:
    return all(anchor in source for anchor in anchors)


def audit(
    *,
    scheduler: str,
    v2_model_runner: str,
    dflash_speculator: str,
    dflash2_speculator: str,
    dflash2_model: str,
    speculative_config: str,
    overlay: str,
    llm_base_proposer: str = "",
) -> dict[str, Any]:
    checks = {
        "target_verify_follows_len_spec_token_ids": _contains_all(
            v2_model_runner,
            (
                "draft_tokens = scheduler_output.scheduled_spec_decode_tokens",
                "num_draft_tokens_per_req = np.fromiter(",
                "(len(draft_tokens.get(req_id, ())) for req_id in req_ids)",
            ),
        ),
        "scheduler_copies_spec_token_ids": (
            "scheduled_spec_decode_tokens[request.request_id] = spec_token_ids"
            in scheduler
        ),
        "v2_dflash_propose_omits_runtime_k": _contains_all(
            dflash_speculator,
            (
                "def propose(\n        self,\n        input_batch: InputBatch,",
                "self.num_query_per_req = 1 + self.num_speculative_steps",
            ),
        ),
        "query_rows_coupled_to_num_speculative_steps": _contains_all(
            dflash_speculator,
            (
                "self.num_query_per_req = 1 + self.num_speculative_steps",
                "max_num_sampled_tokens = "
                "self.max_num_reqs * self.num_speculative_steps",
                "num_query_tokens = num_reqs * self.num_query_per_req",
            ),
        ),
        "selector_and_cache_native_steps": _contains_all(
            dflash2_speculator,
            (
                "self.num_speculative_steps,",
                "num_steps=self.num_speculative_steps",
            ),
        ),
        "grouped_conv_block_coupled": (
            "block_size=1 + speculative_config.num_speculative_tokens"
            in dflash2_model
        ),
        "adaptive_verification_restricted_to_dspark": (
            'Currently only supported for method="dspark"' in speculative_config
            or (
                'if self.method != "dspark" and self.enable_adaptive_verification:'
                in speculative_config
            )
        ),
        "legacy_proposer_would_rebind_k": (
            "self.num_speculative_tokens = num_speculative_tokens"
            in llm_base_proposer
            if llm_base_proposer
            else True
        ),
        "overlay_omits_draft_batch_k_hook": (
            "num_spec_tokens_to_schedule = _GLM53_ADAPTIVE_K.batch_k(" not in overlay
            and "_GLM53_ADAPTIVE_K.apply(" in overlay
            and "decode_query_lens = _glm53_adaptive_k_query_lens" in overlay
        ),
        "overlay_keeps_schedule_k_anchor": (
            "num_spec_tokens_to_schedule = self.num_spec_tokens" in overlay
            and "if self.dynamic_sd_lookup is not None" in overlay
        ),
    }
    split_ok = all(
        checks[name]
        for name in (
            "target_verify_follows_len_spec_token_ids",
            "scheduler_copies_spec_token_ids",
            "v2_dflash_propose_omits_runtime_k",
            "query_rows_coupled_to_num_speculative_steps",
            "selector_and_cache_native_steps",
            "grouped_conv_block_coupled",
            "overlay_omits_draft_batch_k_hook",
        )
    )
    if split_ok:
        decision = "proceed"
        reason = (
            "Target verify query length follows len(spec_token_ids). V2 DFlash "
            "propose() does not take a runtime k; query/mask/selector/cache stay "
            "native 8. Overlay trims only spec_token_ids and extra FULL graphs "
            "are target 3/5. Do not copy kit #139's batch_k schedule hook."
        )
    else:
        decision = "park"
        reason = (
            "Exact-image draft/verify split is missing or the overlay still "
            "touches the drafter count. Do not restart."
        )
    return {
        "decision": decision,
        "supported_verification_only_arm": split_ok,
        "native_draft_query_rows": 8,
        "target_verify_query_lens": [3, 5, 8],
        "checks": checks,
        "reason": reason,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scheduler", type=Path, required=True)
    p.add_argument("--v2-model-runner", type=Path, required=True)
    p.add_argument("--dflash-speculator", type=Path, required=True)
    p.add_argument("--dflash2-speculator", type=Path, required=True)
    p.add_argument("--dflash2-model", type=Path, required=True)
    p.add_argument("--speculative-config", type=Path, required=True)
    p.add_argument("--overlay", type=Path, required=True)
    p.add_argument("--llm-base-proposer", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = audit(
        scheduler=args.scheduler.read_text(),
        v2_model_runner=args.v2_model_runner.read_text(),
        dflash_speculator=args.dflash_speculator.read_text(),
        dflash2_speculator=args.dflash2_speculator.read_text(),
        dflash2_model=args.dflash2_model.read_text(),
        speculative_config=args.speculative_config.read_text(),
        overlay=args.overlay.read_text(),
        llm_base_proposer=(
            args.llm_base_proposer.read_text() if args.llm_base_proposer else ""
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] == "proceed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
