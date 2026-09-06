#!/usr/bin/env python3
"""Classify whether a DFlash2 verification-only length arm is supported."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _contains_all(source: str, anchors: tuple[str, ...]) -> bool:
    return all(anchor in source for anchor in anchors)


def audit(
    config: dict[str, Any],
    base_speculator: str,
    dflash2_speculator: str,
    dflash2_model: str,
    speculative_config: str,
    candidates: list[int],
) -> dict[str, Any]:
    architectures = config.get("architectures") or []
    draft_config = config.get("dflash_config") or {}
    block_size = draft_config.get("block_size")
    native_tokens = block_size - 1 if isinstance(block_size, int) else None

    checks = {
        "dflash2_checkpoint": "DFlash2DraftModel" in architectures,
        "native_block_size_present": isinstance(block_size, int) and block_size > 1,
        "query_rows_coupled_to_num_speculative_tokens": _contains_all(
            base_speculator,
            (
                "self.num_query_per_req = 1 + self.num_speculative_steps",
                "max_num_sampled_tokens = "
                "self.max_num_reqs * self.num_speculative_steps",
                "num_query_tokens = num_reqs * self.num_query_per_req",
            ),
        ),
        "selector_and_cache_coupled_to_num_speculative_tokens": _contains_all(
            dflash2_speculator,
            (
                "self.num_speculative_steps,",
                "num_steps=self.num_speculative_steps",
                "num_sample = num_reqs * self.num_speculative_steps",
            ),
        ),
        "grouped_conv_block_coupled_to_num_speculative_tokens": (
            "block_size=1 + speculative_config.num_speculative_tokens"
            in dflash2_model
        ),
        "adaptive_verification_restricted_to_dspark": (
            'if self.method != "dspark" and self.enable_adaptive_verification:'
            in speculative_config
            and 'raise ValueError("Adaptive verification only supported with DSpark")'
            in speculative_config
        ),
    }

    recognized_legacy_path = all(checks.values())
    if recognized_legacy_path:
        decision = "park"
        reason = (
            "The legacy DFlash2 path has no independent verification-length "
            "control. Changing num_speculative_tokens also changes the native "
            "query rows, grouped-convolution block, selector walk, and draft-logit "
            "cache. Adaptive verification is restricted to DSpark."
        )
    else:
        decision = "unknown"
        reason = (
            "The audited source or checkpoint contract drifted. Do not run a "
            "verification-length arm until the new path is reviewed."
        )

    return {
        "decision": decision,
        "supported_verification_only_arm": False,
        "requested_num_speculative_tokens": candidates,
        "native_block_size": block_size,
        "native_num_speculative_tokens": native_tokens,
        "checks": checks,
        "reason": reason,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-speculator", type=Path, required=True)
    parser.add_argument("--dflash2-speculator", type=Path, required=True)
    parser.add_argument("--dflash2-model", type=Path, required=True)
    parser.add_argument("--speculative-config", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=int,
        default=[],
        help="Proposed num_speculative_tokens value; repeat for multiple arms.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = args.candidate or [4, 3]
    if any(candidate < 1 for candidate in candidates):
        raise SystemExit("--candidate values must be positive")
    report = audit(
        json.loads(args.config.read_text()),
        args.base_speculator.read_text(),
        args.dflash2_speculator.read_text(),
        args.dflash2_model.read_text(),
        args.speculative_config.read_text(),
        candidates,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] == "park" else 2


if __name__ == "__main__":
    raise SystemExit(main())
