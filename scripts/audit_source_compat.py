#!/usr/bin/env python3
"""Fail-closed source/image/overlay compatibility matrix for GLM-5.3 EXL3.

This is the task 27 + 13 + 17 fixture lane. It classifies the live fork
against the current production overlays. It does not swap the stock image,
does not raise LPTT, and does not treat batch-size dynamic speculative
decoding as a configuration knob.

Decisions:
  adopt  — inherited KVCacheSpec.merge still uses ``assert`` and the
           raise overlay can apply; remaining rebase PRs are parked
  park   — the inherited merge is already the raise form, or source
           drifted in a way that still recognizes the live contracts
  refuse — overlay/V2 mismatch, V1 force, MTP capture-size, or
           unrecognized source
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_merge_overlay():
    path = Path(__file__).resolve().parents[1] / "overlay" / "patch_kv_merge_assert.py"
    spec = importlib.util.spec_from_file_location("glm53_kv_merge_assert", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to load merge overlay from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MERGE_OVERLAY = _load_merge_overlay()
ANY_NONCAUSAL = "non_causal_multi_token_decode=any("
FP8_656 = "return self.block_size * 656"
GDN_DRAFT_TAGS = "num_decode_draft_tokens_cpu"
KPOOL_CLASS = "class SparseAttnIndexerKpool"
FUSED_APPLY = "def apply_exl3_fused_moe("
FAT_INDEX_SELECT = "torch.index_select(xh, 0, token_idx, out=h)"
V2_RUNNER_PATH = "v1/worker/gpu/model_runner.py"
V1_RUNNER_PATH = "v1/worker/gpu_model_runner.py"
W28_RUNNER_MARK = "self.model_state.set_kv_cache_config(  # [glm53-w28-correctness]"
GLM_V2_ARCH = '"Glm5NextForCausalLM"'


def _contains(source: str, needle: str) -> bool:
    return needle in source


def classify_merge(kv_interface: str) -> dict[str, Any]:
    merge_state = MERGE_OVERLAY.inherited_merge_state(kv_interface)
    has_any = ANY_NONCAUSAL in kv_interface
    has_656 = FP8_656 in kv_interface
    return {
        "inherited_merge": merge_state,
        "overlay_applicable": merge_state in {"assert", "raise"},
        "anchor_count": kv_interface.count(MERGE_OVERLAY.ANCHOR),
        "non_causal_any_merge": has_any,
        "fp8_ds_mla_page_bytes": 656 if has_656 else None,
        "fp8_ds_mla_656": has_656,
    }


def classify_overlays(overlay_sources: dict[str, str]) -> dict[str, Any]:
    w28 = overlay_sources.get("patch_w28_correctness.py", "")
    targets_v2 = V2_RUNNER_PATH in w28
    targets_v1 = V1_RUNNER_PATH in w28
    return {
        "w28_targets_v2_gpu_model_runner": targets_v2,
        "w28_targets_v1_gpu_model_runner": targets_v1,
        "hybrid_prefix_hit_present": "glm53-hybrid-apc" in overlay_sources.get(
            "patch_hybrid_prefix_hit.py", ""
        ),
        "decode_floor_present": "glm53-decode-floor" in overlay_sources.get(
            "patch_scheduler_decode_floor.py", ""
        ),
        "indexer_workspace_present": "glm53-indexer-workspace"
        in overlay_sources.get("patch_indexer_workspace.py", ""),
        "kv_merge_overlay_present": "glm53-kv-merge-assert"
        in overlay_sources.get("patch_kv_merge_assert.py", ""),
    }


def classify_runner(config_vllm: str, gpu_runner: str, v1_runner: str) -> dict[str, Any]:
    return {
        "config_vllm_present": bool(config_vllm.strip()),
        "gpu_model_runner_present": bool(gpu_runner.strip()),
        "glm_listed_default_v2": GLM_V2_ARCH in config_vllm,
        "w28_mark_on_v2_runner": W28_RUNNER_MARK in gpu_runner,
        "w28_mark_on_v1_runner": W28_RUNNER_MARK in v1_runner,
        "v2_env_key_present": "VLLM_USE_V2_MODEL_RUNNER" in config_vllm
        or "VLLM_USE_V2_MODEL_RUNNER" in gpu_runner,
    }


def classify_spec_paths(gdn: str, dflash: str, kpool: str, exl3: str) -> dict[str, Any]:
    fused = FUSED_APPLY in exl3
    fat_gather = FAT_INDEX_SELECT in exl3
    dflash_present = bool(dflash.strip())
    return {
        "dflash_source_present": dflash_present,
        "gdn_classifies_speculative_rows_by_draft_tags": GDN_DRAFT_TAGS in gdn,
        "legacy_dflash_has_set_attn": "def set_attn" in dflash,
        "legacy_dflash_mentions_aot_schedule": "aot_schedule" in dflash,
        "kpool_indexer_present": KPOOL_CLASS in kpool,
        "exl3_fused_apply_present": fused,
        "exl3_fat_path_uses_index_select": fat_gather,
        "c4_decode_uses_fused_exl3_moe": fused,
    }


def classify_launcher(values: dict[str, Any]) -> dict[str, Any]:
    spec = str(values.get("spec_method", "dflash"))
    seqs = int(values.get("max_num_seqs", 4))
    lptt = int(values.get("long_prefill_token_threshold", 1792))
    v2 = values.get("vllm_use_v2_model_runner")
    kpool = int(values.get("index_kpool", 4))
    rows_per_rank = (lptt + 1) // 2  # TP=2 solo prefill
    return {
        "spec_method": spec,
        "max_num_seqs": seqs,
        "long_prefill_token_threshold": lptt,
        "vllm_use_v2_model_runner": v2,
        "solo_prefill_rows_per_rank": rows_per_rank,
        "dsa_row_shard_gate_1024": rows_per_rank >= 1024,
        "mtp_capture_size_guard": spec == "mtp" and seqs > 12,
        "v1_force": v2 in (False, 0, "0"),
        "index_kpool": kpool,
    }


def decide(checks: dict[str, Any]) -> tuple[str, str]:
    if checks["launcher"]["v1_force"]:
        return (
            "refuse",
            "VLLM_USE_V2_MODEL_RUNNER=0 would force the V1 runner. "
            "Production already uses V2 and W28 patches v1/worker/gpu/model_runner.py.",
        )
    if checks["launcher"]["mtp_capture_size_guard"]:
        return (
            "refuse",
            "SPEC_METHOD=mtp refuses to boot when MAX_NUM_SEQS > 12 "
            "(MTP capture-size guard).",
        )
    if not checks["overlays"]["w28_targets_v2_gpu_model_runner"]:
        return (
            "refuse",
            "W28 overlay no longer targets the live V2 GPU model runner.",
        )
    if checks["overlays"]["w28_targets_v1_gpu_model_runner"]:
        return (
            "refuse",
            "W28 overlay unexpectedly targets the V1 gpu_model_runner path.",
        )
    missing_sources = [
        name
        for name, present in (
            ("kv_cache_interface", bool(checks["merge"].get("inherited_merge"))),
            ("config_vllm", checks["runner"]["config_vllm_present"]),
            ("gpu_model_runner", checks["runner"]["gpu_model_runner_present"]),
            ("gdn_attn", checks["spec"]["gdn_classifies_speculative_rows_by_draft_tags"]),
            ("dflash", checks["spec"]["dflash_source_present"]),
            ("kpool", checks["spec"]["kpool_indexer_present"]),
            ("exl3", checks["spec"]["exl3_fused_apply_present"]),
        )
        if not present
    ]
    if missing_sources:
        return (
            "refuse",
            "Missing production sources: " + ", ".join(missing_sources) + ".",
        )
    missing_overlays = [
        name
        for name, present in (
            (
                "patch_w28_correctness.py",
                checks["overlays"]["w28_targets_v2_gpu_model_runner"],
            ),
            (
                "patch_hybrid_prefix_hit.py",
                checks["overlays"]["hybrid_prefix_hit_present"],
            ),
            (
                "patch_scheduler_decode_floor.py",
                checks["overlays"]["decode_floor_present"],
            ),
            (
                "patch_indexer_workspace.py",
                checks["overlays"]["indexer_workspace_present"],
            ),
            (
                "patch_kv_merge_assert.py",
                checks["overlays"]["kv_merge_overlay_present"],
            ),
        )
        if not present
    ]
    if missing_overlays:
        return (
            "refuse",
            "Missing production overlays: " + ", ".join(missing_overlays) + ".",
        )
    merge = checks["merge"]["inherited_merge"]
    if merge == "unknown" or not checks["merge"]["overlay_applicable"]:
        return (
            "refuse",
            "Inherited KVCacheSpec.merge drifted; the raise overlay cannot apply.",
        )
    required = (
        checks["merge"]["non_causal_any_merge"],
        checks["merge"]["fp8_ds_mla_656"],
        checks["spec"]["gdn_classifies_speculative_rows_by_draft_tags"],
        checks["spec"]["kpool_indexer_present"],
        checks["spec"]["exl3_fused_apply_present"],
        checks["runner"]["glm_listed_default_v2"],
        checks["overlays"]["kv_merge_overlay_present"],
        checks["overlays"]["hybrid_prefix_hit_present"],
        checks["overlays"]["decode_floor_present"],
        checks["overlays"]["indexer_workspace_present"],
        checks["spec"]["dflash_source_present"],
        checks["runner"]["gpu_model_runner_present"],
        checks["runner"]["config_vllm_present"],
    )
    if not all(required):
        return (
            "refuse",
            "A live production contract is missing from the audited sources.",
        )
    if merge == "assert":
        return (
            "adopt",
            "Port KVCacheSpec.merge from assert to raise AssertionError so "
            "grouping stays fail-closed under python -O. Live path is already "
            "the V2 runner; #54374/#55178/#54394/#55061 remain parked.",
        )
    return (
        "park",
        "Inherited merge is already the raise form. Remaining rebase PRs "
        "stay parked on this exact image.",
    )


def audit(
    sources: dict[str, str],
    overlay_sources: dict[str, str],
    launcher: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "merge": classify_merge(sources.get("kv_cache_interface", "")),
        "overlays": classify_overlays(overlay_sources),
        "runner": classify_runner(
            sources.get("config_vllm", ""),
            sources.get("gpu_model_runner", ""),
            sources.get("gpu_model_runner_v1", ""),
        ),
        "spec": classify_spec_paths(
            sources.get("gdn_attn", ""),
            sources.get("dflash", ""),
            sources.get("kpool", ""),
            sources.get("exl3", ""),
        ),
        "launcher": classify_launcher(launcher),
        "rebase": {
            "pr_54374_legacy_dflash_aot": False,
            "pr_55178_basemamba_padded_tail_is_not_gdn": True,
            "pr_55449_packed_page_already_656": True,
            "pr_54394_lptt_too_small_for_row_shard": not classify_launcher(launcher)[
                "dsa_row_shard_gate_1024"
            ],
            "pr_55061_c4_decode_is_fused_exl3_moe": classify_spec_paths(
                "", "", "", sources.get("exl3", "")
            )["c4_decode_uses_fused_exl3_moe"],
        },
    }
    decision, reason = decide(checks)
    parked = {
        "54374": (
            "Live proposer is overlay/dflash2_speculator.py plus "
            "v1/spec_decode/dflash.py; it has no DFlashSpeculator.set_attn "
            "AOT disable. Include #54374 only at a MRV2 rebase."
        ),
        "55178": (
            "BaseMamba padded-tail fix is not the GLM KDA path. Live GDN "
            "classifies speculative rows by draft-count tags."
        ),
        "55449": (
            "Live MLAAttentionSpec.real_page_size_bytes already returns "
            "656 B/token for fp8_ds_mla. Fixture only; no capacity gain."
        ),
        "54394": (
            "Live indexer is SparseAttnIndexerKpool. Solo LPTT=1792 at TP2 "
            "is 896 rows/rank, below the ≥1024 gate. Do not raise LPTT."
        ),
        "55061": (
            "C4 decode uses fused exl3_moe. index_select + per-expert fat "
            "launch is the overflow path, not the C4 decode hot path. "
            "No ncu probe is claimed here."
        ),
        "v2_runner_knob": (
            "Production already logs 'Using V2 Model Runner'. Forcing "
            "VLLM_USE_V2_MODEL_RUNNER=1 is a no-op; forcing 0 is refused. "
            "Adaptive verification stays parked."
        ),
    }
    return {
        "decision": decision,
        "supported_v2_force_arm": False,
        "supported_mtp_rollback_above_12_seqs": False,
        "supported_dsa_row_shard": False,
        "supported_verification_length_arm": False,
        "checks": checks,
        "parked": parked,
        "reason": reason,
    }


def load_text(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-cache-interface", type=Path)
    parser.add_argument("--config-vllm", type=Path)
    parser.add_argument("--gpu-model-runner", type=Path)
    parser.add_argument("--gpu-model-runner-v1", type=Path)
    parser.add_argument("--gdn-attn", type=Path)
    parser.add_argument("--dflash", type=Path)
    parser.add_argument("--kpool", type=Path)
    parser.add_argument("--exl3", type=Path)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--spec-method", default="dflash")
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=1792)
    parser.add_argument("--vllm-use-v2-model-runner", default="")
    parser.add_argument("--index-kpool", type=int, default=4)
    parser.add_argument("--vllm-site", type=Path)
    return parser.parse_args()


def sources_from_site(site: Path) -> dict[str, str]:
    def read(rel: str) -> str:
        path = site / rel
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    return {
        "kv_cache_interface": read("v1/kv_cache_interface.py"),
        "config_vllm": read("config/vllm.py"),
        "gpu_model_runner": read("v1/worker/gpu/model_runner.py"),
        "gpu_model_runner_v1": read("v1/worker/gpu_model_runner.py"),
        "gdn_attn": read("v1/attention/backends/gdn_attn.py"),
        "dflash": read("v1/spec_decode/dflash.py"),
        "kpool": read("model_executor/layers/sparse_attn_indexer_kpool.py"),
        "exl3": read("model_executor/layers/quantization/exl3.py"),
    }


def overlays_from_dir(overlay_dir: Path) -> dict[str, str]:
    names = (
        "patch_w28_correctness.py",
        "patch_hybrid_prefix_hit.py",
        "patch_scheduler_decode_floor.py",
        "patch_indexer_workspace.py",
        "patch_kv_merge_assert.py",
    )
    return {
        name: (overlay_dir / name).read_text(encoding="utf-8")
        if (overlay_dir / name).is_file()
        else ""
        for name in names
    }


def v2_env_value(raw: str) -> bool | None:
    text = raw.strip()
    if text == "":
        return None
    if text in ("1", "true", "True"):
        return True
    if text in ("0", "false", "False"):
        return False
    raise SystemExit(f"invalid VLLM_USE_V2_MODEL_RUNNER={raw!r}")


def main() -> int:
    args = parse_args()
    if args.vllm_site is not None:
        sources = sources_from_site(args.vllm_site)
    else:
        sources = {
            "kv_cache_interface": load_text(args.kv_cache_interface),
            "config_vllm": load_text(args.config_vllm),
            "gpu_model_runner": load_text(args.gpu_model_runner),
            "gpu_model_runner_v1": load_text(args.gpu_model_runner_v1),
            "gdn_attn": load_text(args.gdn_attn),
            "dflash": load_text(args.dflash),
            "kpool": load_text(args.kpool),
            "exl3": load_text(args.exl3),
        }
    overlay_dir = args.overlay_dir or (
        Path(__file__).resolve().parents[1] / "overlay"
    )
    report = audit(
        sources,
        overlays_from_dir(overlay_dir),
        {
            "spec_method": args.spec_method,
            "max_num_seqs": args.max_num_seqs,
            "long_prefill_token_threshold": args.long_prefill_token_threshold,
            "vllm_use_v2_model_runner": v2_env_value(args.vllm_use_v2_model_runner),
            "index_kpool": args.index_kpool,
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] in {"adopt", "park"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
