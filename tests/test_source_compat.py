#!/usr/bin/env python3
"""CPU tests for the task 27 + 13 + 17 source/overlay compatibility matrix."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_source_compat.py"
SPEC = importlib.util.spec_from_file_location("source_compat_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

FIXTURES = ROOT / "tests/fixtures/source-compat"
LIVE = ROOT / "tests/fixtures/live-image-vllm"
START = (ROOT / "start.sh").read_text(encoding="utf-8")
OVERLAY = ROOT / "overlay"
PATCH = "patch_kv_merge_assert.py"


def compact_sources(**overrides: str) -> dict[str, str]:
    values = {
        "kv_cache_interface": (FIXTURES / "kv_cache_interface.py").read_text(),
        "config_vllm": (FIXTURES / "config_vllm.py").read_text(),
        "gpu_model_runner": (FIXTURES / "gpu_model_runner.py").read_text(),
        "gpu_model_runner_v1": (FIXTURES / "gpu_model_runner_v1.py").read_text(),
        "gdn_attn": (FIXTURES / "gdn_attn.py").read_text(),
        "dflash": (FIXTURES / "dflash.py").read_text(),
        "kpool": (FIXTURES / "kpool.py").read_text(),
        "exl3": (FIXTURES / "exl3.py").read_text(),
    }
    values.update(overrides)
    return values


def compact_overlays(**overrides: str) -> dict[str, str]:
    names = (
        "patch_w28_correctness.py",
        "patch_hybrid_prefix_hit.py",
        "patch_scheduler_decode_floor.py",
        "patch_indexer_workspace.py",
        "patch_kv_merge_assert.py",
    )
    values = {name: (OVERLAY / name).read_text() for name in names}
    values.update(overrides)
    return values


def run_audit(launcher: dict | None = None, **overrides):
    return MODULE.audit(
        compact_sources(**overrides.get("sources", {})),
        compact_overlays(**overrides.get("overlays", {})),
        launcher
        or {
            "spec_method": "dflash",
            "max_num_seqs": 4,
            "long_prefill_token_threshold": 1792,
            "vllm_use_v2_model_runner": None,
            "index_kpool": 4,
        },
    )


def test_compact_live_contracts_adopt_merge_overlay() -> None:
    report = run_audit()
    assert report["decision"] == "adopt"
    assert report["supported_v2_force_arm"] is False
    assert report["supported_mtp_rollback_above_12_seqs"] is False
    assert report["supported_dsa_row_shard"] is False
    assert report["supported_verification_length_arm"] is False
    assert report["checks"]["merge"]["inherited_merge"] == "assert"
    assert report["checks"]["merge"]["non_causal_any_merge"] is True
    assert report["checks"]["merge"]["fp8_ds_mla_page_bytes"] == 656
    assert report["checks"]["runner"]["glm_listed_default_v2"] is True
    assert report["checks"]["overlays"]["w28_targets_v2_gpu_model_runner"] is True
    assert report["checks"]["overlays"]["w28_targets_v1_gpu_model_runner"] is False
    assert report["checks"]["spec"]["c4_decode_uses_fused_exl3_moe"] is True
    assert report["checks"]["spec"]["legacy_dflash_has_set_attn"] is False
    assert report["checks"]["launcher"]["solo_prefill_rows_per_rank"] == 896
    assert report["checks"]["rebase"]["pr_54394_lptt_too_small_for_row_shard"] is True
    for pr in ("54374", "55178", "55449", "54394", "55061", "v2_runner_knob"):
        assert pr in report["parked"]


def test_already_raised_merge_parks() -> None:
    report = run_audit(
        sources={
            "kv_cache_interface": (FIXTURES / "kv_cache_interface.py")
            .read_text()
            .replace(
                "        assert all(spec == specs[0] for spec in specs[1:]), (\n"
                '            "All layers in the same KV cache group must be the same."\n'
                "        )\n",
                "        if not all(spec == specs[0] for spec in specs[1:]):\n"
                "            raise AssertionError(  # [glm53-kv-merge-assert]\n"
                '                "All layers in the same KV cache group must be the same."\n'
                "            )\n",
            )
        }
    )
    assert report["decision"] == "park"
    assert report["checks"]["merge"]["inherited_merge"] == "raise"


def test_v1_force_is_refused() -> None:
    report = run_audit(
        {
            "spec_method": "dflash",
            "max_num_seqs": 4,
            "long_prefill_token_threshold": 1792,
            "vllm_use_v2_model_runner": False,
            "index_kpool": 4,
        }
    )
    assert report["decision"] == "refuse"
    assert "VLLM_USE_V2_MODEL_RUNNER=0" in report["reason"]


def test_mtp_above_12_seqs_is_refused() -> None:
    report = run_audit(
        {
            "spec_method": "mtp",
            "max_num_seqs": 13,
            "long_prefill_token_threshold": 1792,
            "vllm_use_v2_model_runner": None,
            "index_kpool": 4,
        }
    )
    assert report["decision"] == "refuse"
    assert "MAX_NUM_SEQS > 12" in report["reason"]


def test_source_drift_fails_closed() -> None:
    report = run_audit(sources={"kv_cache_interface": "class KVCacheSpec:\n    pass\n"})
    assert report["decision"] == "refuse"
    assert "cannot apply" in report["reason"] or "drifted" in report["reason"]
    assert report["checks"]["merge"]["inherited_merge"] == "unknown"
    assert report["checks"]["merge"]["overlay_applicable"] is False


def test_missing_production_overlays_refuse() -> None:
    report = run_audit(
        overlays={
            "patch_hybrid_prefix_hit.py": "",
            "patch_scheduler_decode_floor.py": "",
            "patch_indexer_workspace.py": "",
        }
    )
    assert report["decision"] == "refuse"
    assert "Missing production overlays" in report["reason"]
    for name in (
        "patch_hybrid_prefix_hit.py",
        "patch_scheduler_decode_floor.py",
        "patch_indexer_workspace.py",
    ):
        assert name in report["reason"]


def test_missing_dflash_and_runner_sources_refuse() -> None:
    report = run_audit(
        sources={
            "dflash": "",
            "gpu_model_runner": "",
            "config_vllm": "",
        }
    )
    assert report["decision"] == "refuse"
    assert "Missing production sources" in report["reason"]
    for name in ("dflash", "gpu_model_runner", "config_vllm"):
        assert name in report["reason"]


def test_return_block_drift_refuses_and_overlay_cannot_apply() -> None:
    drifted = (
        (FIXTURES / "kv_cache_interface.py")
        .read_text()
        .replace("return copy.deepcopy(specs[0])", "return specs[0]", 1)
    )
    report = run_audit(sources={"kv_cache_interface": drifted})
    assert report["decision"] == "refuse"
    assert report["checks"]["merge"]["inherited_merge"] == "unknown"
    assert report["checks"]["merge"]["overlay_applicable"] is False
    assert report["checks"]["merge"]["anchor_count"] == 0


def test_duplicate_merge_anchors_refuse() -> None:
    source = (FIXTURES / "kv_cache_interface.py").read_text()
    doubled = source.replace(
        MODULE.MERGE_OVERLAY.ANCHOR,
        MODULE.MERGE_OVERLAY.ANCHOR + MODULE.MERGE_OVERLAY.ANCHOR,
        1,
    )
    report = run_audit(sources={"kv_cache_interface": doubled})
    assert report["decision"] == "refuse"
    assert report["checks"]["merge"]["inherited_merge"] == "unknown"
    assert report["checks"]["merge"]["anchor_count"] == 2


def test_missing_input_cli_exits_2(tmp_path: Path) -> None:
    empty = tmp_path / "empty.py"
    empty.write_text("")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--kv-cache-interface",
            str(empty),
            "--overlay-dir",
            str(OVERLAY),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert '"decision": "refuse"' in result.stdout


def test_overlay_wired_six_ways() -> None:
    host_var = "KV_MERGE_ASSERT_PATCH_HOST"
    fname = PATCH
    assert (OVERLAY / fname).is_file()
    assert f'{host_var}="${{{host_var}:-$SCRIPT_DIR/overlay/{fname}}}"' in START
    assert f'[ -f "${host_var}" ] || die "${host_var} missing"' in START
    assert (
        f'scp -q -o BatchMode=yes "${host_var}" "${{WORKER_SSH}}:/tmp/{fname}"'
        in START
    )
    assert f"-v '/tmp/{fname}:/opt/glm53/{fname}:ro'" in START
    assert f'-v "${host_var}:/opt/glm53/{fname}:ro"' in START
    assert START.count(f"python3 -S /opt/glm53/{fname}") == 2


def test_live_dump_adopts_when_present() -> None:
    if not (LIVE / "v1/kv_cache_interface.py").is_file():
        return
    report = MODULE.audit(
        MODULE.sources_from_site(LIVE),
        compact_overlays(),
        {
            "spec_method": "dflash",
            "max_num_seqs": 4,
            "long_prefill_token_threshold": 1792,
            "vllm_use_v2_model_runner": None,
            "index_kpool": 4,
        },
    )
    assert report["decision"] == "adopt"
    assert report["checks"]["merge"]["inherited_merge"] == "assert"
    assert report["checks"]["merge"]["fp8_ds_mla_656"] is True
    assert report["checks"]["spec"]["legacy_dflash_has_set_attn"] is False
