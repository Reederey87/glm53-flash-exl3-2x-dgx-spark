#!/usr/bin/env python3
"""Static wiring guards for verification-only adaptive-k (task 25 / kit #139 split).

The overlay must be wired six ways (host var, preflight, worker scp, both
container mounts, both patch-exec blocks) and its env must reach both ranks.
Extra FULL graphs are an independent B0 variable; stock capture stays
`1 2 4 8 16 24 32`.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = (ROOT / "start.sh").read_text(encoding="utf-8")
PROD_START = (ROOT / "local" / "prod-start.sh").read_text(encoding="utf-8")
OVERLAY = (ROOT / "overlay" / "patch_adaptive_k.py").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / "env.example").read_text(encoding="utf-8")
DOCS_02 = (ROOT / "docs" / "02-parameters.md").read_text(encoding="utf-8")


def _count(s: str) -> int:
    return START.count(s)


def test_overlay_wired_six_ways() -> None:
    host_var = "ADAPTIVE_K_PATCH_HOST"
    fname = "patch_adaptive_k.py"
    assert (ROOT / "overlay" / fname).is_file()
    assert f'{host_var}="${{{host_var}:-$SCRIPT_DIR/overlay/{fname}}}"' in START
    assert f'[ -f "${host_var}" ] || die "${host_var} missing"' in START
    assert f'scp -q -o BatchMode=yes "${host_var}" "${{WORKER_SSH}}:/tmp/{fname}"' in START
    assert f"-v '/tmp/{fname}:/opt/glm53/{fname}:ro'" in START
    assert f'-v "${host_var}:/opt/glm53/{fname}:ro"' in START
    assert _count(f"python3 -S /opt/glm53/{fname}") == 2
    assert START.index("python3 -S /opt/glm53/patch_align_floor.py") < START.index(
        "python3 -S /opt/glm53/patch_adaptive_k.py"
    )
    assert START.index("python3 -S /opt/glm53/patch_adaptive_k.py") < START.index(
        "python3 -S /opt/glm53/patch_kv_capacity_log.py"
    )


def test_env_defaults_off_and_reach_both_ranks() -> None:
    assert 'GLM53_ADAPTIVE_K="${GLM53_ADAPTIVE_K:-off}"' in START
    assert 'GLM53_ADAPTIVE_K_CAPTURE="${GLM53_ADAPTIVE_K_CAPTURE:-0}"' in START
    assert 'GLM53_ADAPTIVE_K_SET="${GLM53_ADAPTIVE_K_SET:-2,4,7}"' in START
    for knob in (
        "GLM53_ADAPTIVE_K",
        "GLM53_ADAPTIVE_K_CAPTURE",
        "GLM53_ADAPTIVE_K_SET",
        "GLM53_ADAPTIVE_K_ALPHA",
        "GLM53_ADAPTIVE_K_MARGIN",
        "GLM53_ADAPTIVE_K_MIN_STEPS",
        "GLM53_ADAPTIVE_K_SATURATE",
        "GLM53_ADAPTIVE_K_HIST",
    ):
        assert f'-e "{knob}=${knob}"' in START
    assert '"${nccl_common[@]}"' in START and 'for e in "${nccl_common[@]}"' in START


def test_stock_capture_is_default_and_extra_is_gated() -> None:
    assert "--cudagraph-capture-sizes 1 2 4 8 16 24 32" in START
    assert (
        "--cudagraph-capture-sizes 1 2 3 4 5 6 8 9 10 12 15 16 20 24 32" in START
    )
    assert START.index('GLM53_ADAPTIVE_K="${GLM53_ADAPTIVE_K:-off}"') < START.index(
        "--cudagraph-capture-sizes 1 2 3 4 5 6 8 9 10 12 15 16 20 24 32"
    )
    extra_block_start = START.index("# LOCAL: task 25 B0/B extra FULL graphs")
    extra_block = START[extra_block_start : extra_block_start + 900]
    assert "case \"${GLM53_ADAPTIVE_K}\" in ema|on|1)" in extra_block
    assert "case \"${GLM53_ADAPTIVE_K_CAPTURE}\" in 1|on|true|yes)" in extra_block


def test_overlay_refuses_kit139_draft_hook() -> None:
    assert "num_spec_tokens_to_schedule = _GLM53_ADAPTIVE_K.batch_k(" not in OVERLAY
    assert "_GLM53_ADAPTIVE_K.apply(" in OVERLAY
    assert "if not self.boot_enabled:" in OVERLAY


def test_prod_start_hashes_extra_args_not_policy_knobs() -> None:
    hash_line = [ln for ln in PROD_START.splitlines() if "shape_hash=" in ln][0]
    assert "EXTRA_ARGS" in hash_line
    assert "DFLASH_TOKENS" in hash_line
    assert "GLM53_ADAPTIVE_K_CAPTURE" in hash_line
    # Policy-only EMA must not itself force a JIT wipe.
    assert "GLM53_ADAPTIVE_K=" not in hash_line or "GLM53_ADAPTIVE_K_CAPTURE" in hash_line
    assert "printf 'GLM53_ADAPTIVE_K_CAPTURE=%s" in PROD_START


def test_docs_and_env_example_name_the_knobs() -> None:
    assert "GLM53_ADAPTIVE_K=ema" in ENV_EXAMPLE
    assert "GLM53_ADAPTIVE_K_CAPTURE=1" in ENV_EXAMPLE
    assert "verification-only" in ENV_EXAMPLE
    assert "Rollback: GLM53_ADAPTIVE_K=off" in ENV_EXAMPLE
    assert "GLM53_ADAPTIVE_K" in DOCS_02
    assert "B0" in DOCS_02


if __name__ == "__main__":
    test_overlay_wired_six_ways()
    test_env_defaults_off_and_reach_both_ranks()
    test_stock_capture_is_default_and_extra_is_gated()
    test_overlay_refuses_kit139_draft_hook()
    test_prod_start_hashes_extra_args_not_policy_knobs()
    test_docs_and_env_example_name_the_knobs()
    print("adaptive-k wiring guards OK")
