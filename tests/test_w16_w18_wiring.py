#!/usr/bin/env python3
"""Static wiring guards for the W16-W18 ports (kit PRs #51, #59, #69).

Each overlay must be wired six ways (host var, preflight, worker scp, both
container mounts, both patch-exec blocks) and its env must reach both ranks.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = (ROOT / "start.sh").read_text(encoding="utf-8")
PROD_START = (ROOT / "local" / "prod-start.sh").read_text(encoding="utf-8")


def _count(s: str) -> int:
    return START.count(s)


def test_overlays_wired_six_ways() -> None:
    for host_var, fname in (("SPINWAIT_PATCH_HOST", "patch_spinwait_gb10.py"),
                            ("FGAPC_PATCH_HOST", "patch_fine_grained_apc.py")):
        assert (ROOT / "overlay" / fname).is_file()
        assert f'{host_var}="${{{host_var}:-$SCRIPT_DIR/overlay/{fname}}}"' in START
        assert f'[ -f "${host_var}" ] || die "${host_var} missing"' in START
        assert f'scp -q -o BatchMode=yes "${host_var}" "${{WORKER_SSH}}:/tmp/{fname}"' in START
        assert f"-v '/tmp/{fname}:/opt/glm53/{fname}:ro'" in START  # worker
        assert f'-v "${host_var}:/opt/glm53/{fname}:ro"' in START  # head
        assert _count(f"python3 -S /opt/glm53/{fname}") == 2  # both ranks


def test_env_defaults_off_and_reach_both_ranks() -> None:
    assert 'GLM53_SPINWAIT_2MS="${GLM53_SPINWAIT_2MS:-0}"' in START
    assert 'GLM53_FINE_GRAINED_APC="${GLM53_FINE_GRAINED_APC:-0}"' in START
    # shared nccl_common array is expanded into BOTH docker runs
    assert '-e "GLM53_SPINWAIT_2MS=$GLM53_SPINWAIT_2MS"' in START
    assert '-e "GLM53_FINE_GRAINED_APC=$GLM53_FINE_GRAINED_APC"' in START
    assert '"${nccl_common[@]}"' in START and 'for e in "${nccl_common[@]}"' in START


def test_default_max_new_tokens_is_hash_neutral() -> None:
    assert 'DEFAULT_MAX_NEW_TOKENS="${DEFAULT_MAX_NEW_TOKENS:-}"' in START  # empty = unchanged
    line = ('[ -n "${DEFAULT_MAX_NEW_TOKENS:-}" ] && ARGS+=(--override-generation-config '
            '"{\\"max_new_tokens\\": ${DEFAULT_MAX_NEW_TOKENS}}")')
    assert _count(line) == 2  # head + worker
    hash_line = re.search(r'^shape_hash=.*$', PROD_START, re.M).group(0)
    assert "DEFAULT_MAX_NEW_TOKENS" not in hash_line
    assert "LONG_PREFILL_TOKEN_THRESHOLD" not in hash_line


def test_prod_start_hash_and_wipe_image() -> None:
    hash_line = re.search(r'^shape_hash=.*$', PROD_START, re.M).group(0)
    for k in ("DFLASH_MODEL", "DFLASH_REVISION", "DFLASH_TOKENS", "DFLASH_DRAFT_TP", "IMAGE"):
        assert k in hash_line, k
    # wipe container resolved from IMAGE= verbatim (self-built images have no ghcr digest)
    assert "sed -n 's/^IMAGE=//p' .env" in PROD_START
    # a missed wipe must not advance the stamp
    assert "stamp left unchanged" in PROD_START


if __name__ == "__main__":
    test_overlays_wired_six_ways()
    test_env_defaults_off_and_reach_both_ranks()
    test_default_max_new_tokens_is_hash_neutral()
    test_prod_start_hash_and_wipe_image()
    print("W16-W18 wiring guards OK")
