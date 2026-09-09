#!/usr/bin/env python3
"""Static wiring guards for the Task 30 fused_recurrent_kda overlay."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = (ROOT / "start.sh").read_text(encoding="utf-8")
PROD_START = (ROOT / "local" / "prod-start.sh").read_text(encoding="utf-8")
OVERLAY = (ROOT / "overlay" / "patch_kda_recurrent.py").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / "env.example").read_text(encoding="utf-8")
DOCS_02 = (ROOT / "docs" / "02-parameters.md").read_text(encoding="utf-8")


def _count(s: str) -> int:
    return START.count(s)


def test_overlay_wired_six_ways() -> None:
    host_var = "KDA_REC_PATCH_HOST"
    fname = "patch_kda_recurrent.py"
    assert (ROOT / "overlay" / fname).is_file()
    assert f'{host_var}="${{{host_var}:-$SCRIPT_DIR/overlay/{fname}}}"' in START
    assert f'[ -f "${host_var}" ] || die "${host_var} missing"' in START
    assert f'scp -q -o BatchMode=yes "${host_var}" "${{WORKER_SSH}}:/tmp/{fname}"' in START
    assert f"-v '/tmp/{fname}:/opt/glm53/{fname}:ro'" in START
    assert f'-v "${host_var}:/opt/glm53/{fname}:ro"' in START
    assert _count(f"python3 -S /opt/glm53/{fname}") == 2
    assert START.index("python3 -S /opt/glm53/patch_adaptive_k.py") < START.index(
        "python3 -S /opt/glm53/patch_kda_recurrent.py"
    )
    assert START.index("python3 -S /opt/glm53/patch_kda_recurrent.py") < START.index(
        "python3 -S /opt/glm53/patch_kv_capacity_log.py"
    )


def test_env_defaults_off_and_reach_both_ranks() -> None:
    assert 'GLM53_KDA_REC_WARPS="${GLM53_KDA_REC_WARPS:-}"' in START
    assert 'GLM53_KDA_REC_STAGES="${GLM53_KDA_REC_STAGES:-}"' in START
    assert 'GLM53_KDA_REC_BV_CAP="${GLM53_KDA_REC_BV_CAP:-}"' in START
    for knob in (
        "GLM53_KDA_REC_WARPS",
        "GLM53_KDA_REC_STAGES",
        "GLM53_KDA_REC_BV_CAP",
    ):
        assert f'-e "{knob}=${knob}"' in START
    assert '"${nccl_common[@]}"' in START and 'for e in "${nccl_common[@]}"' in START


def test_overlay_refuses_flashinfer_decode() -> None:
    assert "import" not in OVERLAY.lower().split("fused_kda_decode", 1)[0][-40:]
    assert "Does not" in OVERLAY or "does not" in OVERLAY
    assert "fused_kda_decode" in OVERLAY  # named only as a refused drop-in
    assert "from vllm" not in OVERLAY
    assert "num_warps = 1" in OVERLAY
    assert "num_stages = 3" in OVERLAY
    assert "min(next_power_of_2(V), 8)" in OVERLAY


def test_prod_start_hashes_kda_recurrent() -> None:
    hash_line = [ln for ln in PROD_START.splitlines() if "shape_hash=" in ln][0]
    assert "GLM53_KDA_REC_WARPS" in hash_line
    assert "GLM53_KDA_REC_STAGES" in hash_line
    assert "GLM53_KDA_REC_BV_CAP" in hash_line
    assert "printf 'GLM53_KDA_REC_WARPS=%s" in PROD_START


def test_docs_and_env_example_name_the_knobs() -> None:
    assert "GLM53_KDA_REC_WARPS" in ENV_EXAMPLE
    assert "GLM53_KDA_REC_STAGES" in ENV_EXAMPLE
    assert "GLM53_KDA_REC_BV_CAP" in ENV_EXAMPLE
    assert "Task 30" in ENV_EXAMPLE or "task 30" in ENV_EXAMPLE.lower()
    assert "GLM53_KDA_REC_WARPS" in DOCS_02
    assert "fused_recurrent_kda" in DOCS_02


if __name__ == "__main__":
    test_overlay_wired_six_ways()
    test_env_defaults_off_and_reach_both_ranks()
    test_overlay_refuses_flashinfer_decode()
    test_prod_start_hashes_kda_recurrent()
    test_docs_and_env_example_name_the_knobs()
    print("kda-recurrent wiring guards OK")
