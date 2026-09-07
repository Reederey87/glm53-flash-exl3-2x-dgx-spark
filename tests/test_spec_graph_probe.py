#!/usr/bin/env python3
"""CPU tests for the spec-graph probe classifier."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_spec_graph_probe.py"
WRAPPER = ROOT / "scripts/spec-graph-probe.sh"
SPEC = importlib.util.spec_from_file_location("spec_graph_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


COLLAPSE = {
    "drafts": 200,
    "draft_tokens": 1400,
    "accepted": 200,
    "pos": {0: 200.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0},
}
INCONCLUSIVE = {
    "drafts": 200,
    "draft_tokens": 1400,
    "accepted": 700,
    "pos": {0: 200.0, 1: 150.0, 2: 120.0, 3: 90.0, 4: 70.0, 5: 40.0, 6: 30.0},
}


def metrics(
    *,
    drafts: float,
    draft_tokens: float,
    accepted: float,
    pos: list[float],
) -> str:
    lines = [
        f'vllm:spec_decode_num_drafts_total{{engine="0",model_name="GLM-5.3-Flash-EXL3"}} {drafts}',
        f'vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="GLM-5.3-Flash-EXL3"}} {draft_tokens}',
        f'vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="GLM-5.3-Flash-EXL3"}} {accepted}',
    ]
    for index, value in enumerate(pos):
        lines.append(
            'vllm:spec_decode_num_accepted_tokens_per_pos_total'
            f'{{engine="0",model_name="GLM-5.3-Flash-EXL3",position="{index}"}} {value}'
        )
    return "\n".join(lines) + "\n"


def test_parse_live_prom_labels() -> None:
    text = metrics(
        drafts=260,
        draft_tokens=1820,
        accepted=1013,
        pos=[226, 184, 156, 136, 113, 103, 95],
    )
    snapshot = MODULE.parse_metrics(text)
    assert snapshot["drafts"] == 260
    assert snapshot["pos"][6] == 95


def test_pr70_false_positive_on_structured_ceiling() -> None:
    # Healthy structured 1.000/7.000: every position is 1.00. Kit PR #70 would FAIL.
    snapshot = {
        "drafts": 200,
        "draft_tokens": 1400,
        "accepted": 1400,
        "pos": {i: 200.0 for i in range(7)},
    }
    report = MODULE.classify(snapshot)
    assert report["decision"] == "healthy-ceiling"
    assert report["graph_vs_eager_arm_justified"] is False
    assert report["accepted_drafts_per_step"] == 7.0
    assert report["output_tokens_per_step"] == 8.0
    assert report["accepted_fraction"] == 1.0
    assert report["quantities"]["acceptance_length_1"].startswith("collapse")


def test_mixed_decay_is_healthy_not_collapse() -> None:
    snapshot = MODULE.parse_metrics(
        metrics(
            drafts=260,
            draft_tokens=1820,
            accepted=1013,
            pos=[226, 184, 156, 136, 113, 103, 95],
        )
    )
    report = MODULE.classify(snapshot)
    assert report["decision"] == "healthy-decay"
    assert report["graph_vs_eager_arm_justified"] is False
    assert report["pin_signature"] is False
    assert report["pos_ratios"][0] < 0.999
    assert report["pos_ratios"][-1] < report["pos_ratios"][0]


def test_length1_collapse_justifies_guarded_eager_arm() -> None:
    report = MODULE.classify(COLLAPSE)
    assert report["decision"] == "collapse"
    assert report["graph_vs_eager_arm_justified"] is True
    assert report["accepted_drafts_per_step"] == 1.0
    assert report["pin_signature"] is True
    assert report["argv_known"] is True


def test_flat_one_token_pin_is_collapse() -> None:
    snapshot = {
        "drafts": 150,
        "draft_tokens": 1050,
        "accepted": 150,
        "pos": {i: 150.0 for i in range(7)},
    }
    report = MODULE.classify(snapshot)
    assert report["decision"] == "collapse"
    assert report["accepted_fraction"] == 150 / 1050


def test_skip_below_min_drafts() -> None:
    report = MODULE.classify({"drafts": 12, "draft_tokens": 84, "accepted": 10, "pos": {}})
    assert report["decision"] == "skip"
    assert report["graph_vs_eager_arm_justified"] is False


def test_eager_live_argv_does_not_queue_graph_arm() -> None:
    report = MODULE.classify(COLLAPSE, enforce_eager=True)
    assert report["decision"] == "collapse"
    assert report["graph_vs_eager_arm_justified"] is False
    assert report["enforce_eager"] is True


def test_inconclusive_never_justifies_eager_arm() -> None:
    report = MODULE.classify(INCONCLUSIVE)
    assert report["decision"] == "inconclusive"
    assert report["graph_vs_eager_arm_justified"] is False
    unknown = MODULE.classify(INCONCLUSIVE, argv_known=False, capture_sizes=None)
    assert unknown["graph_vs_eager_arm_justified"] is False
    eager = MODULE.classify(INCONCLUSIVE, enforce_eager=True)
    assert eager["graph_vs_eager_arm_justified"] is False


def test_unknown_argv_withholds_collapse_arm() -> None:
    report = MODULE.classify(COLLAPSE, argv_known=False, capture_sizes=None)
    assert report["decision"] == "collapse"
    assert report["graph_vs_eager_arm_justified"] is False
    assert report["argv_known"] is False
    assert report["enforce_eager"] is None
    assert report["capture_sizes"] is None


def test_capture_sizes_match_k7_c4() -> None:
    report = MODULE.classify(
        {"drafts": 0, "draft_tokens": 0, "accepted": 0, "pos": {}},
        capture_sizes=(1, 2, 4, 8, 16, 24, 32),
    )
    assert report["capture_sizes_match_k7_c4"] is True
    bad = MODULE.classify(
        {"drafts": 0, "draft_tokens": 0, "accepted": 0, "pos": {}},
        capture_sizes=(1, 2, 3, 4, 6, 8, 12),
    )
    assert bad["capture_sizes_match_k7_c4"] is False


def _write_bindir(tmp_path: Path, curl_body: str, docker_body: str) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in (("curl", curl_body), ("docker", docker_body)):
        path = bindir / name
        path.write_text("#!/usr/bin/env bash\n" + body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return bindir


def _run_wrapper(tmp_path: Path, bindir: Path, env_extra: dict[str, str] | None = None):
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "PYTHON_BIN": str(Path(os.environ.get("PYTHON_BIN", "") or ROOT / ".venv/bin/python")),
        "HEAD_CONTAINER": "glm53-exl3-head",
    }
    env.pop("VLLM_API_KEY", None)
    if env_extra:
        env.update(env_extra)
    python = Path(env["PYTHON_BIN"])
    if not python.exists():
        env["PYTHON_BIN"] = "python3"
    return subprocess.run(
        ["bash", str(WRAPPER), "http://127.0.0.1:18000"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_wrapper_unauthenticated_macos_path(tmp_path: Path) -> None:
    prom = metrics(
        drafts=260,
        draft_tokens=1820,
        accepted=1013,
        pos=[226, 184, 156, 136, 113, 103, 95],
    )
    bindir = _write_bindir(
        tmp_path,
        f'cat <<\'EOF\'\n{prom}EOF\n',
        'exit 1\n',
    )
    result = _run_wrapper(tmp_path, bindir)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["decision"] == "healthy-decay"
    assert report["argv_known"] is False
    assert report["graph_vs_eager_arm_justified"] is False
    assert report["enforce_eager"] is None
    assert report["capture_sizes"] is None


def test_wrapper_unavailable_docker_withholds_collapse_arm(tmp_path: Path) -> None:
    prom = metrics(
        drafts=200,
        draft_tokens=1400,
        accepted=200,
        pos=[200, 0, 0, 0, 0, 0, 0],
    )
    bindir = _write_bindir(
        tmp_path,
        f'cat <<\'EOF\'\n{prom}EOF\n',
        'echo not-running\n',
    )
    result = _run_wrapper(tmp_path, bindir)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["decision"] == "collapse"
    assert report["argv_known"] is False
    assert report["graph_vs_eager_arm_justified"] is False


def test_wrapper_failed_exec_withholds_arm(tmp_path: Path) -> None:
    prom = metrics(
        drafts=200,
        draft_tokens=1400,
        accepted=200,
        pos=[200, 0, 0, 0, 0, 0, 0],
    )
    bindir = _write_bindir(
        tmp_path,
        f'cat <<\'EOF\'\n{prom}EOF\n',
        'if [[ "$*" == inspect* ]]; then echo true; exit 0; fi; exit 1\n',
    )
    result = _run_wrapper(tmp_path, bindir)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["decision"] == "collapse"
    assert report["argv_known"] is False
    assert report["graph_vs_eager_arm_justified"] is False


def test_wrapper_already_eager_target_parks_arm(tmp_path: Path) -> None:
    prom = metrics(
        drafts=200,
        draft_tokens=1400,
        accepted=200,
        pos=[200, 0, 0, 0, 0, 0, 0],
    )
    argv = (
        "python3 /usr/local/bin/vllm serve model --enforce-eager "
        "--cudagraph-capture-sizes 1 2 4 8 16 24 32"
    )
    bindir = _write_bindir(
        tmp_path,
        f'cat <<\'EOF\'\n{prom}EOF\n',
        f'if [[ "$*" == inspect* ]]; then echo true; exit 0; fi; printf "%s" "{argv}"\n',
    )
    result = _run_wrapper(tmp_path, bindir)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["decision"] == "collapse"
    assert report["argv_known"] is True
    assert report["enforce_eager"] is True
    assert report["graph_vs_eager_arm_justified"] is False
    assert report["capture_sizes"] == [1, 2, 4, 8, 16, 24, 32]


def test_cli_exit_codes(tmp_path: Path) -> None:
    ceiling = tmp_path / "ceiling.prom"
    ceiling.write_text(
        metrics(
            drafts=200,
            draft_tokens=1400,
            accepted=1400,
            pos=[200] * 7,
        )
    )
    collapse = tmp_path / "collapse.prom"
    collapse.write_text(
        metrics(
            drafts=200,
            draft_tokens=1400,
            accepted=200,
            pos=[200, 0, 0, 0, 0, 0, 0],
        )
    )
    ok = subprocess.run(
        ["python3", str(SCRIPT), "--metrics-file", str(ceiling)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    assert '"healthy-ceiling"' in ok.stdout
    bad = subprocess.run(
        ["python3", str(SCRIPT), "--metrics-file", str(collapse)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 1, bad.stderr
    assert '"collapse"' in bad.stdout
