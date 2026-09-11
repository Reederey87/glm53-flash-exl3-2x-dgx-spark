"""CPU-only tests for the task 35 §6 qualification window.

No cluster, no torch, no HTTP: the probe's pure helpers, the auditor's
decision logic, and the runner's arm/phase wiring are all exercised against
synthetic receipts. The real window's observations come from the cluster; these
tests exist so a parser or contract edit cannot silently change a verdict.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

KIT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = KIT_ROOT / "scripts"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load(SCRIPTS / "probe_v149_qualification.py", "glm53_probe_v149")
audit = _load(SCRIPTS / "audit_v149_qualification.py", "glm53_audit_v149")
window = _load(SCRIPTS / "run_v149_qualification_window.py", "glm53_window_v149")


# --- probe ------------------------------------------------------------------

def test_probe_kinds_cover_every_contract_lane():
    assert set(probe.KINDS) == set(audit.LANES)


def test_probe_structured_prompt_is_the_standing_gate_payload():
    # Same payload as tests/bench_decode.py's STRUCTURED_PROMPT; the whole point
    # is comparability with the 69-70 tok/s band in the existing receipts.
    assert probe.STRUCTURED_PROMPT.startswith("Count from 1 to 200.")


def _decode_run(tok_s=70.0, completion_tokens=200, nan=False, ttft_s=0.3, http=200, error=None):
    return {
        "http": http,
        "error": error,
        "ttft_s": ttft_s,
        "decode_s": 2.0,
        "tok_s": tok_s,
        "completion_tokens": completion_tokens,
        "prompt_tokens": 34,
        "finish_reason": "length",
        "nan": nan,
        "text_head": "1 2 3",
        "spec": {"drafts": 25, "draft_tokens": 175, "accepted": 175,
                 "accept_ratio": 1.0, "accepted_per_step": 7.0, "pos": [1.0] * 7},
    }


def _prefill_run(rate=1400.0, prompt_tokens=None, cached_tokens=0, target=60_000, ttft_s=57.0):
    prompt_tokens = target if prompt_tokens is None else prompt_tokens
    return {
        "http": 200,
        "error": None,
        "target": target,
        "filler_count": 10,
        "tokenize_estimate": prompt_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "ttft_s": ttft_s,
        "wall_s": ttft_s + 1,
        "prefill_tok_s": rate,
        "finish_reason": "stop",
        "nan": False,
        "text_head": "OK",
    }


def test_decode_run_invalid_accepts_a_full_block():
    assert probe.decode_run_invalid(_decode_run(), 200) is None


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"nan": True}, "NaN"),
        ({"http": 500}, "http 500"),
        ({"ttft_s": None}, "no first token"),
        ({"completion_tokens": 12}, "short completion"),
        ({"tok_s": None}, "completion_tokens"),
        ({"error": "URLError: refused"}, "URLError"),
    ],
)
def test_decode_run_invalid_rejects(kwargs, needle):
    run = _decode_run()
    run.update(kwargs)
    reason = probe.decode_run_invalid(run, 200)
    assert reason is not None and needle in reason


def test_prefill_run_invalid_accepts_a_cold_run():
    assert probe.prefill_run_invalid(_prefill_run()) is None


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"cached_tokens": 4096}, "warm request"),
        ({"prompt_tokens": 20_000}, "outside"),
        ({"http": 503}, "http 503"),
        ({"ttft_s": None}, "no first token"),
        ({"prefill_tok_s": None}, "no prefill rate"),
        ({"prompt_tokens": 0}, "no prompt_tokens"),
    ],
)
def test_prefill_run_invalid_rejects(kwargs, needle):
    run = _prefill_run()
    run.update(kwargs)
    reason = probe.prefill_run_invalid(run)
    assert reason is not None and needle in reason


def test_summarize_reports_median_and_flags_nan_and_cache_hits():
    decode = probe.summarize("structured", [_decode_run(70.0), _decode_run(80.0)], [])
    assert decode["valid_runs"] == 2
    assert decode["tok_s_median"] == 75.0
    assert decode["any_nan"] is False
    assert decode["accepted_per_step_median"] == 7.0
    flagged = probe.summarize("structured", [_decode_run(nan=True)], [])
    assert flagged["any_nan"] is True
    prefill = probe.summarize("prefill60k", [_prefill_run(cached_tokens=99)], [])
    assert prefill["any_cache_hit"] is True
    assert prefill["prefill_tok_s_median"] == 1400.0


def test_probe_refuses_to_measure_an_unhealthy_server(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "health", lambda: 503)
    rc = probe.main(["--kind", "structured", "--runs", "1", "--out", str(tmp_path / "o.json")])
    assert rc == 2
    assert not (tmp_path / "o.json").exists()


def test_probe_exits_nonzero_when_a_run_is_invalid(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "health", lambda: 200)
    monkeypatch.setattr(probe, "served_model", lambda: "GLM-5.3-Flash-EXL3")
    monkeypatch.setattr(probe, "spec_snapshot", lambda: {})
    monkeypatch.setattr(probe, "decode_run", lambda *a, **k: _decode_run(nan=True))
    out = tmp_path / "o.json"
    rc = probe.main(["--kind", "structured", "--runs", "3", "--out", str(out)])
    assert rc == 1
    payload = json.loads(out.read_text())
    assert payload["valid_runs"] == 0 and len(payload["invalid_runs"]) == 3


# --- auditor ----------------------------------------------------------------

def _probe_receipt(kind, values, *, valid=None, any_nan=False, any_cache_hit=False):
    key = "tok_s" if kind in ("structured", "essay", "hashmap") else "prefill_tok_s"
    runs = []
    for index, value in enumerate(values, start=1):
        if kind in ("structured", "essay", "hashmap"):
            run = _decode_run(tok_s=value)
        else:
            target = probe.PREFILL_KINDS[kind]
            run = _prefill_run(rate=value, prompt_tokens=target, target=target)
        run["i"] = index
        runs.append(run)
    return {
        "schema": 1,
        "kind": kind,
        "metric": key,
        "runs": runs,
        "invalid_runs": [],
        "valid_runs": len(values) if valid is None else valid,
        f"{key}_median": sorted(values)[len(values) // 2],
        "any_nan": any_nan,
        "any_cache_hit": any_cache_hit,
    }


def _window_receipt(tmp_path, arm_values, *, gates_ok=True, write_probes=True):
    """arm_values: {arm: {lane: [values]}} — one probe file per (arm, lane)."""
    probes = {}
    for arm, lanes in arm_values.items():
        probes[arm] = {}
        for lane, values in lanes.items():
            name = f"task35b-{arm}-{lane}.json"
            if write_probes:
                (tmp_path / name).write_text(json.dumps(_probe_receipt(lane, values)))
            probes[arm][lane] = name
    arms = {}
    for arm in audit.ARMS:
        arms[arm] = {
            "image_tag": audit.ARM_IMAGE[arm],
            "exllamav3_version": audit.ARM_EXLLAMAV3[arm],
            "worker_image_tag": audit.ARM_IMAGE[arm],
            "worker_exllamav3_version": audit.ARM_EXLLAMAV3[arm],
            "jit_stamp": "stamp-b" if arm in ("b", "b2") else "stamp-a",
        }
    gates = {
        "acceptance_rc": 0,
        "memfree_head_gib": 4.1,
        "memfree_worker_gib": 3.3,
        "pool_line": "GPU KV cache size: 1,396,551 tokens",
        "pool_line_before": "GPU KV cache size: 1,396,551 tokens",
        "jit_stamp": "stamp-b",
        "jit_stamp_arm_b": "stamp-b",
    }
    if not gates_ok:
        gates["acceptance_rc"] = 1
    return {"schema": 1, "window": "task35b", "arms": arms, "gates": gates, "probes": probes}


def _uniform(value_map):
    """Build arm_values from {arm: {lane: value}}."""
    return {arm: {lane: [value] * audit.REQUIRED_RUNS[lane] for lane, value in lanes.items()}
            for arm, lanes in value_map.items()}


LANE_BASELINE = {"structured": 70.0, "essay": 25.0, "hashmap": 30.0,
                 "prefill60k": 1450.0, "prefill240k": 1400.0}


def _values_with(overrides):
    """Per-arm values: control arms at baseline, candidate arms at overrides."""
    out = {}
    for arm in audit.ARMS:
        base = dict(LANE_BASELINE)
        if arm in audit.CANDIDATE_ARMS:
            base.update(overrides)
        out[arm] = {lane: [value] * audit.REQUIRED_RUNS[lane] for lane, value in base.items()}
    return out


def test_auditor_adopts_a_non_inferior_candidate(tmp_path):
    receipt = _window_receipt(tmp_path, _values_with({"structured": 69.5}))
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ADOPT", result["errors"]
    assert result["lanes"]["structured"]["candidate_over_control"] == round(69.5 / 70.0, 4)


def test_auditor_reverts_a_regressed_lane(tmp_path):
    receipt = _window_receipt(tmp_path, _values_with({"prefill240k": 1200.0}))
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "REVERT"
    assert result["lanes"]["prefill240k"]["verdict"] == "FAIL"
    assert result["lanes"]["structured"]["verdict"] == "PASS"


def test_auditor_reports_a_win_without_requiring_one(tmp_path):
    receipt = _window_receipt(tmp_path, _values_with({"structured": 75.0, "hashmap": 33.0}))
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ADOPT"
    assert result["lanes"]["structured"]["candidate_over_control"] > 1.0


def test_auditor_is_inconclusive_when_the_control_arms_drift(tmp_path):
    values = _values_with({})
    # A2 is the same image as A; a 9% gap means the window moved under itself.
    values["a2"]["essay"] = [27.3] * audit.REQUIRED_RUNS["essay"]
    receipt = _window_receipt(tmp_path, values)
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["lanes"]["essay"]["verdict"] == "INCONCLUSIVE"


def test_auditor_aborts_on_too_few_valid_runs(tmp_path):
    values = _values_with({})
    values["b"]["structured"] = [70.0] * 3
    receipt = _window_receipt(tmp_path, values)
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("§6 requires 9" in message for message in result["errors"])


def test_auditor_aborts_on_a_nan_run(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    (tmp_path / "task35b-b-essay.json").write_text(
        json.dumps(_probe_receipt("essay", [25.0] * 9, any_nan=True))
    )
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("NaN" in message for message in result["errors"])


def test_auditor_aborts_on_a_warm_cold_prefill(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    (tmp_path / "task35b-a-prefill60k.json").write_text(
        json.dumps(_probe_receipt("prefill60k", [1450.0] * 5, any_cache_hit=True))
    )
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("prefix cache" in message for message in result["errors"])


def test_auditor_aborts_when_an_arm_is_not_the_image_it_claims(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["arms"]["b"]["image_tag"] = audit.ARM_IMAGE["a"]
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("booted" in message for message in result["errors"])


def test_auditor_aborts_when_the_worker_ran_the_wrong_revision(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["arms"]["b2"]["worker_exllamav3_version"] = "1.4.7"
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("worker reports exllamav3" in message for message in result["errors"])


def test_auditor_aborts_on_a_missing_probe_receipt(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    (tmp_path / "task35b-a2-hashmap.json").unlink()
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("arm a2 lane hashmap" in message for message in result["errors"])


def test_auditor_aborts_on_a_failed_acceptance_battery(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values, gates_ok=False)
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("acceptance battery" in message for message in result["errors"])


def test_auditor_aborts_when_the_kv_pool_moved(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["gates"]["pool_line"] = "GPU KV cache size: 1,200,000 tokens"
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("KV pool changed" in message for message in result["errors"])


def test_auditor_aborts_when_the_final_jit_stamp_is_not_the_candidate_stamp(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["gates"]["jit_stamp"] = "stamp-a"
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("JIT shape stamp" in message for message in result["errors"])


# --- window runner wiring ---------------------------------------------------

def test_window_is_a_pre_registered_a_b_b_a_over_two_images():
    assert window.PHASES == (
        "preflight", "disarm",
        "arm_a", "measure_a",
        "arm_b", "measure_b",
        "arm_b2", "measure_b2",
        "arm_a2", "measure_a2",
        "restore", "gates", "rearm", "judge",
    )
    assert [window.ARMS[a]["tag"] for a in ("a", "b", "b2", "a2")] == [
        "glm53-selfbuild:e3-w3-zfill",
        "glm53-selfbuild:e3-w3-zfill-v149",
        "glm53-selfbuild:e3-w3-zfill-v149",
        "glm53-selfbuild:e3-w3-zfill",
    ]
    assert window.PRODUCTION_IMAGE == window.ARMS["b"]["tag"]


def test_window_lane_counts_match_the_auditor_contract():
    assert {lane: runs for lane, (runs, _timeout) in window.LANES.items()} == audit.REQUIRED_RUNS
    assert set(window.LANES) == set(audit.LANES)


def test_window_arm_versions_match_the_auditor_identity():
    for arm, spec in window.ARMS.items():
        assert spec["exllamav3"] == audit.ARM_EXLLAMAV3[arm]
        assert spec["tag"] == audit.ARM_IMAGE[arm]


def test_set_image_appends_a_last_wins_line(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("IMAGE=glm53-selfbuild:e3-w3-zfill-v149\nGLM53_ADAPTIVE_K=ema\n")
    monkeypatch.setattr(window, "ENV_FILE", env)
    window.set_image("glm53-selfbuild:e3-w3-zfill")
    text = env.read_text()
    assert text.count("IMAGE=") == 2
    assert window.effective_env_all()["IMAGE"] == "glm53-selfbuild:e3-w3-zfill"
    assert window.effective_env_all()["GLM53_ADAPTIVE_K"] == "ema"


def test_effective_env_all_ignores_comments_and_bad_keys(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# IMAGE=not-this\n"
        "IMAGE=first\n"
        "IMAGE=second\n"
        "NOT A KEY=value\n"
        'QUOTED="spaced value"\n'
    )
    monkeypatch.setattr(window, "ENV_FILE", env)
    parsed = window.effective_env_all()
    assert parsed["IMAGE"] == "second"
    assert parsed["QUOTED"] == "spaced value"
    assert "NOT A KEY" not in parsed


def test_needs_env_restore_tracks_the_touched_flag_and_the_live_image(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"IMAGE={window.PRODUCTION_IMAGE}\n")
    monkeypatch.setattr(window, "ENV_FILE", env)
    assert window.needs_env_restore({}) is False
    assert window.needs_env_restore({"backup": "x"}) is False
    assert window.needs_env_restore({"backup": "x", "env_touched": True}) is True
    env.write_text("IMAGE=glm53-selfbuild:e3-w3-zfill\n")
    assert window.needs_env_restore({"backup": "x"}) is True


def test_judge_writes_the_audit_next_to_the_window_receipt(monkeypatch, tmp_path):
    receipt = tmp_path / "task35b-window-20260911-000000.json"
    receipt.write_text(json.dumps({"arms": {}, "gates": {}, "probes": {}}))
    monkeypatch.setattr(window, "_RECEIPT", receipt)
    state: dict = {}
    window.phase_judge(state)
    assert state["verdict"] == "ABORT"
    assert state["audit_receipt"] == "task35b-audit-20260911-000000.json"
    assert (tmp_path / state["audit_receipt"]).is_file()


# --- ISA probe (task 38 item 2) ---------------------------------------------

isa = _load(SCRIPTS / "probe_isa_e2m1_silicon.py", "glm53_probe_isa")


def test_isa_reference_follows_the_ptx_nibble_order():
    # PTX ISA: `a` is converted into the upper nibble, `b` into the lower.
    # 1.0 -> code 2, 2.0 -> code 4.
    assert isa.expected_pair(1.0, 2.0) == 0x24
    assert isa.reversed_pair(1.0, 2.0) == 0x42
    # The task 38 record shows 0x42 for this pair, i.e. the original probe fed
    # (2.0, 1.0) -- or its inputs were ordered opposite to the PTX spec.
    assert isa.expected_pair(2.0, 1.0) == 0x42


@pytest.mark.parametrize(
    "value,code",
    [
        (0.0, 0b0000), (0.5, 0b0001), (1.0, 0b0010), (1.5, 0b0011),
        (2.0, 0b0100), (3.0, 0b0101), (4.0, 0b0110), (6.0, 0b0111),
        (-1.0, 0b1010), (-6.0, 0b1111),
    ],
)
def test_isa_e2m1_codes(value, code):
    assert isa.e2m1_code(value) == code


def test_isa_e2m1_rounds_to_nearest_even_on_a_tie():
    assert isa.e2m1_code(0.25) == 0b0000  # tie 0.0/0.5 -> even code 0
    assert isa.e2m1_code(0.75) == 0b0010  # tie 0.5/1.0 -> even code 2
    assert isa.e2m1_code(0.4) == 0b0001
    assert isa.e2m1_code(0.6) == 0b0001
    assert isa.e2m1_code(2.7) == 0b0101


def test_isa_e2m1_saturates_and_rejects_nan():
    assert isa.e2m1_code(1e30) == 0b0111
    assert isa.e2m1_code(-1e30) == 0b1111
    with pytest.raises(ValueError):
        isa.e2m1_code(float("nan"))


def test_isa_ptx_uses_one_parameter_fed_kernel():
    ptx = isa.build_ptx()
    assert ".target sm_121a" in ptx
    assert ".entry probe_e2m1" in ptx
    # One kernel, one conversion: the cases are separate launches.
    assert ptx.count("cvt.rn.satfinite.e2m1x2.f32") == 1
    # Operands must arrive as parameters, never as `mov.f32` immediates: ptxas
    # mis-materializes the immediates and every case then reads 0x00.
    assert "ld.param.f32" in ptx
    assert "mov.f32" not in ptx
    assert "0f" not in ptx
    # The destination is a single byte.
    assert ".reg .b8" in ptx
    assert "st.global.u8" in ptx


def test_isa_judge_confirms_spec_correct_silicon():
    observed = [isa.expected_pair(a, b) for _n, a, b, _t in isa.CASES]
    result = isa.judge(observed)
    assert result["verdict"] == "SILICON_CONFIRMED"
    assert result["errors"] == []
    assert result["reversed_order_cases"] == 0
    assert result["cases_matching_spec"] == len(isa.CASES)


def test_isa_judge_rejects_a_reversed_nibble_order():
    observed = [isa.reversed_pair(a, b) for _n, a, b, _t in isa.CASES]
    result = isa.judge(observed)
    assert result["verdict"] == "SILICON_MISMATCH"
    assert result["reversed_order_cases"] > 0
    assert any("REVERSED nibble order" in message for message in result["errors"])


def test_isa_judge_rejects_a_wrong_conversion():
    observed = [isa.expected_pair(a, b) for _n, a, b, _t in isa.CASES]
    observed[0] = 0x00
    result = isa.judge(observed)
    assert result["verdict"] == "SILICON_MISMATCH"
    assert any("recorded-reversed-2.0-1.0" in message for message in result["errors"])


def test_isa_judge_rejects_an_all_zero_result():
    """The exact failure the immediate-operand form produced must not pass.

    `(0.0, 0.0)` really does encode as 0x00, so that one case matches; the
    discriminating cases must still be reported as mismatches.
    """
    result = isa.judge([0x00] * len(isa.CASES))
    assert result["verdict"] == "SILICON_MISMATCH"
    assert result["cases_matching_spec"] == 1
    flagged = {message.split(" ")[0] for message in result["errors"]}
    assert {"distinct-1.0-2.0", "top-of-range", "negative"} <= flagged
    assert "zero" not in flagged


def test_isa_ties_are_recorded_but_not_gated():
    rows = {row["case"]: row for row in isa.judge([0x00] * len(isa.CASES))["cases"]}
    assert rows["tie-0.25-0.75"]["tie"] is True
    assert isa.judge([0x00] * len(isa.CASES))["gated_cases"] == len(isa.CASES) - 1


def test_isa_judge_rejects_a_short_result():
    with pytest.raises(RuntimeError):
        isa.judge([0x00] * (len(isa.CASES) - 1))


def test_isa_ptx_only_mode_needs_no_gpu(monkeypatch, tmp_path):
    def explode(*_a, **_k):
        raise AssertionError("--ptx-only must not touch the driver")

    monkeypatch.setattr(isa, "run_on_gpu", explode)
    assert isa.main(["--ptx-only", "--workdir", str(tmp_path)]) == 0
    assert (tmp_path / "run_probe.ptx").is_file()


def test_isa_reports_probe_unavailable_when_the_driver_refuses(monkeypatch, tmp_path):
    def refuse(_ptx, _cases=isa.CASES):
        raise RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")

    monkeypatch.setattr(isa, "run_on_gpu", refuse)
    out = tmp_path / "isa.json"
    assert isa.main(["--out", str(out)]) == 2
    payload = json.loads(out.read_text())
    assert payload["verdict"] == "PROBE_UNAVAILABLE"
    assert "OUT_OF_MEMORY" in payload["error"]


def test_isa_main_writes_a_silicon_confirmed_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(isa, "run_on_gpu", lambda ptx, cases=isa.CASES: [
        isa.expected_pair(a, b) for _n, a, b, _t in cases
    ])
    out = tmp_path / "isa.json"
    assert isa.main(["--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["verdict"] == "SILICON_CONFIRMED"
    assert payload["probe"] == "cvt.rn.satfinite.e2m1x2.f32"


def test_window_has_no_isa_phase():
    """The probe runs alongside production (1-byte alloc + 1-thread launch
    succeed with the serving stack up), so the window must not stop production
    for it."""
    assert "isa" not in window.PHASES
    assert not hasattr(window, "ISA_PROBE")
    assert not hasattr(window, "phase_isa")
