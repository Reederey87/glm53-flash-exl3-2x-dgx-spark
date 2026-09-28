#!/usr/bin/env python3
"""CPU-only tests for `scripts/probe_agentic_cache.py` (the W43 gate battery).

The instrument is the deliverable's evidence, so the tests have to show it can
fail. Every verdict case below is built twice: once in the shape that must PASS
and once with the specific defect the check exists for (field omitted, cold
cache reporting a hit, counter disagreement, SSE/HTTP disagreement, a stale
answer, a chaff throughput loss). A check that cannot be driven to FAIL is not
a gate.

Also covered: the metric parser (summed and label-keyed forms, `_created`
gauges excluded), the prompt builder's needle placement, and the decode-floor
comparison, including the skipped form when no decode receipt is supplied.

Run:  python3 tests/test_probe_agentic_cache.py   (or pytest)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import probe_agentic_cache as probe  # noqa: E402


METRICS_BODY = """\
# HELP vllm:num_requests_waiting Number of waiting requests.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="m"} 2.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="m",reason="capacity"} 1.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="m",reason="deferred"} 3.0
vllm:num_requests_running{engine="0",model_name="m"} 4.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 100.0
vllm:prefix_cache_hits_created{engine="0",model_name="m"} 1.7e+09
vllm:num_preemptions_total{engine="0",model_name="m"} 0.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m",position="0"} 5.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m",position="1"} 3.0
"""


def args(**over) -> SimpleNamespace:
    base = dict(
        floor={"structured": 68.8, "prose": 30.0, "essay": 20.0},
        decode={},
        attr_tokens=1000,
        attr_max_tokens=8,
        needle_depth=512,
        salt="S",
        overload_requests=6,
    )
    base.update(over)
    return SimpleNamespace(**base)


def case(cached, hits, prompt=1000, details=True, error=None, nan=False, **over):
    rec = {
        "cached_tokens": cached,
        "hits_delta": hits,
        "queries_delta": prompt,
        "prompt_tokens": prompt,
        "details_present": details,
        "usage_present": True,
        "error": error,
        "nan": nan,
        "answer_code": None,
        "completion_tokens": 8,
        "ttft_s": 0.1,
        "wall_s": 1.0,
        "gaps_s": [],
        "label": over.pop("label", "case"),
        "stream": over.pop("stream", False),
        "no_store": over.pop("no_store", None),
    }
    rec.update(over)
    return rec


def attribution_receipt(cold=None, replay=None, negative=None, sse=None, overload=None):
    return {
        "cases": {
            "cold": cold if cold is not None else case(0, 0, details=True, label="cold"),
            "replay": replay if replay is not None else case(7000, 7000, label="replay"),
            "negative": negative if negative is not None else case(0, 0, label="negative"),
            "replay_sse": sse if sse is not None else case(7000, 7000, label="replay_sse",
                                                          stream=True),
        },
        "overload": overload if overload is not None else {
            "requests": 6, "completed": 6, "errors": [],
            "max_waiting": 2.0, "max_waiting_capacity": 1.0,
            "max_running": 4.0, "max_kv_cache_usage_perc": 0.9,
            "preemptions_delta": 0.0, "results": [],
        },
    }


def chaff(label, ok=True, error=None, nan=False, code="CODE-aaaaaaaaaaaa"):
    return {
        "label": label, "error": error, "nan": nan,
        "answer_code": code if ok else "CODE-ffffffffffff",
        "expected_code": code, "prompt_tokens": 31000,
    }


def arm(name, hit_pct, tok_s, *, stale=False, error=False, chaff_wrong=False,
        nan=False, prefill_s=10.0):
    return {
        "arm": name,
        "chaff_no_store": name == "treatment",
        "chaff": [chaff(f"chaff{i}", ok=not chaff_wrong,
                        error="HTTPError: 500" if (error and i == 0) else None,
                        nan=nan)
                  for i in range(3)],
        "replays": [{
            "label": f"replay{i}", "error": None, "nan": nan,
            "answer_code": "CODE-bbbbbbbbbbbb" if (stale and i == 0) else "CODE-aaaaaaaaaaaa",
            "expected_code": "CODE-aaaaaaaaaaaa",
            "return_hit_pct": hit_pct, "prompt_tokens": 46500,
            "hits_delta": int(46500 * hit_pct / 100),
        } for i in range(2)],
        "chaff_wall_s": 100.0,
        "chaff_prompt_tokens": 93000,
        "chaff_prompt_tok_s": tok_s,
        "chaff_prefill_count": 18.0,
        "chaff_prefill_mean_s": prefill_s,
        "chaff_queue_mean_s": 0.01,
        "mean_return_hit_pct": hit_pct,
    }


def checks(verdict) -> dict[str, dict]:
    return {c["check"]: c for c in verdict["checks"]}


# ------------------------------------------------------------------- helpers


def test_metric_parser_sums_and_keeps_labels() -> None:
    c = probe.Client("http://x")
    c.get_text = lambda *a, **k: METRICS_BODY  # type: ignore[method-assign]
    summed = c.metrics()
    assert summed["vllm:num_requests_waiting"] == 2.0
    assert summed["vllm:num_requests_waiting_by_reason"] == 4.0
    assert summed["vllm:spec_decode_num_accepted_tokens_total"] == 8.0
    assert "vllm:prefix_cache_hits_created" not in summed, "_created gauges must be excluded"
    labelled = c.metrics_labelled()
    key = next(k for k in labelled if k.startswith("vllm:num_requests_waiting_by_reason")
               and 'reason="capacity"' in k)
    assert labelled[key] == 1.0


def test_usage_readers_distinguish_zero_from_missing() -> None:
    assert probe.cached_tokens({"prompt_tokens_details": {"cached_tokens": 0}}) == 0
    assert probe.cached_tokens({"prompt_tokens_details": None}) is None
    assert probe.cached_tokens({"prompt_tokens": 10}) is None
    assert probe.cached_tokens(None) is None
    assert probe.prompt_tokens({"prompt_tokens": 10}) == 10
    assert probe.prompt_tokens(None) is None


def test_prompt_builder_places_a_unique_needle() -> None:
    import random
    rng = random.Random(7)
    p1 = probe.build_prompt(rng, 4000, "CODE-aaaaaaaaaaaa", 512, "salt-a")
    p2 = probe.build_prompt(rng, 4000, "CODE-aaaaaaaaaaaa", 512, "salt-b")
    assert p1 != p2, "the salt must make each run's prompt unique"
    assert probe.extract_code(p1) == "CODE-aaaaaaaaaaaa"
    assert probe.extract_code("no code here") is None
    assert "CODE-aaaaaaaaaaaa" in p1.split("Reply with only")[0], (
        "the needle must live in the prefix, not in the final instruction"
    )
    assert probe.contains_nan("all fine") is False
    assert probe.contains_nan("value is NaN") is True
    assert probe.contains_nan("locklock") is True


def test_parse_floors_and_decode_receipts() -> None:
    assert probe.parse_floors("structured=68.8, prose=30") == {"structured": 68.8, "prose": 30.0}
    assert probe.parse_floors("") == {}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "structured-measured.json"
        p.write_text(json.dumps({"phase": "structured", "tok_s_median": 71.0}))
        q = Path(tmp) / "essay.json"
        q.write_text(json.dumps({"phase": "essay", "tok_s_median": 25.0}))
        r = Path(tmp) / "prose.json"
        r.write_text(json.dumps({"phase": "prose", "tok_s_median": 33.0}))
        lanes = probe.read_decode_receipts([str(p), str(q), str(r)])
    assert lanes == {"structured": 71.0, "essay": 25.0, "prose": 33.0}


# --------------------------------------------------------- attribution gate


def test_attribution_gate_passes_on_a_correct_receipt() -> None:
    v = probe.verdict_attribution(attribution_receipt(), args())
    assert v["passed"], [c for c in v["checks"] if not c["ok"]]


def test_attribution_gate_fails_when_the_field_is_omitted() -> None:
    """The flag-off shape: prompt_tokens_details is absent, so cold is unreadable."""
    v = probe.verdict_attribution(
        attribution_receipt(cold=case(None, 0, details=False)), args())
    assert not v["passed"]
    assert not checks(v)["cold_field_present"]["ok"]
    assert not checks(v)["cold_cached_zero"]["ok"]


def test_attribution_gate_fails_when_a_cold_request_reports_a_hit() -> None:
    v = probe.verdict_attribution(
        attribution_receipt(cold=case(900, 900)), args())
    assert not v["passed"]
    assert not checks(v)["cold_cached_zero"]["ok"]


def test_attribution_gate_fails_when_the_replay_reports_zero() -> None:
    """A cache that is inert looks exactly like this."""
    v = probe.verdict_attribution(
        attribution_receipt(replay=case(0, 0), sse=case(0, 0, stream=True)), args())
    assert not v["passed"]
    assert not checks(v)["replay_hit_positive"]["ok"]


def test_attribution_gate_fails_when_the_counter_disagrees() -> None:
    v = probe.verdict_attribution(
        attribution_receipt(replay=case(7000, 4096)), args())
    assert not v["passed"]
    assert not checks(v)["replay_counter_agrees"]["ok"]


def test_attribution_gate_fails_when_the_negative_control_hits() -> None:
    v = probe.verdict_attribution(
        attribution_receipt(negative=case(4096, 4096)), args())
    assert not v["passed"]
    assert not checks(v)["negative_zero"]["ok"]
    assert not checks(v)["negative_counter_agrees"]["ok"]


def test_attribution_gate_fails_when_sse_disagrees_with_http() -> None:
    v = probe.verdict_attribution(
        attribution_receipt(sse=case(4096, 4096, stream=True)), args())
    assert not v["passed"]
    assert not checks(v)["sse_matches_http"]["ok"]


def test_attribution_gate_fails_when_sse_usage_is_absent() -> None:
    bad = case(7000, 7000, stream=True)
    bad["usage_present"] = False
    v = probe.verdict_attribution(attribution_receipt(sse=bad), args())
    assert not v["passed"]
    assert not checks(v)["sse_usage_present"]["ok"]


def test_attribution_gate_fails_on_overload_errors() -> None:
    ov = {"requests": 6, "completed": 5, "errors": ["request 3: HTTPError: 500"],
          "max_waiting": 2.0, "max_waiting_capacity": 1.0, "max_running": 4.0,
          "max_kv_cache_usage_perc": 0.9, "preemptions_delta": 0.0, "results": []}
    v = probe.verdict_attribution(attribution_receipt(overload=ov), args())
    assert not v["passed"]
    assert not checks(v)["overload_completed"]["ok"]
    assert not checks(v)["overload_no_errors"]["ok"]


def test_attribution_gate_fails_on_nan() -> None:
    v = probe.verdict_attribution(
        attribution_receipt(replay=case(7000, 7000, nan=True)), args())
    assert not v["passed"]
    assert not checks(v)["no_nan"]["ok"]


# ----------------------------------------------------------- retention gate


def test_retention_gate_passes_on_a_treatment_win() -> None:
    v = probe.verdict_retention(arm("control", 38.5, 1000.0),
                                arm("treatment", 98.6, 1010.0), args())
    assert v["passed"], [c for c in v["checks"] if not c["ok"]]
    assert v["return_hit_gain_pp"] == 60.1


def test_retention_gate_fails_below_the_registered_gain() -> None:
    """The measured no-op shape: both arms at the control's retention."""
    v = probe.verdict_retention(arm("control", 38.56, 1000.0),
                                arm("treatment", 38.56, 1000.0), args())
    assert not v["passed"]
    c = checks(v)["return_hit_gain_pp"]
    assert not c["ok"] and "gain=0.0pp" in c["detail"]


def test_retention_gate_fails_just_under_the_threshold() -> None:
    v = probe.verdict_retention(arm("control", 40.0, 1000.0),
                                arm("treatment", 54.9, 1000.0), args())
    assert not v["passed"]
    assert not checks(v)["return_hit_gain_pp"]["ok"]


def test_retention_gate_fails_on_a_stale_answer() -> None:
    v = probe.verdict_retention(arm("control", 38.5, 1000.0),
                                arm("treatment", 98.6, 1010.0, stale=True), args())
    assert not v["passed"]
    assert not checks(v)["treatment_replays_return_own_code"]["ok"]


def test_retention_gate_fails_on_a_chaff_prefill_regression() -> None:
    """The sequential row is a diagnostic; the paired mode owns this gate."""
    v = probe.verdict_retention(arm("control", 38.5, 1000.0, prefill_s=10.0),
                                arm("treatment", 98.6, 1010.0, prefill_s=12.4), args())
    assert v["passed"], [c for c in v["checks"] if not c["ok"]]
    seq = checks(v)["chaff_prefill_time_ratio_sequential"]
    assert seq["skipped"], "sequential arms must not gate the chaff cost"
    assert "ratio=1.24" in seq["detail"]


def test_retention_gate_records_the_client_rate_without_gating_on_it() -> None:
    """A 0.76x client rate must not fail the gate when prefill agrees."""
    v = probe.verdict_retention(arm("control", 38.5, 1234.5, prefill_s=10.0),
                                arm("treatment", 98.6, 937.3, prefill_s=10.05), args())
    assert v["passed"], [c for c in v["checks"] if not c["ok"]]
    diag = checks(v)["chaff_client_rate_diagnostic"]
    assert diag["skipped"] and "0.7593" in diag["detail"]


# ---------------------------------------------------------- chaff-cost gate


def cost_receipt(*, caching=33.0, no_store=33.2, spread=None, **kw):
    base = {
        "caching_median_s": caching,
        "no_store_median_s": no_store,
        "caching_mean_s": caching,
        "no_store_mean_s": no_store,
        "paired_delta_mean_s": (no_store - caching) if caching and no_store else None,
        "paired_delta_median_s": (no_store - caching) if caching and no_store else None,
        "paired_delta_min_s": -5.0,
        "paired_delta_max_s": 5.0,
        "spread_s": spread if spread is not None else 20.0,
        "errors": [],
        "wrong_answers": [],
        "nan": False,
    }
    base.update(kw)
    return base


def test_chaff_cost_gate_passes_when_the_paired_ratio_holds() -> None:
    v = probe.verdict_chaff_cost(cost_receipt(), args())
    assert v["passed"], [c for c in v["checks"] if not c["ok"]]
    assert checks(v)["paired_median_prefill_ratio"]["ok"]


def test_chaff_cost_gate_fails_on_a_real_paired_regression() -> None:
    """A 1.54x paired regression, the shape a sequential run reported, fails."""
    v = probe.verdict_chaff_cost(cost_receipt(caching=33.0, no_store=50.8), args())
    assert not v["passed"]
    c = checks(v)["paired_median_prefill_ratio"]
    assert not c["ok"] and "ratio=1.5394" in c["detail"]


def test_chaff_cost_gate_fails_just_over_the_ratio_bound() -> None:
    v = probe.verdict_chaff_cost(cost_receipt(caching=33.0, no_store=38.0), args())
    assert not v["passed"]
    assert not checks(v)["paired_median_prefill_ratio"]["ok"]


def test_chaff_cost_gate_tolerates_the_measured_spread() -> None:
    """The 1.6x spread is reported, but it is not by itself a failure."""
    v = probe.verdict_chaff_cost(cost_receipt(spread=19.9), args())
    assert v["passed"], [c for c in v["checks"] if not c["ok"]]
    assert "19.9" in checks(v)["paired_median_prefill_ratio"]["detail"]


def test_chaff_cost_gate_fails_when_the_histogram_is_absent() -> None:
    v = probe.verdict_chaff_cost(cost_receipt(caching=None, no_store=None), args())
    assert not v["passed"]
    assert not checks(v)["paired_median_prefill_ratio"]["ok"]


def test_chaff_cost_gate_fails_on_errors_wrong_answers_or_nan() -> None:
    for kw in ({"errors": ["HTTPError: 500"]},
               {"wrong_answers": ["pair1-st"]},
               {"nan": True}):
        v = probe.verdict_chaff_cost(cost_receipt(**kw), args())
        assert not v["passed"], kw


def test_retention_gate_fails_on_request_errors_and_nan() -> None:
    v = probe.verdict_retention(arm("control", 38.5, 1000.0, error=True, nan=True),
                                arm("treatment", 98.6, 1010.0), args())
    assert not v["passed"]
    assert not checks(v)["control_no_request_errors"]["ok"]
    assert not checks(v)["control_no_nan"]["ok"]


def test_retention_gate_fails_on_wrong_chaff_answers() -> None:
    v = probe.verdict_retention(arm("control", 38.5, 1000.0),
                                arm("treatment", 98.6, 1010.0, chaff_wrong=True), args())
    assert not v["passed"]
    assert not checks(v)["treatment_chaff_answers_correct"]["ok"]


def test_retention_gate_requires_both_arms() -> None:
    v = probe.verdict_retention(None, arm("treatment", 98.6, 1010.0), args())
    assert not v["passed"]
    assert not checks(v)["arms_present"]["ok"]


def test_retention_gate_skips_decode_floors_without_a_receipt() -> None:
    v = probe.verdict_retention(arm("control", 38.5, 1000.0),
                                arm("treatment", 98.6, 1010.0), args())
    for lane in ("structured", "prose", "essay"):
        c = checks(v)[f"decode_floor_{lane}"]
        assert c["skipped"] and c["ok"]
    assert v["passed"], "a skipped check must not fail the gate"


def test_retention_gate_fails_a_decode_floor_regression() -> None:
    a = args(decode={"structured": 60.0})
    v = probe.verdict_retention(arm("control", 38.5, 1000.0),
                                arm("treatment", 98.6, 1010.0), a)
    assert not v["passed"]
    assert not checks(v)["decode_floor_structured"]["ok"]
    assert checks(v)["decode_floor_essay"]["skipped"]


def test_retention_gate_fails_when_an_arm_is_missing_from_the_receipt() -> None:
    v = probe.verdict_retention(arm("control", 38.5, 1000.0), None, args())
    assert not v["passed"]


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
