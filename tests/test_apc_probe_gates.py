#!/usr/bin/env python3
"""Negative tests for the correctness gate in `scripts/probe_apc_tail_correctness.py`.

The probe is the correctness gate for the `[glm53-apc-tail-floor]` overlay, so a
gate that cannot fail is worse than no gate. This file drives the probe's
verdict logic directly, with synthetic case records, and asserts that each
failure mode it claims to catch really does fail:

  * zero reuse on a shape that must reuse (the exact defect the overlay fixes),
  * reuse that overshoots the reachable boundary,
  * a stale leak from superseded state,
  * a wrong answer with correct reuse,
  * a query delta that does not account for the prompt,
  * a cache reset that did not happen,

and that a fully consistent run passes. `tests/test_apc_tail_boundary.py` covers
the overlay's own behaviour; this file covers the instrument that qualifies it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_apc_tail_correctness.py"

UNIT = 64


def _load_probe():
    spec = importlib.util.spec_from_file_location("probe_apc_tail_correctness", PROBE)
    assert spec and spec.loader, PROBE
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe():
    return _load_probe()


def _case(label, hits, prompt_tokens, expected_hits, *, answer_ok=True,
          answer_checked=True, stale_leak=False, queried_tokens=None,
          reset_ok=True, reset_status="ok",
          expect="CORMORANT-8815", stale_probe=None):
    return {
        "case": label,
        "hits": hits,
        "prompt_tokens": prompt_tokens,
        "queried_tokens": prompt_tokens if queried_tokens is None else queried_tokens,
        "expected_hits": expected_hits,
        "ceiling_tokens": prompt_tokens - UNIT,
        "answer_ok": answer_ok,
        "answer_checked": answer_checked,
        "stale_leak": stale_leak,
        "reset_ok": reset_ok,
        "reset_status": reset_status,
        "expect": expect,
        "stale_probe": stale_probe,
    }


def _healthy_run():
    """The armed-boot shape: aligned prompt, ceiling reuse, honest appends."""
    n = 26240
    return [
        _case("cold", 0, n, 0),
        _case("exact_replay", n - UNIT, n, n - UNIT),
        # The accepted 64-token cost on the append shape: the producer's tail is
        # at n - UNIT while the consumer's own ceiling is n. The answer is not
        # checked because the extension lands after the question.
        _case("append_continuation", n - UNIT, n + 32, n - UNIT,
              answer_ok=False, answer_checked=False),
        _case("changed_needle", 12224, n, 12224, expect="PETREL-2277",
              stale_probe="CORMORANT-8815"),
    ]


# --- the gate passes only on a fully consistent run -------------------------

def test_healthy_run_passes(probe):
    ok, failures = probe.evaluate(_healthy_run())
    assert ok, failures
    assert failures == []


# --- and fails on every failure mode it claims to catch ---------------------

def test_zero_reuse_on_exact_replay_fails(probe):
    """The defect itself: stock registers the tail at n, out of reach, so the
    replay falls a whole page short. An answer-only gate would pass this."""
    cases = _healthy_run()
    cases[1] = _case("exact_replay", 0, 26240, 26240 - UNIT)
    ok, failures = probe.evaluate(cases)
    assert not ok
    assert any("exact_replay" in f and "reused 0 tokens" in f for f in failures)


def test_page_short_reuse_fails(probe):
    """The measured baseline: 3,520 tokens short of the ceiling."""
    cases = _healthy_run()
    cases[1] = _case("exact_replay", 3584, 7168, 7168 - UNIT)
    ok, failures = probe.evaluate(cases)
    assert not ok
    assert any("exact_replay" in f for f in failures)


def test_overshooting_the_boundary_fails(probe):
    """Reusing more than the reachable ceiling means the registration is not
    where the key says it is. That must fail, not pass."""
    cases = _healthy_run()
    cases[1] = _case("exact_replay", 26240, 26240, 26240 - UNIT)
    ok, failures = probe.evaluate(cases)
    assert not ok
    assert any("exact_replay" in f for f in failures)


def test_stale_leak_fails_even_with_correct_reuse(probe):
    cases = _healthy_run()
    cases[3] = _case("changed_needle", 12224, 26240, 12224, expect="PETREL-2277",
                     stale_leak=True, stale_probe="CORMORANT-8815")
    ok, failures = probe.evaluate(cases)
    assert not ok
    assert any("STALE LEAK" in f for f in failures)


def test_wrong_answer_fails_even_with_correct_reuse(probe):
    cases = _healthy_run()
    cases[1] = _case("exact_replay", 26240 - UNIT, 26240, 26240 - UNIT, answer_ok=False)
    ok, failures = probe.evaluate(cases)
    assert not ok
    assert any("expected" in f for f in failures)


def test_unchecked_answer_is_not_silently_asserted(probe):
    """The append-continuation shape lands after the question, so its answer is
    not well-formed and must be excluded from the answer check -- explicitly,
    not by accident. Its boundary assertion still applies."""
    cases = _healthy_run()
    cases[2] = _case("append_continuation", 26240 - UNIT, 26272, 26240 - UNIT,
                     answer_ok=False, answer_checked=False)
    ok, failures = probe.evaluate(cases)
    assert ok, failures

    # ...but an unchecked answer must not excuse a wrong boundary.
    cases[2] = _case("append_continuation", 0, 26272, 26240 - UNIT,
                     answer_ok=False, answer_checked=False)
    ok, failures = probe.evaluate(cases)
    assert not ok
    assert any("append_continuation" in f for f in failures)


def test_inconsistent_query_delta_fails(probe):
    """If the query counter does not account for the prompt, the counter read
    is not trustworthy and every hit number derived from it is suspect."""
    cases = _healthy_run()
    cases[1] = _case("exact_replay", 26240 - UNIT, 26240, 26240 - UNIT,
                     queried_tokens=0)
    ok, failures = probe.evaluate(cases)
    assert not ok
    assert any("does not account for" in f for f in failures)


def test_failed_reset_fails(probe):
    """A reset that silently failed leaves earlier state in the cache, so a
    'cold' reading is not cold and the deltas are meaningless."""
    cases = _healthy_run()
    cases[0] = _case("cold", 0, 26240, 0, reset_ok=False,
                     reset_status="err:HTTPError")
    ok, failures = probe.evaluate(cases)
    assert not ok
    assert any("cache reset failed" in f for f in failures)


# --- the arithmetic the expectations rest on --------------------------------

def test_expected_boundary_floors_to_the_hash_grid(probe):
    assert probe.expected_boundary(10000, 10000, 64) == 9984
    assert probe.expected_boundary(10000, 10000, 64) == (10000 - 1) // 64 * 64


def test_expected_boundary_caps_at_the_finder_limit(probe):
    """A prompt that shares everything still cannot exceed (n-1)//unit*unit."""
    assert probe.expected_boundary(7168, 7168, 64) == 7104


def test_stock_registration_at_n_is_out_of_reach(probe):
    """Pins the defect: for a hash-grid-aligned prompt, a registration at `n` is
    one unit above the deepest position the finder may request."""
    n = 7168
    reachable = probe.expected_boundary(n, n, UNIT)
    assert reachable == n - UNIT
    assert n > reachable


def test_shared_prefix_len_stops_at_the_first_difference(probe):
    assert probe.shared_prefix_len([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert probe.shared_prefix_len([1, 2, 3], [1, 2, 3]) == 3
    assert probe.shared_prefix_len([1, 2, 3], [9, 2, 3]) == 0
    assert probe.shared_prefix_len([], [1]) == 0


def test_probe_verdict_function_is_network_free(probe):
    """`evaluate` must be importable and callable without a server: the gate's
    own behaviour is what these tests pin, not the cluster's."""
    assert callable(probe.evaluate)
    assert probe.evaluate([]) == (True, [])
