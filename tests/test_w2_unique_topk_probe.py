#!/usr/bin/env python3
"""CPU tests for the W2 unique-topk / TRF=32 decode-skip probe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_w2_unique_topk.py"
SPEC = importlib.util.spec_from_file_location("probe_w2_unique_topk", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

HELPERS = MODULE.load_helpers()
PINNED = (ROOT / "tests/fixtures/w2-grouped-topk-router.py").read_text()

NATIVE_ROUTER = '''
import torch

def grouped_topk(hidden_states, gating_output, topk, renormalize,
                 num_expert_group=0, topk_group=0, scoring_func="softmax",
                 routed_scaling_factor=1.0, e_score_correction_bias=None):
    topk_weights, topk_ids = torch.topk(gating_output, k=topk, dim=-1, sorted=True)
    return topk_weights, topk_ids
'''

COMMENT_ONLY = '''
def grouped_topk(hidden_states, gating_output, topk, renormalize):
    # torch.topk(gating_output, k=topk, dim=-1)
    return gating_output, gating_output
'''

UNUSED_UNIQUE = '''
def grouped_topk(hidden_states, gating_output, topk, renormalize):
    topk_weights, topk_ids = torch.topk(gating_output, k=topk, dim=-1)
    return topk_weights, topk_ids

class GroupedTopKRouter:
    def _compute_routing(self, hidden_states, router_logits, indices_type):
        ids = torch.multinomial(router_logits.softmax(-1), 8, replacement=True)
        return router_logits, ids
'''

REPLACEMENT = '''
def grouped_topk(hidden_states, gating_output, topk, renormalize):
    ids = torch.multinomial(gating_output.softmax(-1), topk, replacement=True)
    return gating_output, ids
'''

DISCARDED_TOPK = '''
def fused_grouped_topk(hidden_states, gating_output, topk, renormalize,
                       e_score_correction_bias, num_expert_group=0, topk_group=0,
                       scoring_func="softmax", routed_scaling_factor=1.0):
    ops.grouped_topk(gating_output, num_expert_group, topk_group, topk, renormalize)
    ids = gating_output.new_zeros(gating_output.size(0), topk)
    return gating_output, ids

def grouped_topk(hidden_states, gating_output, topk, renormalize,
                 num_expert_group=0, topk_group=0, scoring_func="softmax",
                 routed_scaling_factor=1.0, e_score_correction_bias=None):
    torch.topk(gating_output, k=topk, dim=-1)
    return gating_output, gating_output.new_zeros(gating_output.size(0), topk)
'''


def _mutate_compute_zero() -> str:
    needle = "        return topk_weights, topk_ids\n"
    idx = PINNED.rfind(needle)
    assert idx != -1
    return PINNED[:idx] + "        topk_ids.zero_()\n" + PINNED[idx:]


def test_c4_zipf_unique_ids_and_pinned_router_arm_ok() -> None:
    report = MODULE.judge(
        tokens=32,
        cap=32,
        ids=MODULE.zipf_unique_ids(32, 8),
        router_source=PINNED,
        helpers=HELPERS,
    )
    assert report["decision"] == "ARM-OK"
    assert report["unique_topk"] is True
    assert report["hottest"] == 32
    assert report["unique_skip"] is False
    assert "router_ast" in report["evidence"]
    assert "fingerprint match" in report["router_proof"]


def test_trf128_pinned_router_arm_ok() -> None:
    report = MODULE.judge(
        tokens=32,
        cap=128,
        ids=None,
        router_source=PINNED,
        helpers=HELPERS,
    )
    assert report["decision"] == "ARM-OK"


def test_pinned_live_router_dump_is_arm_ok() -> None:
    report = MODULE.judge(
        tokens=32,
        cap=32,
        ids=None,
        router_source=PINNED,
        helpers=HELPERS,
    )
    assert report["decision"] == "ARM-OK", report
    assert "fingerprint match" in report["router_proof"]


def test_simplified_native_router_does_not_match_fixture() -> None:
    report = MODULE.judge(
        tokens=32,
        cap=32,
        ids=None,
        router_source=NATIVE_ROUTER,
        helpers=HELPERS,
    )
    assert report["decision"] == "ABORT"
    assert "does not match" in report["reason"] or "missing reviewed" in report["reason"]


def test_non_unique_ids_abort_even_with_pinned_router() -> None:
    ids = [[0, 0, 1, 2, 3, 4, 5, 6] for _ in range(32)]
    report = MODULE.judge(
        tokens=32,
        cap=32,
        ids=ids,
        router_source=PINNED,
        helpers=HELPERS,
    )
    assert report["decision"] == "ABORT"
    assert report["unique_topk"] is False


def test_replacement_true_router_aborts() -> None:
    report = MODULE.judge(
        tokens=32,
        cap=32,
        ids=MODULE.zipf_unique_ids(32, 8),
        router_source=REPLACEMENT,
        helpers=HELPERS,
    )
    assert report["decision"] == "ABORT"


def test_missing_evidence_aborts() -> None:
    report = MODULE.judge(
        tokens=32,
        cap=32,
        ids=None,
        router_source=None,
        helpers=HELPERS,
    )
    assert report["decision"] == "ABORT"
    assert "not proven" in report["reason"]


def test_empty_ids_abort() -> None:
    report = MODULE.judge(
        tokens=32,
        cap=32,
        ids=[],
        router_source=PINNED,
        helpers=HELPERS,
    )
    assert report["decision"] == "ABORT"
    assert "empty" in report["reason"]


def test_wrong_shape_ids_abort() -> None:
    report = MODULE.judge(
        tokens=32,
        cap=32,
        ids=[[0, 1, 2, 3, 4, 5, 6, 7]],
        router_source=PINNED,
        helpers=HELPERS,
        topk=8,
    )
    assert report["decision"] == "ABORT"
    assert "rows" in report["reason"]


def test_comment_only_topk_aborts() -> None:
    report = MODULE.judge(
        tokens=32,
        cap=32,
        ids=None,
        router_source=COMMENT_ONLY,
        helpers=HELPERS,
    )
    assert report["decision"] == "ABORT"


def test_unused_unique_path_aborts() -> None:
    report = MODULE.judge(
        tokens=32,
        cap=32,
        ids=None,
        router_source=UNUSED_UNIQUE,
        helpers=HELPERS,
    )
    assert report["decision"] == "ABORT"


def test_discarded_unique_call_aborts() -> None:
    report = MODULE.judge(
        tokens=32,
        cap=32,
        ids=None,
        router_source=DISCARDED_TOPK,
        helpers=HELPERS,
    )
    assert report["decision"] == "ABORT"
    assert "does not match" in report["reason"] or "missing reviewed" in report["reason"]


def test_post_routing_id_mutation_aborts() -> None:
    report = MODULE.judge(
        tokens=32,
        cap=32,
        ids=None,
        router_source=_mutate_compute_zero(),
        helpers=HELPERS,
    )
    assert report["decision"] == "ABORT"
    assert "does not match" in report["reason"]


def test_cli_without_evidence_exits_nonzero() -> None:
    rc = MODULE.main([])
    assert rc == 1


def test_cli_synthetic_cannot_arm() -> None:
    rc = MODULE.main(["--synthetic-selftest"])
    assert rc == 1


def test_cli_pinned_router_exits_zero() -> None:
    rc = MODULE.main(
        ["--router-source", str(ROOT / "tests/fixtures/w2-grouped-topk-router.py")]
    )
    assert rc == 0


def test_cli_empty_ids_exits_nonzero(tmp_path: Path) -> None:
    ids_path = tmp_path / "ids.json"
    ids_path.write_text("[]")
    rc = MODULE.main(
        [
            "--router-source",
            str(ROOT / "tests/fixtures/w2-grouped-topk-router.py"),
            "--ids-json",
            str(ids_path),
        ]
    )
    assert rc == 1


def test_cli_unique_ids_without_router_can_arm(tmp_path: Path) -> None:
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps(MODULE.zipf_unique_ids(32, 8)))
    rc = MODULE.main(["--ids-json", str(ids_path)])
    assert rc == 0
