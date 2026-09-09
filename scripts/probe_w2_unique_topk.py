#!/usr/bin/env python3
"""CPU-only W2 unique-topk / TRF=32 decode-skip judge.

Decode cannot D2H hottest-count under CUDA graphs. Native grouped_topk /
torch.topk is unique per token, so hottest ≤ T. Arming TRF=32 requires
live evidence: a parseable router source whose selected CUDA path is
unique top-k, and/or a non-empty unique ids dump of the requested shape.

Exit 0 = ARM-OK for isolated EXL3_TEMP_ROWS_FUSED=32.
Exit 1 = ABORT (no-fallback skip, or uniqueness not proven).
No CUDA, no vLLM import. Synthetic Zipf rows are self-test only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "overlay" / "exl3.py"
REVIEWED_ROUTER_FIXTURE = ROOT / "tests/fixtures/w2-grouped-topk-router.py"
REVIEWED_ROUTER_FUNCTIONS = (
    "fused_grouped_topk",
    "grouped_topk",
    "GroupedTopKRouter._compute_routing",
)

UNIQUE_CALLS = {"topk", "grouped_topk"}
NON_UNIQUE_CALLS = {"multinomial"}


def load_helpers() -> dict[str, Any]:
    src = OVERLAY.read_text()
    tree = ast.parse(src)
    keep: list[str] = []
    wanted = {
        "fused_moe_decode_skips_fat",
        "fused_moe_decode_skips_unique_topk",
        "fused_temp_rows_decode_floor",
        "unique_per_token_topk_ids",
        "unique_topk_hottest_upper_bound",
        "w2_trf32_no_fallback_skip",
        "temp_rows_fused",
        "TEMP_ROWS_FUSED",
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            keep.append(ast.get_source_segment(src, node) or "")
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(name in wanted for name in names):
                keep.append(ast.get_source_segment(src, node) or "")
    ns: dict[str, Any] = {}
    exec("from __future__ import annotations\n" + "\n\n".join(keep), ns, ns)
    return ns


def zipf_unique_ids(tokens: int, topk: int) -> list[list[int]]:
    """Every token picks the same first expert, remaining k-1 unique."""
    return [[0] + list(range(1, topk)) for _ in range(tokens)]


def hottest_from_ids(ids: list[list[int]]) -> int:
    counts: dict[int, int] = {}
    for row in ids:
        seen: set[int] = set()
        for expert in row:
            expert_id = int(expert)
            if expert_id < 0 or expert_id in seen:
                continue
            seen.add(expert_id)
            counts[expert_id] = counts.get(expert_id, 0) + 1
    return max(counts.values(), default=0)


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _call_has_replacement_true(node: ast.Call) -> bool:
    for kw in node.keywords:
        if kw.arg != "replacement":
            continue
        return isinstance(kw.value, ast.Constant) and kw.value.value is True
    return False


def _walk_functions(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    found: dict[str, ast.FunctionDef] = {}
    for node in tree.body if hasattr(tree, "body") else []:
        if isinstance(node, ast.FunctionDef):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    found[f"{node.name}.{item.name}"] = item
    return found


def _calls_in(fn: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(fn) if isinstance(node, ast.Call)]


def _assigns_name_from(fn: ast.AST, target: str, source: str) -> bool:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if target not in names:
            continue
        value = node.value
        if isinstance(value, ast.Name) and value.id == source:
            return True
        if isinstance(value, ast.Call):
            for kw in value.keywords:
                if isinstance(kw.value, ast.Name) and kw.value.id == source:
                    return True
            for arg in value.args:
                if isinstance(arg, ast.Name) and arg.id == source:
                    return True
    return False


def _function_fingerprint(
    fns: dict[str, ast.FunctionDef], names: tuple[str, ...]
) -> str:
    if any(name not in fns for name in names):
        return ""
    payload = "\n".join(
        ast.dump(fns[name], include_attributes=False) for name in names
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def reviewed_router_fingerprint() -> str:
    if not REVIEWED_ROUTER_FIXTURE.is_file():
        return ""
    tree = ast.parse(REVIEWED_ROUTER_FIXTURE.read_text())
    return _function_fingerprint(_walk_functions(tree), REVIEWED_ROUTER_FUNCTIONS)


def router_unique_proof(source: str) -> tuple[bool, str]:
    """Accept only the reviewed production grouped_topk AST.

    Call presence is not enough: discarded ``torch.topk`` results or a
    post-routing ``topk_ids.zero_()`` would still mention unique APIs.
    The pin is the normalized AST of fused_grouped_topk, grouped_topk, and
    GroupedTopKRouter._compute_routing from
    ``tests/fixtures/w2-grouped-topk-router.py``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"router source is not parseable Python: {exc.msg}"

    fns = _walk_functions(tree)
    expected = reviewed_router_fingerprint()
    if not expected:
        return False, "reviewed grouped_topk fixture is missing or unreadable"
    actual = _function_fingerprint(fns, REVIEWED_ROUTER_FUNCTIONS)
    if not actual:
        return False, (
            "router source missing reviewed fused_grouped_topk / grouped_topk / "
            "GroupedTopKRouter._compute_routing"
        )
    if actual != expected:
        return False, (
            "router AST does not match the reviewed production grouped_topk fixture"
        )

    grouped = fns["grouped_topk"]
    fused = fns["fused_grouped_topk"]
    compute = fns["GroupedTopKRouter._compute_routing"]
    for fn in (grouped, fused, compute):
        for call in _calls_in(fn):
            if _call_has_replacement_true(call):
                return False, "selected router path calls a sampler with replacement=True"
            if _call_name(call) in NON_UNIQUE_CALLS:
                return False, f"selected router path calls non-unique {_call_name(call)}()"
    return True, (
        "AST unique CUDA path: fingerprint match vs "
        "tests/fixtures/w2-grouped-topk-router.py "
        "(grouped_topk -> torch.topk; fused_grouped_topk -> ops.grouped_topk; "
        "GroupedTopKRouter._compute_routing -> grouped_topk)"
    )


def _normalize_ids(ids: Any, tokens: int, topk: int) -> tuple[list[list[int]] | None, str]:
    if ids is None:
        return None, ""
    if not isinstance(ids, list) or len(ids) == 0:
        return None, "ids dump is empty"
    rows: list[list[int]] = []
    for row in ids:
        if not isinstance(row, (list, tuple)):
            return None, "ids dump rows must be sequences"
        rows.append([int(v) for v in row])
    if len(rows) != tokens:
        return None, f"ids dump has {len(rows)} rows, expected tokens={tokens}"
    for row in rows:
        live = [v for v in row if v >= 0]
        if len(live) != topk:
            return None, f"ids dump row width {len(live)} != topk={topk}"
    return rows, ""


def judge(
    *,
    tokens: int,
    cap: int,
    ids: list[list[int]] | None,
    router_source: str | None,
    helpers: dict[str, Any],
    topk: int = 8,
) -> dict[str, Any]:
    unique_fn = helpers["unique_per_token_topk_ids"]
    unique_skip = helpers["fused_moe_decode_skips_unique_topk"]
    skip = helpers["fused_moe_decode_skips_fat"]
    gate = helpers["w2_trf32_no_fallback_skip"]
    floor = helpers["fused_temp_rows_decode_floor"]
    bound = helpers["unique_topk_hottest_upper_bound"]

    report: dict[str, Any] = {
        "tokens": tokens,
        "cap": cap,
        "topk": topk,
        "floor_c4": floor(4, 7),
        "unique_topk": None,
        "hottest": None,
        "unique_skip": unique_skip(tokens, cap),
        "w2_gate_unique": gate(tokens, unique_topk=True, cap=cap),
        "router_proof": None,
        "evidence": [],
        "decision": "ABORT",
        "reason": "",
    }
    if report["floor_c4"] != 32:
        report["reason"] = f"C4 floor is {report['floor_c4']}, expected 32"
        return report
    if tokens <= cap and unique_skip(tokens, cap):
        report["reason"] = "unique-topk decode skip True (hottest bound > cap)"
        return report
    if gate(tokens, unique_topk=True, cap=cap):
        report["reason"] = "w2_trf32_no_fallback_skip unique=True"
        return report
    if cap < 32 and tokens == 32:
        report["reason"] = "cap below C4 floor"
        return report

    proven = False
    if router_source is not None:
        ok, msg = router_unique_proof(router_source)
        report["router_proof"] = msg
        if not ok:
            report["reason"] = msg
            return report
        report["evidence"].append("router_ast")
        proven = True
    elif router_source == "":
        report["reason"] = "router source is empty"
        return report

    rows, id_err = _normalize_ids(ids, tokens, topk)
    if id_err:
        report["reason"] = id_err
        return report
    if ids is not None and rows is None:
        report["reason"] = "ids dump is empty"
        return report
    if rows is not None:
        unique = bool(unique_fn(rows))
        hottest = hottest_from_ids(rows)
        report["unique_topk"] = unique
        report["hottest"] = hottest
        report["evidence"].append("ids_dump")
        if not unique:
            report["reason"] = "dumped top-k ids are not unique per token"
            report["w2_gate_live"] = skip(len(rows), hottest, cap)
            return report
        if hottest > bound(len(rows)):
            report["reason"] = f"hottest {hottest} exceeds unique bound {bound(len(rows))}"
            return report
        if skip(len(rows), hottest, cap):
            report["reason"] = "live hottest > cap with T ≤ cap"
            return report
        proven = True

    if not proven:
        report["reason"] = (
            "uniqueness not proven: need parseable unique router source "
            "and/or a non-empty unique ids dump"
        )
        return report

    report["decision"] = "ARM-OK"
    report["reason"] = (
        "unique-per-token top-k; hottest ≤ T; fused skip is >; "
        f"C4 T={tokens} cap={cap} does not no-fallback-skip; "
        f"evidence={','.join(report['evidence'])}"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--cap", type=int, default=32)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--ids-json", type=Path, default=None)
    ap.add_argument("--router-source", type=Path, default=None)
    ap.add_argument(
        "--synthetic-selftest",
        action="store_true",
        help="Build Zipf-unique ids for helper tests. Not live evidence; cannot ARM.",
    )
    args = ap.parse_args(argv)

    helpers = load_helpers()
    ids = None
    synthetic = False
    if args.ids_json is not None:
        ids = json.loads(args.ids_json.read_text() or "null")
    elif args.synthetic_selftest:
        ids = zipf_unique_ids(args.tokens, args.topk)
        synthetic = True
    router_source = None
    if args.router_source is not None:
        router_source = args.router_source.read_text()

    report = judge(
        tokens=args.tokens,
        cap=args.cap,
        ids=ids,
        router_source=router_source,
        helpers=helpers,
        topk=args.topk,
    )
    if synthetic:
        report["synthetic_selftest"] = True
        if report["decision"] == "ARM-OK":
            report["decision"] = "SELFTEST-OK"
            report["reason"] = (
                "synthetic Zipf rows only; not live evidence — "
                + report["reason"]
            )
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if report["decision"] == "ARM-OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
