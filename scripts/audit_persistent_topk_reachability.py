#!/usr/bin/env python3
"""Fail-closed reachability audit for the persistent-top-k overflow lane (task 39).

vLLM #52149 and #55314 fix a *correctness* defect in ``persistent_topk``: when
the radix threshold bin overflows the candidate buffer, the indexer selects the
wrong top-k pools. #55314 discusses GLM-5.3-Flash indexer logits directly, so
the parked watch item assumed GB10 runs that kernel.

This audit decides applicability from the deployed source rather than from the
watch item's premise. The GLM-5.3-Flash model path uses
``SparseAttnIndexerKpool``; in the deployed build that file's ``persistent_topk``
branch is guarded by a literal ``if False and ...``, installed fail-closed by
``overlay/patch_glm_video_placeholders.py::_disable_gb10_persistent_topk``
("Decode-path persistent_topk oversubscribes GB10 smem on long seqs"). The live
decode kernel is ``top_k_per_row_decode`` instead.

Verdicts:
  NOT_APPLICABLE — the GLM path cannot reach ``persistent_topk``; #52149/#55314
                   have no mechanism here. No logits comparison is owed.
  REACHABLE      — the dead branch is gone (or the GLM path no longer uses the
                   kpool indexer). The overflow lane applies and must be
                   assessed against #55122's post-patch code.
  ABORT          — a required source or anchor is missing/unreadable. Never
                   reports a pass it did not establish.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

GLM_ATTENTION_REL = "models/glm5next/nvidia/attention.py"
KPOOL_REL = "model_executor/layers/sparse_attn_indexer_kpool.py"
PLAIN_INDEXER_REL = "model_executor/layers/sparse_attn_indexer.py"

KPOOL_CLASS = "SparseAttnIndexerKpool"
PLAIN_CLASS = "SparseAttnIndexer"
PERSISTENT_TOPK = "persistent_topk"
LIVE_KERNEL = "top_k_per_row_decode"
OVERLAY_NAME = "patch_glm_video_placeholders.py"
OVERLAY_DEAD_BRANCH = "if False and current_platform.is_cuda() and "
OVERLAY_GUARD_CALL = "_disable_gb10_persistent_topk"
OVERLAY_FAIL_CLOSED = "kpool persistent_topk pattern not found"


class Abort(RuntimeError):
    """A required source or anchor was missing; the audit cannot decide."""


def _read(path: Path, label: str) -> str:
    if not path.is_file():
        raise Abort(f"missing {label}: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - unreadable file
        raise Abort(f"unreadable {label}: {path}: {exc}") from exc


def _parse(source: str, label: str) -> ast.Module:
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise Abort(f"cannot parse {label}: {exc}") from exc


def _is_literal_false(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _calls_named(node: ast.AST, attr: str) -> list[int]:
    """Line numbers of calls whose dotted callee ends in ``attr``."""
    hits: list[int] = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        name = None
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        if name == attr:
            hits.append(sub.lineno)
    return hits


def dead_persistent_topk_branches(source: str, label: str) -> list[dict]:
    """``if False and ...`` blocks whose body calls persistent_topk."""
    tree = _parse(source, label)
    found: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.BoolOp) or not isinstance(test.op, ast.And):
            continue
        if not test.values or not _is_literal_false(test.values[0]):
            continue
        hits = _calls_named(node, PERSISTENT_TOPK)
        if not hits:
            continue
        found.append(
            {
                "line": node.lineno,
                "persistent_topk_calls": hits,
                "else_branch": bool(node.orelse),
                "live_kernel_calls": _calls_named(
                    ast.Module(body=node.orelse, type_ignores=[]), LIVE_KERNEL
                ),
            }
        )
    return found


def live_persistent_topk(source: str, label: str) -> list[int]:
    """persistent_topk calls that are NOT inside a literal-False branch."""
    tree = _parse(source, label)
    dead_lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.BoolOp)
            and isinstance(test.op, ast.And)
            and test.values
            and _is_literal_false(test.values[0])
        ):
            dead_lines.update(_calls_named(node, PERSISTENT_TOPK))
    return [line for line in _calls_named(tree, PERSISTENT_TOPK) if line not in dead_lines]


def glm_uses_kpool(glm_source: str) -> dict:
    # `SparseAttnIndexer` is a prefix of `SparseAttnIndexerKpool`, so a plain
    # substring test would report the GLM module as a user of both indexers.
    plain_re = re.compile(rf"\b{PLAIN_CLASS}\b(?!Kpool)")
    return {
        "imports_kpool_class": bool(re.search(rf"\b{KPOOL_CLASS}\b", glm_source)),
        "instantiates_kpool_class": f"{KPOOL_CLASS}(" in glm_source,
        "uses_plain_indexer": bool(plain_re.search(glm_source)),
    }


def plain_indexer_users(site: Path) -> list[str]:
    """Modules that use the non-kpool indexer (whose persistent_topk is live)."""
    models = site / "models"
    if not models.is_dir():
        raise Abort(f"missing models dir under vLLM site: {models}")
    plain_re = re.compile(rf"\b{PLAIN_CLASS}\b(?!Kpool)")
    kpool_re = re.compile(rf"\b{KPOOL_CLASS}\b")
    users: list[str] = []
    for path in sorted(models.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - unreadable file
            continue
        if plain_re.search(text) and not kpool_re.search(text):
            users.append(str(path.relative_to(site)))
    return users


def overlay_guard(overlay_dir: Path | None) -> dict:
    if overlay_dir is None:
        return {"checked": False, "present": False}
    if not overlay_dir.is_dir():
        raise Abort(f"missing overlay dir: {overlay_dir}")
    path = overlay_dir / OVERLAY_NAME
    source = _read(path, "persistent-topk disabling overlay")
    has_dead = OVERLAY_DEAD_BRANCH in source
    has_guard = OVERLAY_GUARD_CALL in source
    fail_closed = OVERLAY_FAIL_CLOSED in source
    if not has_dead or not has_guard or not fail_closed:
        raise Abort(
            f"{OVERLAY_NAME} no longer installs the fail-closed GB10 "
            f"persistent_topk guard (dead-branch anchor={has_dead}, "
            f"guard call={has_guard}, fail-closed refusal={fail_closed})"
        )
    return {
        "checked": True,
        "present": True,
        "installs_dead_branch": has_dead,
        "guard_call": has_guard,
        "fail_closed": fail_closed,
    }


def audit(site: Path, overlay_dir: Path | None, kernel_dir: Path | None) -> dict:
    glm_src = _read(site / GLM_ATTENTION_REL, "GLM attention module")
    kpool_src = _read(site / KPOOL_REL, "kpool sparse indexer")
    glm = glm_uses_kpool(glm_src)
    if not glm["imports_kpool_class"]:
        raise Abort(
            "the GLM-5.3-Flash model path no longer imports SparseAttnIndexerKpool; "
            "re-derive which indexer it uses before trusting this audit"
        )
    dead = dead_persistent_topk_branches(kpool_src, "sparse_attn_indexer_kpool.py")
    live = live_persistent_topk(kpool_src, "sparse_attn_indexer_kpool.py")
    guard = overlay_guard(overlay_dir)
    plain_users = plain_indexer_users(site)

    fixed_scratch = None
    if kernel_dir is not None:
        sampler = _read(kernel_dir / "sampler.cu", "top_k_per_row_decode launcher")
        if LIVE_KERNEL not in sampler:
            raise Abort(f"{LIVE_KERNEL} not found in the sampler source")
        fixed_scratch = "MAX_INDICES" in sampler or "candidate buffer" in sampler

    if dead and not live:
        verdict = "NOT_APPLICABLE"
        reason = (
            "the GLM-5.3-Flash path uses SparseAttnIndexerKpool and that file's "
            f"persistent_topk branch is dead by construction (line {dead[0]['line']}), "
            f"with {LIVE_KERNEL} on the live else branch. #52149/#55314 cannot "
            "change selection here, so no indexer-logits comparison is owed."
        )
    elif live:
        verdict = "REACHABLE"
        reason = (
            "persistent_topk is live on the GLM indexer path at line(s) "
            f"{live}; assess #52149/#55314 against #55122's post-patch code."
        )
    else:
        verdict = "ABORT"
        reason = (
            "no persistent_topk call was found in the kpool indexer at all; the "
            "source shape this audit reasons about has changed."
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "glm_indexer": glm,
        "dead_branches": dead,
        "live_persistent_topk_lines": live,
        "overlay_guard": guard,
        "plain_indexer_users": plain_users,
        "live_kernel": LIVE_KERNEL,
        "live_kernel_has_fixed_scratch": fixed_scratch,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-site",
        type=Path,
        required=True,
        help="installed vLLM package root (the dir containing models/ and model_executor/)",
    )
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        help="vLLM csrc/libtorch_stable dir, for the live-kernel scratch check",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit(args.vllm_site, args.overlay_dir, args.kernel_dir)
    except Abort as exc:
        print(json.dumps({"verdict": "ABORT", "reason": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["verdict"] == "ABORT":
        return 1
    return 2 if report["verdict"] == "REACHABLE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
