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


def _in_body(statements: list[ast.stmt]) -> ast.Module:
    """Wrap a statement list so ``ast.walk`` sees only those statements.

    The whole point of the dead-branch check is that ``if False: ...`` is
    unreachable. Walking the enclosing ``ast.If`` instead would also traverse
    ``orelse``, which is *live* — and a live ``else: persistent_topk()``
    misread as dead turns a REACHABLE audit into a false NOT_APPLICABLE.
    """
    return ast.Module(body=statements, type_ignores=[])


def _is_literal_false_guard(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.BoolOp)
        and isinstance(test.op, ast.And)
        and bool(test.values)
        and _is_literal_false(test.values[0])
    )


def dead_persistent_topk_branches(source: str, label: str) -> list[dict]:
    """``if False and ...`` blocks whose *body* calls persistent_topk."""
    tree = _parse(source, label)
    found: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not _is_literal_false_guard(node.test):
            continue
        hits = _calls_named(_in_body(node.body), PERSISTENT_TOPK)
        if not hits:
            continue
        found.append(
            {
                "line": node.lineno,
                "persistent_topk_calls": hits,
                "else_branch": bool(node.orelse),
                "live_kernel_calls": _calls_named(
                    _in_body(node.orelse), LIVE_KERNEL
                ),
            }
        )
    return found


def live_persistent_topk(source: str, label: str) -> list[int]:
    """persistent_topk calls that are NOT inside a literal-False *body*."""
    tree = _parse(source, label)
    dead_lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if _is_literal_false_guard(node.test):
            dead_lines.update(_calls_named(_in_body(node.body), PERSISTENT_TOPK))
    return [line for line in _calls_named(tree, PERSISTENT_TOPK) if line not in dead_lines]


KPOOL_MODULE_TAIL = KPOOL_REL.rsplit("/", 1)[-1].removesuffix(".py")
PLAIN_MODULE_TAIL = PLAIN_INDEXER_REL.rsplit("/", 1)[-1].removesuffix(".py")


def _import_bindings(tree: ast.Module) -> dict[str, set[tuple[str, str]]]:
    """Local name -> every (originating module, original name) it is bound to.

    Every binding is kept, not just the last one: a function-local import must
    not silently overwrite a module-level binding of the same name, because the
    two are visible in different scopes.
    """
    bindings: dict[str, set[tuple[str, str]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                bindings.setdefault(alias.asname or alias.name, set()).add(
                    (module, alias.name)
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[-1]
                bindings.setdefault(local, set()).add(
                    (alias.name, alias.name.split(".")[-1])
                )
    return bindings


def _resolve_indexer_class(
    local: str, bindings: dict[str, set[tuple[str, str]]]
) -> str | None:
    """Which indexer class a *name* denotes, by import provenance.

    Identity must come from where the name came from, not from how it is
    spelled: ``from ...sparse_attn_indexer import SparseAttnIndexer as
    SparseAttnIndexerKpool`` constructs the *plain* indexer, whose
    ``persistent_topk`` is live, while reading exactly like the kpool class.

    A name imported from a module that is neither indexer module resolves to
    nothing, the exported symbol must actually be an indexer class, and a name
    bound to more than one distinct indexer class is ambiguous and also resolves
    to nothing. Callers treat "nothing" as unresolved rather than guessing.
    """
    entries = bindings.get(local)
    if not entries:
        # No import binding: a class defined in this module under that name.
        return local if local in (KPOOL_CLASS, PLAIN_CLASS) else None
    resolved = set()
    for module, original in entries:
        tail = module.rsplit(".", 1)[-1] if module else ""
        if tail == KPOOL_MODULE_TAIL and original == KPOOL_CLASS:
            resolved.add(KPOOL_CLASS)
        elif tail == PLAIN_MODULE_TAIL and original == PLAIN_CLASS:
            resolved.add(PLAIN_CLASS)
    if len(resolved) != 1:
        return None
    return resolved.pop()


def _mentions_an_indexer(entries: set[tuple[str, str]]) -> bool:
    for module, original in entries:
        tail = module.rsplit(".", 1)[-1] if module else ""
        if tail in (KPOOL_MODULE_TAIL, PLAIN_MODULE_TAIL):
            return True
        if original in (KPOOL_CLASS, PLAIN_CLASS):
            return True
    return False


def _ambiguous_indexer_names(bindings: dict[str, set[tuple[str, str]]]) -> list[str]:
    """Imported names that look like an indexer but do not resolve uniquely.

    Covers a name bound to two different indexer classes in different scopes (a
    function-local import shadowing a module-level one) and a name imported from
    an indexer module that does not export that class.
    """
    return sorted(
        local
        for local, entries in bindings.items()
        if _mentions_an_indexer(entries) and _resolve_indexer_class(local, bindings) is None
    )


def glm_uses_kpool(glm_source: str, label: str = "GLM attention module") -> dict:
    """Which indexer the GLM module *actually* constructs.

    Parsed, not pattern-matched, and resolved through import provenance rather
    than by spelling: a textual occurrence in a comment, docstring or dead code
    must not stand in for a real construction, the plain ``SparseAttnIndexer``
    (whose ``persistent_topk`` *is* live) is a prefix of the kpool name, and an
    aliased import can make either name denote the other class.
    """
    tree = _parse(glm_source, label)
    bindings = _import_bindings(tree)

    imported: set[str] = set(bindings)

    referenced_classes: set[str] = set()
    constructed_classes: set[str] = set()
    unresolved_references: set[str] = set(_ambiguous_indexer_names(bindings))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                resolved = _resolve_indexer_class(func.id, bindings)
                if resolved is None and func.id in (KPOOL_CLASS, PLAIN_CLASS):
                    unresolved_references.add(func.id)
                elif resolved:
                    constructed_classes.add(resolved)
            elif isinstance(func, ast.Attribute):
                if func.attr in (KPOOL_CLASS, PLAIN_CLASS):
                    constructed_classes.add(func.attr)
        elif isinstance(node, ast.Name):
            resolved = _resolve_indexer_class(node.id, bindings)
            if resolved:
                referenced_classes.add(resolved)
            elif node.id in (KPOOL_CLASS, PLAIN_CLASS):
                unresolved_references.add(node.id)
        elif isinstance(node, ast.Attribute):
            if node.attr in (KPOOL_CLASS, PLAIN_CLASS):
                referenced_classes.add(node.attr)

    kpool_lines = sorted(
        {
            node.lineno
            for node in ast.walk(tree)
            if (
                (isinstance(node, ast.Name) and node.id == KPOOL_CLASS)
                or (isinstance(node, ast.Attribute) and node.attr == KPOOL_CLASS)
            )
        }
    )
    return {
        "imports_kpool_class": KPOOL_CLASS in imported or KPOOL_CLASS in referenced_classes,
        "instantiates_kpool_class": KPOOL_CLASS in constructed_classes,
        "kpool_reference_lines": kpool_lines,
        "uses_plain_indexer": PLAIN_CLASS in referenced_classes,
        "plain_indexer_constructed": PLAIN_CLASS in constructed_classes,
        "unresolved_indexer_names": sorted(unresolved_references),
        "import_bindings": {k: sorted(list(e) for e in v) for k, v in sorted(bindings.items())},
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
    glm = glm_uses_kpool(glm_src, GLM_ATTENTION_REL)
    if not glm["instantiates_kpool_class"]:
        raise Abort(
            "the GLM-5.3-Flash model path does not construct "
            f"{KPOOL_CLASS} (reference lines: {glm['kpool_reference_lines']}); "
            "the indexer routing this audit reasons about has changed — "
            "re-derive it before trusting any verdict"
        )
    if glm["uses_plain_indexer"]:
        raise Abort(
            f"the GLM-5.3-Flash model path also references {PLAIN_CLASS}, whose "
            "persistent_topk is live; routing is unresolved, so the kpool "
            "verdict cannot be claimed"
        )
    if glm["unresolved_indexer_names"]:
        raise Abort(
            "the GLM-5.3-Flash model path uses indexer name(s) "
            f"{glm['unresolved_indexer_names']} whose import provenance does not "
            "resolve to a known indexer module; routing is unresolved"
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
