#!/usr/bin/env python3
"""Fail-closed audit of the unenforced 128-slot ``v_indices`` scratch (task 37).

The pinned ExLlamaV3 extension declares

    #define MAX_INDICES 128
    __device__ int64_t v_indices[128];
    __device__ half    v_weights[128];

and ``exl3_mgemm_kernel`` writes them with a bound of ``bszm`` — the slot count
``MAX(bszm_in, bszm_out)``, i.e. ``batch x top_k`` for the ``num_tokens == 1``
compaction path — never against ``MAX_INDICES``. That is the class of defect
vLLM's own #290 proposes to fix.

This audit does not assume the "always -1" comment in §20/C1 is true. It
enumerates the call sites and decides reachability from the *deployed* serving
path, which is:

    vLLM EXL3 overlay -> LinearEXL3.forward -> BC_LinearEXL3::run_gr
                      -> exl3_gemm_gr / exl3_gemm   (NOT exl3_mgemm)

Verdicts:
  NOT_REACHABLE — no ``exl3_mgemm`` call site is in the import closure of the
                  serving path; the scratch cannot be written at any shape.
  REACHABLE_OK  — reachable call sites exist but their worst-case ``bszm`` at
                  the declared production shapes stays <= MAX_INDICES.
  REACHABLE_OVERFLOW — a reachable call site can exceed MAX_INDICES. This is a
                  silent memory-corruption class defect and outranks every
                  performance task.
  ABORT         — a required source or anchor is missing/unreadable. Never
                  reports a pass it did not establish.

Basis and limits of a NOT_REACHABLE verdict — it is *static*, not observed:

  * The verdict rests on ``overlay/exl3_namespace.py`` installing
    ``exllamav3``, ``exllamav3.modules``, ``exllamav3.model`` and
    ``exllamav3.model.config`` as synthetic module objects before the serving
    module is imported. ``stub_contract`` proves that structurally *and* by
    executing the installer and inspecting the mapping, and
    ``_installer_invocation`` proves the loader calls it before its first
    ``exllamav3`` import. If the deployment stops stubbing any of them, the
    closure grows and the verdict reverts to ABORT or worse -- fail-closed.
  * ``exllamav3.model.config`` is the load-bearing one: the serving module's
    first import is ``from ...model.config import Config``, and the real
    ``model/config.py`` reaches ``architecture/architectures.py`` (which imports
    every architecture, including those pulling in ``modules.attn``,
    ``modules.dsv4`` and ``modules.gated_delta_net``) from a function-local
    import inside ``Config``. Stubbing the namespace is what keeps that fan-out
    out of the runtime closure.
  * Not modelled: importing ``exllamav3.modules.quant.exl3`` really does execute
    the real ``modules/quant/__init__.py`` (its parents ``exllamav3`` and
    ``exllamav3.modules`` are synthetic), because ``modules.quant`` is not a
    stub. On the pinned revision that file imports only ``.fp16`` and ``.exl3``,
    neither of which reaches ``architecture/``, so it does not change this
    verdict -- but it is a gap in the model, not a proof about it.
  * No live ``sys.modules`` observation backs any of this. ``__pycache__``
    mtimes cannot substitute: the image precompiles the whole tree at build
    time. If a static basis is ever judged insufficient, the answer is a
    one-time check in a stopped window, not a stronger claim from this script.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import types
from pathlib import Path

PACKAGE_DIR = "exllamav3"
KERNEL_REL = "exllamav3_ext/quant/exl3_gemm_kernel.cuh"
GEMM_REL = "exllamav3_ext/quant/exl3_gemm.cu"
LINEAR_REL = "exllamav3_ext/libtorch/linear.cpp"
BINDINGS_REL = "exllamav3_ext/bindings.cpp"
SERVING_PY_REL = "modules/quant/exl3.py"

# ``overlay/exl3_namespace.py::inject_config_stub`` installs these as synthetic
# ``types.ModuleType`` namespaces before anything else imports them, so the real
# ``modules/__init__.py`` (which pulls in block_sparse_mlp, attn, sliding_attn,
# gated_delta_net, mlp -- every native module holding an ``exl3_mgemm`` call
# site) never executes in this deployment. Treating them as leaves is what makes
# the reachability answer faithful instead of a conservative over-approximation.
#
# ``exllamav3.model.config`` matters for the same reason and is the subtler of
# the three: the serving module's first import is
# ``from ...model.config import Config``, and the real ``model/config.py``
# reaches ``architecture/architectures.py`` -- which imports every architecture,
# including the ones that pull in ``modules.attn``, ``modules.dsv4`` and
# ``modules.gated_delta_net`` -- from a *function-local* import inside
# ``Config``. Stubbing the namespace is what keeps that whole fan-out out of the
# runtime closure. The installer writes this one through its own ``__name__``
# (``modules[config.__name__] = config``) rather than a literal key, so
# ``_installed_stub_names`` resolves it from the ``ModuleType`` literal.
STUB_NAMESPACES = frozenset(
    {"exllamav3.modules", "exllamav3.model", "exllamav3.model.config"}
)

MAX_INDICES_DECL = "#define MAX_INDICES 128"
V_INDICES_DECL = "__device__ int64_t v_indices[128];"
V_WEIGHTS_DECL = "__device__ half v_weights[128];"
SERVING_CALL_GEMM_GR = "exl3_gemm_gr("
SERVING_CALL_GEMM = "exl3_gemm("
SERVING_CLASS = "BC_LinearEXL3::run_gr"
SLICED_GUARD = "min_index < 0"
PER_MATRIX_GUARD = "per-matrix widths incompatible"

WRITE_RE = re.compile(r"\b(v_indices|v_weights)\s*\[[^\]]*\]\s*=")
GLOBAL_RE = re.compile(r"__global__[^\n]*\n\s*void\s+(\w+)\s*\(")
CALL_RE = re.compile(r"\bext\.(exl3_mgemm\w*)\s*\(|\b(exl3_mgemm\w*)\s*\(")
MODULE_REF_RE = re.compile(r"\bexllamav3(?:\.[A-Za-z_]\w*)+")
EXT_SYMBOL_RE = re.compile(r"\bexllamav3_ext\.([A-Za-z_]\w*)")
MGEMM_BIND_RE = re.compile(r'm\.def\(\s*"(exl3_mgemm\w*)"')


class Abort(RuntimeError):
    """A required source or anchor was missing; the audit cannot decide."""


def _read(path: Path, label: str) -> str:
    if not path.is_file():
        raise Abort(f"missing {label}: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - unreadable file
        raise Abort(f"unreadable {label}: {path}: {exc}") from exc


def kernel_scratch(kernel_src: str) -> dict:
    """Declarations plus the per-kernel containment of every scratch write."""
    if MAX_INDICES_DECL not in kernel_src:
        raise Abort(f"anchor not found: {MAX_INDICES_DECL!r}")
    if V_INDICES_DECL not in kernel_src:
        raise Abort(f"anchor not found: {V_INDICES_DECL!r}")
    if V_WEIGHTS_DECL not in kernel_src:
        raise Abort(f"anchor not found: {V_WEIGHTS_DECL!r}")

    spans = []
    for match in GLOBAL_RE.finditer(kernel_src):
        spans.append((match.group(1), match.start(), match.end()))
    if not spans:
        raise Abort("no __global__ kernels found in the pinned kernel header")
    bounds = []
    for index, (name, _, body_start) in enumerate(spans):
        body_end = spans[index + 1][1] if index + 1 < len(spans) else len(kernel_src)
        bounds.append((name, body_start, body_end))

    writers: list[dict] = []
    for match in WRITE_RE.finditer(kernel_src):
        owner = None
        for name, start, end in bounds:
            if start <= match.start() < end:
                owner = name
                break
        writers.append(
            {
                "array": match.group(1),
                "offset": kernel_src[: match.start()].count("\n") + 1,
                "kernel": owner,
            }
        )
    if not writers:
        raise Abort("no writes to the v_indices/v_weights scratch were found")

    orphans = [w for w in writers if w["kernel"] != "exl3_mgemm_kernel"]
    return {
        "max_indices": 128,
        "arrays": sorted({w["array"] for w in writers}),
        "writer_count": len(writers),
        "writer_kernels": sorted({str(w["kernel"]) for w in writers}),
        "kernels": [name for name, _, _ in bounds],
        "writes_outside_mgemm_kernel": orphans,
    }


def serving_path(linear_src: str) -> dict:
    """The deployed entry point must reach exl3_gemm, never exl3_mgemm."""
    if SERVING_CLASS not in linear_src:
        raise Abort(f"anchor not found: {SERVING_CLASS!r}")
    start = linear_src.index(SERVING_CLASS)
    end = linear_src.find("\nvoid ", start)
    body = linear_src[start:] if end == -1 else linear_src[start : start + end]
    if SERVING_CALL_GEMM_GR not in body:
        raise Abort(f"{SERVING_CLASS} does not call {SERVING_CALL_GEMM_GR}")
    if SERVING_CALL_GEMM not in body:
        raise Abort(f"{SERVING_CLASS} does not call {SERVING_CALL_GEMM}")
    return {
        "entry": SERVING_CLASS,
        "calls_exl3_gemm_gr": SERVING_CALL_GEMM_GR in body,
        "calls_exl3_gemm": SERVING_CALL_GEMM in body,
        "calls_exl3_mgemm": "exl3_mgemm" in body,
    }


def gemm_guards(gemm_src: str) -> dict:
    """Structural guards on the mgemm host entry that bound the scratch."""
    if "exl3_mgemm_gr" not in gemm_src:
        raise Abort("anchor not found: 'exl3_mgemm_gr'")
    if SLICED_GUARD not in gemm_src:
        raise Abort(f"anchor not found: sliced-mode guard {SLICED_GUARD!r}")
    if PER_MATRIX_GUARD not in gemm_src:
        raise Abort(f"anchor not found: {PER_MATRIX_GUARD!r}")
    return {
        "sliced_mode_requires_min_index_negative": SLICED_GUARD in gemm_src,
        "per_matrix_widths_require_min_index_negative": PER_MATRIX_GUARD in gemm_src,
        "declared_limit_slots": 128,
    }


def _module_name(path: Path, python_dir: Path) -> str:
    """Fully qualified name, matching the form ``import_closure`` returns.

    The package prefix must be kept: the closure is a set of dotted names such
    as ``exllamav3.modules.quant.exl3``, so a bare ``modules.quant.exl3`` key
    would never compare equal and the audit would report NOT_REACHABLE even
    when a serving-path module does call ``exl3_mgemm``.
    """
    rel = path.relative_to(python_dir).with_suffix("")
    return ".".join((python_dir.name, *rel.parts))


def mgemm_call_sites(python_dir: Path) -> dict[str, list[int]]:
    if not python_dir.is_dir():
        raise Abort(f"missing python package dir: {python_dir}")
    sites: dict[str, list[int]] = {}
    for path in sorted(python_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - unreadable file
            continue
        hits = [
            text[: match.start()].count("\n") + 1
            for match in CALL_RE.finditer(text)
            if match.group(1) or match.group(2)
        ]
        if hits:
            sites[_module_name(path, python_dir)] = hits
    return sites


def _resolve_module(python_dir: Path, name: str) -> tuple[Path | None, bool]:
    """Map a dotted module name to its file. Returns (path, is_package)."""
    parts = name.split(".")
    if parts and parts[0] == python_dir.name:
        parts = parts[1:]
    base = python_dir.joinpath(*parts) if parts else python_dir
    module_file = base.with_suffix(".py")
    if module_file.is_file():
        return module_file, False
    package_file = base / "__init__.py"
    if package_file.is_file():
        return package_file, True
    return None, False


def import_closure(python_dir: Path, seeds: list[str]) -> dict:
    """Transitive in-package imports, resolving relative (``level``) imports.

    Relative imports are the norm inside ExLlamaV3 (``from ...model.config
    import Config``), so a regex over absolute names silently under-counts the
    closure. ``ast`` plus explicit ``level`` handling is used instead.

    Returns the resolved module set **plus** the in-package references that
    could not be resolved or parsed. Skipping those silently is a fail-open:
    deleting the serving module, or corrupting a module inside the closure,
    shrinks the closure and turns an undecidable tree into a confident
    ``NOT_REACHABLE``. Names introduced by ``Import`` (and by an
    ``ImportFrom``'s module part) are module references; names introduced by an
    ``ImportFrom`` *alias* are usually attributes, so they are only treated as
    modules when they actually resolve.
    """
    seen: set[str] = set()
    required: set[str] = set()
    resolvable: set[str] = set()
    unparseable: set[str] = set()
    root = python_dir.name
    queue: list[tuple[str, bool]] = [(name, True) for name in seeds]
    while queue:
        name, is_module_ref = queue.pop()
        # Record the requirement *before* the dedup check: a name first seen as
        # a weak alias candidate (an attribute) and later as a hard `import`
        # must still be validated as a module, whatever the queue order.
        if is_module_ref:
            required.add(name)
        if name in seen:
            continue
        seen.add(name)
        if name in STUB_NAMESPACES:
            resolvable.add(name)
            continue
        path, is_package = _resolve_module(python_dir, name)
        if path is None:
            continue
        resolvable.add(name)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            unparseable.add(f"{name}: unreadable ({exc})")
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            unparseable.add(f"{name}: {exc}")
            continue
        parts = name.split(".")
        package_parts = parts if is_package else parts[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    queue.append((alias.name, True))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    drop = node.level - 1
                    if drop > len(package_parts):
                        continue
                    base_parts = package_parts[: len(package_parts) - drop]
                    if node.module:
                        base_parts = base_parts + node.module.split(".")
                    if not base_parts:
                        continue
                    base = ".".join(base_parts)
                    queue.append((base, True))
                    # `from . import helper` names a *submodule*, and its call
                    # sites are as reachable as any other import's. Queue the
                    # alias as a module candidate: it resolves when it really is
                    # a submodule and is ignored when it is an attribute of
                    # `base` (the same treatment the absolute branch gives).
                    for alias in node.names:
                        queue.append((f"{base}.{alias.name}", False))
                elif node.module:
                    queue.append((node.module, True))
                    for alias in node.names:
                        queue.append((f"{node.module}.{alias.name}", False))
    unresolved = sorted(
        name
        for name in required
        if (name == root or name.startswith(root + ".")) and name not in resolvable
    )
    return {
        "modules": seen,
        "unresolved": unresolved,
        "unparseable": sorted(unparseable),
    }


def overlay_module_seeds(overlay_dir: Path | None) -> list[str]:
    """ExLlamaV3 modules the vLLM overlay itself *imports* (fail-closed if absent).

    Parsed, not regex-scanned. Overlay sources also *name* ExLlamaV3 modules
    they merely patch (file paths, anchors, docstrings) and dotted attribute
    chains such as ``exllamav3.model.config.InferParams``. Seeding the closure
    from those both over-counts (patch targets are not call targets) and
    under-counts (an attribute chain resolves to no module at all), and the
    previous regex did both.
    """
    if overlay_dir is None:
        return []
    if not overlay_dir.is_dir():
        raise Abort(f"missing overlay dir: {overlay_dir}")
    seeds: set[str] = set()
    found_any = False
    for path in sorted(overlay_dir.glob("*.py")):
        found_any = True
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - unreadable file
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:  # pragma: no cover - overlay must be importable
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "exllamav3" or alias.name.startswith("exllamav3."):
                        seeds.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                module = node.module or ""
                if module == "exllamav3" or module.startswith("exllamav3."):
                    seeds.add(module)
    if not found_any:
        raise Abort(f"no overlay sources found in {overlay_dir}")
    return sorted(seeds)


def overlay_ext_symbols(overlay_dir: Path | None) -> set[str]:
    """``exllamav3_ext.<name>`` symbols the vLLM overlay actually calls."""
    if overlay_dir is None:
        return set()
    if not overlay_dir.is_dir():
        raise Abort(f"missing overlay dir: {overlay_dir}")
    symbols: set[str] = set()
    for path in sorted(overlay_dir.glob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - unreadable file
            continue
        symbols.update(EXT_SYMBOL_RE.findall(text))
    if not symbols:
        raise Abort(
            f"no exllamav3_ext.<symbol> references found in {overlay_dir}; "
            "cannot establish which extension entries the deployment reaches"
        )
    return symbols


def mgemm_entry_points(bindings_src: str) -> set[str]:
    """Extension entry points whose name is an ``exl3_mgemm*`` variant."""
    entries = set(MGEMM_BIND_RE.findall(bindings_src))
    if not entries:
        raise Abort(
            "no m.def(\"exl3_mgemm...\") entry found in the extension bindings; "
            "the scratch owner cannot be identified"
        )
    return entries


def worst_case_slots(top_k: int, max_num_seqs: int, draft_tokens: int) -> dict:
    """Slot counts the two ``min_index >= 0`` kernel branches can reach."""
    decode = max_num_seqs * (draft_tokens + 1)
    return {
        "production_top_k": top_k,
        "production_max_num_seqs": max_num_seqs,
        "production_draft_tokens": draft_tokens,
        "num_tokens_1_slots_batch_x_topk": decode,
        "num_tokens_gt_1_slots_bszm": decode,
    }


def _installed_stub_names(source: str) -> set[str]:
    """Namespaces a stub source actually *installs* into a module mapping.

    Structural, not textual: `unused = types.ModuleType(name)` contains every
    token a token-presence check looks for while installing nothing. What
    matters is a subscript assignment whose key is a namespace string and whose
    value is a ``ModuleType`` construction, possibly via a local alias
    (``module = types.ModuleType(name)`` ... ``mapping[name] = module``).
    """
    tree = ast.parse(source)
    module_typed: set[str] = set()
    # local variable -> the literal name passed to `types.ModuleType(...)`, so a
    # later `mapping[var.__name__] = var` can be resolved back to a namespace.
    module_type_literal: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if _callee_name(node.value.func) == "ModuleType":
                literal = _module_type_name(node.value)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module_typed.add(target.id)
                        if literal:
                            module_type_literal.setdefault(target.id, set()).update(literal)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
            if _callee_name(node.value.func) == "ModuleType" and isinstance(node.target, ast.Name):
                module_typed.add(node.target.id)
                literal = _module_type_name(node.value)
                if literal:
                    module_type_literal.setdefault(node.target.id, set()).update(literal)

    # Loop variables bound to string literals, e.g.
    # `for name, path in (("exllamav3", ...), ("exllamav3.modules", ...)):`.
    # Each iterated element is itself a tuple matching the loop target, so zip
    # the target names against that element's values.
    literal_names: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        if not isinstance(node.iter, (ast.Tuple, ast.List)):
            continue
        names = [t.id for t in ast.walk(node.target) if isinstance(t, ast.Name)]
        for item in node.iter.elts:
            if not isinstance(item, (ast.Tuple, ast.List)):
                continue
            for var, value in zip(names, item.elts):
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    literal_names.setdefault(var, set()).add(value.value)

    def _keys(node: ast.Subscript, assigned: ast.expr) -> set[str]:
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            return {key.value}
        if isinstance(key, ast.Name):
            return literal_names.get(key.id, set())
        # `mapping[config.__name__] = config` right after
        # `config = types.ModuleType("exllamav3.model.config")`. Only the same
        # variable on both sides counts: `mapping[other.__name__] = config` does
        # not install `config`'s namespace, so it stays unresolved.
        if (
            isinstance(assigned, ast.Name)
            and isinstance(key, ast.Attribute)
            and key.attr == "__name__"
            and isinstance(key.value, ast.Name)
            and key.value.id == assigned.id
        ):
            return module_type_literal.get(assigned.id, set())
        return set()

    def _is_module_value(value: ast.expr) -> bool:
        if isinstance(value, ast.Call):
            func = value.func
            callee = func.attr if isinstance(func, ast.Attribute) else (
                func.id if isinstance(func, ast.Name) else None
            )
            return callee == "ModuleType"
        if isinstance(value, ast.Name):
            return value.id in module_typed
        return False

    installed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not _is_module_value(node.value):
            continue
        for target in targets:
            if isinstance(target, ast.Subscript):
                installed |= _keys(target, node.value)
    return installed


def _callee_name(func: ast.expr) -> str | None:
    """The bare name of a called expression (``types.ModuleType`` -> ``ModuleType``)."""
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _module_type_name(call: ast.Call) -> set[str]:
    """The literal namespace passed to ``types.ModuleType(...)``, if it is one."""
    if _callee_name(call.func) != "ModuleType":
        return set()
    if (
        call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    ):
        return {call.args[0].value}
    for keyword in call.keywords:
        if (
            keyword.arg == "name"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ):
            return {keyword.value.value}
    return set()


def _installer_invocation(overlay_dir: Path) -> str | None:
    """The overlay file that calls the stub installer before importing exllamav3.

    Executing the installer proves it *can* install; it does not prove the
    deployment *runs* it. If the loader imports ``exllamav3`` without calling
    the installer first, ``modules/__init__.py`` executes and pulls in every
    native module holding an ``exl3_mgemm`` call site, so pruning those
    namespaces would be unsound. Returns the qualifying file, or ``None``.
    """
    for path in sorted(overlay_dir.glob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - unreadable file
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:  # pragma: no cover
            continue
        call_line: int | None = None
        first_exllamav3_import: int | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                callee = func.attr if isinstance(func, ast.Attribute) else (
                    func.id if isinstance(func, ast.Name) else None
                )
                if callee == "inject_config_stub" and call_line is None:
                    call_line = node.lineno
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "exllamav3" or module.startswith("exllamav3."):
                    if first_exllamav3_import is None:
                        first_exllamav3_import = node.lineno
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "exllamav3" or alias.name.startswith("exllamav3."):
                        if first_exllamav3_import is None:
                            first_exllamav3_import = node.lineno
        if call_line is None:
            continue
        if first_exllamav3_import is not None and first_exllamav3_import < call_line:
            continue
        return str(path)
    return None


def stub_contract(overlay_dir: Path | None) -> dict:
    """Justify pruning ``STUB_NAMESPACES`` from the closure.

    Treating ``exllamav3.modules``/``exllamav3.model``/``exllamav3.model.config``
    as leaves is only sound if the deployment really installs them as synthetic
    module objects before anything imports them -- otherwise their ``__init__.py``
    (or, for ``model.config``, the real module body) runs and pulls in every
    native module that holds an ``exl3_mgemm`` call site.

    Checked structurally rather than by token presence, and then *behaviourally*:
    the installer is actually run against a fresh mapping and the result is
    inspected. A source that mentions ``types.ModuleType`` and the namespace
    strings while installing nothing is rejected.
    """
    if overlay_dir is None:
        raise Abort(
            "no --overlay-dir given: the stub-namespace pruning cannot be "
            "justified, so the import closure would be unsound"
        )
    path = overlay_dir / "exl3_namespace.py"
    source = _read(path, "exllamav3 namespace stub")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise Abort(f"cannot parse {path.name}: {exc}") from exc
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "inject_config_stub"
        for node in tree.body
    ):
        raise Abort(f"{path.name} has no module-level inject_config_stub entry point")
    declared = _installed_stub_names(source)
    if STUB_NAMESPACES - declared:
        raise Abort(
            f"{path.name} has no `mapping[name] = types.ModuleType(name)` "
            f"assignment for {sorted(STUB_NAMESPACES - declared)}; refusing to "
            "prune them from the import closure"
        )

    # Behavioural confirmation: run the installer and look at what it produced.
    namespace: dict = {}
    try:
        exec(compile(source, str(path), "exec"), namespace)  # noqa: S102
        installer = namespace["inject_config_stub"]
        mapping: dict = {}
        installer(Path("."), mapping)
    except Exception as exc:  # pragma: no cover - defensive
        raise Abort(f"{path.name} could not be executed to verify the stub: {exc}") from exc
    not_installed = sorted(
        name for name in STUB_NAMESPACES if not isinstance(mapping.get(name), types.ModuleType)
    )
    if not_installed:
        raise Abort(
            f"{path.name} declares but does not install {not_installed} as module "
            "objects; refusing to prune them from the import closure"
        )

    invoker = _installer_invocation(overlay_dir)
    if invoker is None:
        raise Abort(
            "no overlay source calls inject_config_stub() before importing "
            "exllamav3; the deployment may load the real exllamav3.modules "
            "package, so pruning the stub namespaces is unsound"
        )
    return {
        "checked": True,
        "source": str(path),
        "installs": sorted(STUB_NAMESPACES),
        "verified_by": "structural assignment check + executed installer",
        "invoked_by": invoker,
    }


def audit(
    exl3_root: Path,
    python_dir: Path,
    serving_module: str,
    shapes: dict,
    overlay_dir: Path | None = None,
) -> dict:
    package_dir = exl3_root / PACKAGE_DIR
    if not package_dir.is_dir():
        raise Abort(f"missing ExLlamaV3 package dir: {package_dir}")
    kernel_src = _read(package_dir / KERNEL_REL, "pinned kernel header")
    gemm_src = _read(package_dir / GEMM_REL, "mgemm host entry")
    linear_src = _read(package_dir / LINEAR_REL, "LinearEXL3 binding")
    bindings_src = _read(package_dir / BINDINGS_REL, "extension bindings")
    scratch = kernel_scratch(kernel_src)
    serving = serving_path(linear_src)
    guards = gemm_guards(gemm_src)
    entries = mgemm_entry_points(bindings_src)
    stubs = stub_contract(overlay_dir)
    ext_symbols = overlay_ext_symbols(overlay_dir)
    binding_reaches_mgemm = sorted(ext_symbols & entries)
    sites = mgemm_call_sites(python_dir)
    seeds = [serving_module] + overlay_module_seeds(overlay_dir)
    closure = import_closure(python_dir, seeds)

    # A seed that will not resolve or parse means the reachability question
    # cannot be posed at all -- never a silent shrink of the closure.
    bad_seeds = [
        name
        for name in seeds
        if name in closure["unresolved"] or any(
            entry.split(":")[0] == name for entry in closure["unparseable"]
        )
    ]
    if bad_seeds:
        raise Abort(
            f"serving-path seed module(s) missing or unparseable: {bad_seeds}; "
            "the import closure cannot be established"
        )
    if closure["unparseable"]:
        raise Abort(
            "in-package module(s) inside the closure could not be parsed: "
            f"{closure['unparseable'][:8]}; the closure is incomplete"
        )
    if closure["unresolved"]:
        raise Abort(
            "in-package import(s) inside the closure did not resolve: "
            f"{closure['unresolved'][:8]}; the closure is incomplete"
        )

    closure_modules = closure["modules"]
    closure_hits = sorted(name for name in sites if name in closure_modules)

    # Decisive facts -- each one *is* an observed reachable path to an
    # exl3_mgemm entry: the overlay naming an mgemm extension symbol, the
    # serving Python module calling one, or the C++ bridge calling one.
    decisive: set[str] = set(binding_reaches_mgemm)
    if serving_module in sites:
        decisive.add(serving_module)
    if serving["calls_exl3_mgemm"]:
        decisive.add(f"{serving['entry']} -> exl3_mgemm")
    decisive = sorted(decisive)

    # Closure modules that hold mgemm call sites but are not themselves a
    # decisive entry. Importing a module does not execute its call sites, so
    # these are *unresolved*, not unreachable -- a call graph would be needed
    # to discharge them. Reporting them as advisory context and then claiming
    # NOT_REACHABLE is exactly the fail-open this audit exists to prevent.
    unresolved_calls = sorted(set(closure_hits) - {serving_module})

    if scratch["writes_outside_mgemm_kernel"]:
        verdict = "ABORT"
        reason = (
            "the v_indices/v_weights scratch is written outside exl3_mgemm_kernel; "
            "the containment premise of this audit is broken."
        )
    elif decisive:
        worst = max(
            shapes["num_tokens_1_slots_batch_x_topk"],
            shapes["num_tokens_gt_1_slots_bszm"],
        )
        if worst > scratch["max_indices"]:
            verdict = "REACHABLE_OVERFLOW"
            reason = (
                f"the deployment reaches an exl3_mgemm entry point ({decisive}) and "
                f"can reach {worst} slots at the declared production shapes, above "
                f"MAX_INDICES={scratch['max_indices']}. Silent memory corruption; "
                "outranks all performance work. Adopt the bounded-slot fix."
            )
        else:
            verdict = "REACHABLE_OK"
            reason = (
                f"the deployment reaches an exl3_mgemm entry point ({decisive}) but "
                f"stays at or below MAX_INDICES={scratch['max_indices']} at the "
                "declared production shapes."
            )
    elif unresolved_calls:
        verdict = "ABORT"
        reason = (
            "the deployment's own entry points do not reach an exl3_mgemm entry, "
            f"but {len(unresolved_calls)} module(s) inside the serving import "
            f"closure hold exl3_mgemm call sites: {unresolved_calls}. Importing a "
            "module is not executing its call sites, so static non-reachability "
            "is NOT established. Discharge these with call-graph evidence (or "
            "prove they are never loaded) before any NOT_REACHABLE verdict."
        )
    else:
        verdict = "NOT_REACHABLE"
        reason = (
            "no exl3_mgemm entry point is reachable: the 128-slot scratch is "
            "written only inside exl3_mgemm_kernel, the only bridge the vLLM "
            "overlay uses (BC_LinearEXL3 -> exl3_gemm_gr/exl3_gemm) never calls "
            "exl3_mgemm, the extension symbols the overlay names "
            f"({sorted(ext_symbols)}) exclude every exl3_mgemm* entry, the "
            "serving module itself has no exl3_mgemm call site, and no other "
            "module in the serving import closure has one either."
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "serving_module": serving_module,
        "seed_modules": sorted(set(seeds)),
        "overlay_ext_symbols": sorted(ext_symbols),
        "mgemm_entry_points": sorted(entries),
        "binding_reaches_mgemm": binding_reaches_mgemm,
        "decisive_reachable": decisive,
        "unresolved_closure_call_site_modules": unresolved_calls,
        "stub_contract": stubs,
        "scratch": scratch,
        "serving_path": serving,
        "guards": guards,
        "call_sites_total": sum(len(v) for v in sites.values()),
        "call_site_modules": {k: len(v) for k, v in sorted(sites.items())},
        "import_closure_size": len(closure_modules),
        "worst_case": shapes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exl3-root",
        type=Path,
        required=True,
        help="ExLlamaV3 repo root containing exllamav3/exllamav3_ext",
    )
    parser.add_argument(
        "--python-dir",
        type=Path,
        help="exllamav3 python package dir (default: <exl3-root>/exllamav3)",
    )
    parser.add_argument("--serving-module", default="exllamav3.modules.quant.exl3")
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        help="vLLM overlay dir whose exllamav3 references seed the closure",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--draft-tokens", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python_dir = args.python_dir or (args.exl3_root / PACKAGE_DIR)
    shapes = worst_case_slots(args.top_k, args.max_num_seqs, args.draft_tokens)
    try:
        report = audit(
            args.exl3_root, python_dir, args.serving_module, shapes, args.overlay_dir
        )
    except Abort as exc:
        print(json.dumps({"verdict": "ABORT", "reason": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["verdict"] == "ABORT":
        return 1
    return 2 if report["verdict"] == "REACHABLE_OVERFLOW" else 0


if __name__ == "__main__":
    raise SystemExit(main())
