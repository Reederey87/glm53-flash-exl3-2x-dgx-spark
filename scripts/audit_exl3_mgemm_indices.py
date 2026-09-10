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
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

PACKAGE_DIR = "exllamav3"
KERNEL_REL = "exllamav3_ext/quant/exl3_gemm_kernel.cuh"
GEMM_REL = "exllamav3_ext/quant/exl3_gemm.cu"
LINEAR_REL = "exllamav3_ext/libtorch/linear.cpp"
BINDINGS_REL = "exllamav3_ext/bindings.cpp"
SERVING_PY_REL = "modules/quant/exl3.py"

# ``overlay/exl3_namespace.py::inject_config_stub`` installs these two names as
# synthetic ``types.ModuleType`` namespaces before anything else imports them,
# so the real ``modules/__init__.py`` (which pulls in block_sparse_mlp, attn,
# sliding_attn, gated_delta_net, mlp -- every native module holding an
# ``exl3_mgemm`` call site) never executes in this deployment. Treating them as
# leaves is what makes the reachability answer faithful instead of a
# conservative over-approximation.
STUB_NAMESPACES = frozenset({"exllamav3.modules", "exllamav3.model"})

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


def import_closure(python_dir: Path, seeds: list[str]) -> set[str]:
    """Transitive in-package imports, resolving relative (``level``) imports.

    Relative imports are the norm inside ExLlamaV3 (``from ...model.config
    import Config``), so a regex over absolute names silently under-counts the
    closure. ``ast`` plus explicit ``level`` handling is used instead.
    """
    seen: set[str] = set()
    queue = list(seeds)
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in STUB_NAMESPACES:
            continue
        path, is_package = _resolve_module(python_dir, name)
        if path is None:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        parts = name.split(".")
        package_parts = parts if is_package else parts[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    queue.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    drop = node.level - 1
                    if drop > len(package_parts):
                        continue
                    base_parts = package_parts[: len(package_parts) - drop]
                    if node.module:
                        base_parts = base_parts + node.module.split(".")
                    if base_parts:
                        queue.append(".".join(base_parts))
                elif node.module:
                    queue.append(node.module)
                    for alias in node.names:
                        queue.append(f"{node.module}.{alias.name}")
    return seen


def overlay_module_seeds(overlay_dir: Path | None) -> list[str]:
    """ExLlamaV3 modules the vLLM overlay itself names (fail-closed if absent)."""
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
        for match in MODULE_REF_RE.finditer(text):
            seeds.add(match.group(0))
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
    ext_symbols = overlay_ext_symbols(overlay_dir)
    binding_reaches_mgemm = sorted(ext_symbols & entries)
    sites = mgemm_call_sites(python_dir)
    seeds = [serving_module] + overlay_module_seeds(overlay_dir)
    closure = import_closure(python_dir, seeds)
    closure_hits = sorted(name for name in sites if name in closure)
    # The static closure over-approximates: importing a module is not the same
    # as executing a call site inside it. Only two facts are decisive -- the
    # overlay naming an mgemm extension entry, and the serving module itself
    # calling one. Everything else is reported as advisory context.
    decisive = sorted(set(binding_reaches_mgemm) | (set(sites) & {serving_module}))
    advisory = sorted(set(closure_hits) - {serving_module})

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
    else:
        verdict = "NOT_REACHABLE"
        reason = (
            "no exl3_mgemm entry point is reachable: the 128-slot scratch is "
            "written only inside exl3_mgemm_kernel, the only bridge the vLLM "
            "overlay uses (BC_LinearEXL3 -> exl3_gemm_gr/exl3_gemm) never calls "
            "exl3_mgemm, the extension symbols the overlay names "
            f"({sorted(ext_symbols)}) exclude every exl3_mgemm* entry, and the "
            "serving module itself has no exl3_mgemm call site."
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
        "scratch": scratch,
        "serving_path": serving,
        "guards": guards,
        "call_sites_total": sum(len(v) for v in sites.values()),
        "call_site_modules": {k: len(v) for k, v in sorted(sites.items())},
        "advisory_closure_call_site_modules": advisory,
        "import_closure_size": len(closure),
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
