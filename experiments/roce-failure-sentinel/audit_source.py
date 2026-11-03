#!/usr/bin/env python3
"""Reproduce the pinned host failure-sentinel defect without GPU imports.

This executes only two extracted host methods with synthetic control words.
It does not simulate CUDA execution or qualify a replacement protocol.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

REVISION = "91271757f7746f3dfb58f7c2aeb44a663389de9e"
PREFIX = "flashinfer/experimental/b12x/comm/roce/"
HASHES = {
    "roce_oneshot.py": "44802a72762a3af27ad5c97d138612c54f0f177256f3be316de248de83ae9931",
    "_oneshot_cute.py": "080032d608229602b9cb871cb541e8ca1e7c2ba4fe5e88709eec166cdc2f1d10",
    "_allgather_cute.py": "9f6a61a618aecff93ece0579c88e4e820069ffd8c6b811f88ee993942082afcf",
    "_varlen_cute.py": "3a59ebe946242b9c7b24bd17ee0fbe256ea34d94e68351cb649695101297ccd9",
    "_cute_intrinsics.py": "7cfddbf51f3d36e60b94a7527c472c43804f45cabf7e7165bc7a9e8c0e892f39",
}


def audit(repo: Path) -> dict:
    sources = {}
    for name, expected in HASHES.items():
        data = subprocess.check_output(["git", "show", f"{REVISION}:{PREFIX}{name}"], cwd=repo)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"unreviewed source: {name}")
        sources[name] = data.decode()
    module = ast.parse(sources["roce_oneshot.py"])
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef)
               and n.name == "RoceOneshotAllReduce")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in {"check_health", "poisoned"}]
    if len(methods) != 2:
        raise ValueError("expected exactly two reviewed host methods")
    for method in methods:
        method.decorator_list = []
    varlen = ast.parse(sources["_varlen_cute.py"])
    invalid = next(ast.literal_eval(n.value) for n in varlen.body
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                   and t.id == "ERROR_KIND_INVALID_COUNT" for t in n.targets))
    namespace = {"_varlen_cute": SimpleNamespace(ERROR_KIND_INVALID_COUNT=invalid)}
    # No original module imports, constructors, CUDA calls or live state.
    exec(compile(ast.Module(body=methods, type_ignores=[]),
                 "<pinned-host-methods>", "exec"), namespace)
    rows = []
    cases = [("idle", 0, False), ("ordinary_timeout", 1, True),
             ("last_nonzero_timeout", 0xFFFFFFFF, True), ("wrapped_timeout", 0, True)]
    for label, sequence, timeout in cases:
        words = [0] * 8
        words[0] = sequence
        if timeout:
            words[2], words[3], words[6] = sequence, 1, 0
        state = SimpleNamespace(_proxy=SimpleNamespace(failed=lambda: False),
                                _ctrl_np=words, rank=0)
        error = ""
        try:
            namespace["check_health"](state)
        except RuntimeError as exc:
            error = str(exc)
        rows.append({"case": label, "timeout_recorded": timeout, "sequence": sequence,
                     "control_words": words, "host_raises": bool(error),
                     "poisoned": namespace["poisoned"](state), "error": error})
    for name in ("_oneshot_cute.py", "_allgather_cute.py"):
        for fragment in ("seq = epoch + Uint32(1)",
                         "st_relaxed_sys_u32(ctrl_base + Int64(8), seq)",
                         "st_release_gpu_u32(poison_ptr, seq)"):
            if fragment not in sources[name]:
                raise ValueError(f"expected timeout protocol not found: {name}")
    reproduced = rows[1]["host_raises"] and rows[2]["host_raises"] and not (
        rows[3]["host_raises"] or rows[3]["poisoned"])
    return {"scope": "exact pinned host methods with synthetic control words; GPU stores inspected only",
            "flashinfer_commit": REVISION, "source_sha256": HASHES, "cases": rows,
            "host_defect_reproduced": reproduced, "gpu_reproduced": False,
            "runtime_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.repo)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"host_defect_reproduced": report["host_defect_reproduced"],
                      "gpu_reproduced": False, "receipt": str(args.out)}))
