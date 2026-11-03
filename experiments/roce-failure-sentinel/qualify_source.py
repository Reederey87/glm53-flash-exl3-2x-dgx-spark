#!/usr/bin/env python3
"""CPU qualification of the candidate's real host methods and protocol sources."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def qualify(source: Path) -> dict:
    root = source / "flashinfer/experimental/b12x/comm/roce"
    names = ["_control.py", "roce_oneshot.py", "_proxy.py", "_roce_proxy.c",
             "_oneshot_cute.py", "_allgather_cute.py", "_twoshot_cute.py", "_varlen_cute.py"]
    texts = {name: (root / name).read_text() for name in names}
    manifest = json.loads((Path(__file__).parent / "manifest.json").read_text())
    prefix = "flashinfer/experimental/b12x/comm/roce/"
    expected = {Path(item["path"]).name: item["candidate_sha256"]
                for item in manifest["files"] if item["path"].startswith(prefix)}
    for item in manifest["files"]:
        if hashlib.sha256((source / item["path"]).read_bytes()).hexdigest() != item["candidate_sha256"]:
            raise ValueError(f"source differs from frozen manifest: {item['path']}")
    actual = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in texts.items()}
    if actual != expected:
        raise ValueError("source differs from frozen candidate manifest")
    constants = {}
    exec(compile(texts["_control.py"], "<control-constants>", "exec"), constants)
    assert constants["CTRL_FAILED_WORD"] == 14
    assert constants["CTRL_FAILED"] == 56 and constants["CTRL_HOST_BYTES"] == 60
    cls = next(n for n in ast.parse(texts["roce_oneshot.py"]).body
               if isinstance(n, ast.ClassDef) and n.name == "RoceOneshotAllReduce")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in {"check_health", "poisoned"}]
    assert len(methods) == 2
    for method in methods:
        method.decorator_list = []
    namespace = {**constants, "_varlen_cute": SimpleNamespace(ERROR_KIND_INVALID_COUNT=1)}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "<candidate-host-methods>", "exec"), namespace)
    rows = []
    for sequence in (0, 1, 0xFFFFFFFF):
        for failed in (False, True):
            for invalid_count in (False, True):
                words = [0] * 15
                words[2] = sequence if sequence < 0x80000000 else sequence - 0x100000000
                words[3], words[6], words[7], words[14] = 1, 0, int(invalid_count), int(failed)
                state = SimpleNamespace(_proxy=None, _ctrl_np=words, rank=0)
                error = ""
                try:
                    namespace["check_health"](state)
                except RuntimeError as exc:
                    error = str(exc)
                poisoned = namespace["poisoned"](state)
                assert bool(error) == failed and poisoned == failed
                if failed and not invalid_count:
                    assert f"sequence {sequence};" in error
                    assert f"epoch stopped at {(sequence - 1) & 0xFFFFFFFF}," in error
                if failed and invalid_count:
                    assert "unusable byte count" in error
                rows.append({"sequence": sequence, "failed": failed,
                             "invalid_count": invalid_count, "host_raises": bool(error),
                             "poisoned": poisoned, "error": error})
    coverage = {}
    for name, stores, gates in [("_oneshot_cute.py", 1, 1), ("_allgather_cute.py", 1, 1),
                                ("_twoshot_cute.py", 1, 2), ("_varlen_cute.py", 2, 2)]:
        text = texts[name]
        assert "st_release_gpu_u32(poison_ptr, seq)" not in text
        assert "ld_relaxed_sys_u32(ctrl_base + Int64(8))" not in text
        assert text.count("st_release_gpu_u32(poison_ptr, Uint32(1))") == stores
        assert text.count("st_relaxed_sys_u32(ctrl_base + Int64(CTRL_FAILED), Uint32(1))") == stores
        assert text.count("ld_relaxed_sys_u32(ctrl_base + Int64(CTRL_FAILED))") == gates
        coverage[name] = {"failure_stores": stores, "publication_gates": gates}
    assert "ABI_VERSION = 8" in texts["_proxy.py"]
    assert "#define ROCE_ABI_VERSION 8" in texts["_roce_proxy.c"]
    assert "#define CTRL_FAILED 14" in texts["_roce_proxy.c"]
    assert texts["_roce_proxy.c"].count("__atomic_load_n(&ctrl[CTRL_FAILED], __ATOMIC_ACQUIRE)") == 2
    assert "if (ctrl[CTRL_ERROR] != 0)" not in texts["_roce_proxy.c"]
    assert "API_VERSION = 1" in texts["roce_oneshot.py"]
    return {"host_cases": rows, "source_coverage": coverage,
            "source_sha256": actual,
            "host_checks_passed": True, "gpu_qualified": False,
            "limitation": "Source counts are coverage checks, not native execution or a concurrency proof."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = qualify(args.source)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"host_cases_passed": len(report["host_cases"]), "gpu_qualified": False}))
