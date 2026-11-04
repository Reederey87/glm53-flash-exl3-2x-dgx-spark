#!/usr/bin/env python3
"""Restore original compiler directories after the BOTH-node idle gate.

Default is a read-only plan. --apply retains the final arm by rename, restores
the originals by rename, and never deletes cache or checkpoint contents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import stat
import subprocess
import sys

CACHES = (".cache/vllm", ".cache/flashinfer", ".cache/b12x", ".triton", ".tilelang")
ARMS = ("incumbent", "incumbent-retry1", "control", "candidate")
ORIGINAL_HASHES = {
    "head": "be393e87c273b8ae5b052d3adb4df50c7e1091d96062d6995f362ca4dd73dc9b",
    "worker": "3b32a5f03c371656af0fbcd8d116c127206ad3355ced4303c01f95b0f2dbae2e",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identity(path: Path) -> dict:
    require(path.resolve() == path and path.is_dir(), "cache must be a direct directory")
    value = path.stat()
    require((value.st_uid, value.st_gid) == (os.geteuid(), os.getegid()), "cache owner/group differs")
    return {"device": value.st_dev, "inode": value.st_ino,
            "mode": oct(stat.S_IMODE(value.st_mode)), "uid": value.st_uid,
            "gid": value.st_gid, "mtime_ns": value.st_mtime_ns}


def original_receipt(path: Path, node: str) -> dict:
    raw = path.read_bytes()
    require(node in ORIGINAL_HASHES and hashlib.sha256(raw).hexdigest() == ORIGINAL_HASHES[node], "original preservation receipt hash differs")
    return json.loads(raw)


def plan(root: Path, original: dict, arm: str) -> list[dict]:
    require(arm in ARMS, "unregistered final arm")
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), "invalid home directory")
    require(original.get("ok") is True and original.get("arm") == "incumbent", "requires original pre-experiment preservation")
    records = original.get("caches")
    require(isinstance(records, list) and len(records) == len(CACHES), "incomplete original cache receipt")
    rows = []
    for relative, record in zip(CACHES, records):
        active = root / relative
        saved = active.with_name(active.name + ".int64-before-incumbent-20261103")
        retained = active.with_name(active.name + f".int64-after-{arm}-20261103")
        require(record.get("cache") == str(active) and record.get("preserved") == str(saved)
                and record.get("fresh_empty") is True, "original path/ordering differs")
        require(not retained.exists() and not retained.is_symlink(), "final-arm archive already exists")
        current_id, saved_id = identity(active), identity(saved)
        require({k: saved_id[k] for k in ("device", "inode", "mode", "uid", "gid")}
                == {k: record[k] for k in ("device", "inode", "mode", "uid", "gid")}, "original directory identity differs")
        require(current_id["device"] == saved_id["device"] == root.stat().st_dev, "cross-device cache move refused")
        require(current_id["inode"] != saved_id["inode"], "active cache aliases original")
        rows.append({"active": str(active), "original": str(saved), "retain_final_arm": str(retained),
                     "active_identity": current_id, "original_identity": saved_id})
    return rows


def restore(rows: list[dict]) -> None:
    # Revalidate every directory before any mutation, even if a plan was saved.
    for row in rows:
        require(identity(Path(row["active"])) == row["active_identity"], "active cache changed after planning")
        require(identity(Path(row["original"])) == row["original_identity"], "original cache changed after planning")
        target = Path(row["retain_final_arm"])
        require(not target.exists() and not target.is_symlink(), "archive appeared after planning")
    journal = []
    try:
        for row in rows:
            for source, target, expected in (
                (Path(row["active"]), Path(row["retain_final_arm"]), row["active_identity"]),
                (Path(row["original"]), Path(row["active"]), row["original_identity"]),
            ):
                require(not target.exists() and not target.is_symlink(), "rename destination occupied")
                source.rename(target)
                journal.append((source, target, expected))
                require(identity(target) == expected, "directory identity changed during rename")
    except Exception:
        for source, target, expected in reversed(journal):
            require(not source.exists() and not source.is_symlink(), "rollback destination occupied; inspect retained directories")
            require(identity(target) == expected, "rollback identity changed; inspect retained directories")
            target.rename(source)
        raise


def local_idle_guard(node: str) -> None:
    running = subprocess.check_output(["docker", "ps", "-q"], text=True).strip()
    serving = subprocess.check_output(["docker", "ps", "-aq", "--filter", "name=vllm_node"], text=True).strip()
    require(not running and not serving, "containers remain; stop the window through the unit first")
    if node == "head":
        for unit in ("vllm-glm53fp4.service", "vllm-glm53fp4-watchdog.timer"):
            state = subprocess.check_output(["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "MainPID"], text=True)
            values = dict(line.split("=", 1) for line in state.splitlines() if "=" in line)
            require(values.get("ActiveState") == "inactive" and values.get("MainPID", "0") == "0", "serving unit or healer is not stopped")
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    require(int(memory["MemFree"].split()[0]) >= 94371840, "MemFree below 90 GiB")
    require(Path("/proc/sys/vm/min_free_kbytes").read_text().strip() == "1048576", "reserve changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", choices=ORIGINAL_HASHES, required=True)
    parser.add_argument("--final-arm", choices=ARMS, required=True)
    parser.add_argument("--original-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    require(sys.platform == "linux" and os.uname().machine == "aarch64", "requires ARM64 Linux")
    require(pwd.getpwuid(os.geteuid()).pw_name == "nvidia", "requires runtime owner nvidia")
    original = original_receipt(args.original_receipt, args.node)
    rows = plan(Path("/home/nvidia"), original, args.final_arm)
    if args.apply:
        local_idle_guard(args.node)
    with args.receipt.open("x") as stream:
        result = {"node": args.node, "final_arm": args.final_arm, "applied": False,
                  "original_receipt_sha256": ORIGINAL_HASHES[args.node],
                  "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "caches": rows}
        try:
            if args.apply:
                restore(rows)
                result["applied"] = True
            result["ok"] = True
        except Exception as exc:
            result.update(ok=False, error=type(exc).__name__)
            raise
        finally:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps({"ok": True, "applied": args.apply, "node": args.node, "cache_count": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
