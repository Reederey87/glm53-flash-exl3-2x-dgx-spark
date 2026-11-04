#!/usr/bin/env python3
"""Compare private Docker snapshots without exporting environment values.

Read-only component of the Int64 serving gate. Effective vLLM argv, installed
source hashes, checkpoint, API, KV and speed qualification remain separate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

IMAGES = {
    "incumbent": "sha256:a7f76ae9f18c7701c9cde5f72c606393d9a21ca94dd8e53b27589935f64fcbe2",
    "control": "sha256:3b5da79c42ed4204440cfb07c6de1eba46441d5cba6b57d5428d44e60c10d0ed",
    "candidate": "sha256:bad0f2005665489026ef9febafe8bbc28f6fe0821681cdb82479a21802d1287b",
}
BUILD_KEYS = {"CMAKE_BUILD_PARALLEL_LEVEL", "MAKEFLAGS", "MAX_JOBS", "NINJAFLAGS"}
TRANSITIONS = {("incumbent", "incumbent"), ("incumbent", "control"), ("control", "candidate")}
LAUNCH_FIELDS = ("Entrypoint", "Cmd", "WorkingDir", "User", "StopSignal", "Healthcheck")


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def one(value: object) -> dict:
    require(isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict), "expected one inspected object")
    return value[0]


def environment(config: dict) -> dict:
    rows = config.get("Env")
    require(isinstance(rows, list), "missing environment")
    result = {}
    for row in rows:
        require(isinstance(row, str) and "=" in row, "invalid environment record")
        key, value = row.split("=", 1)
        require(bool(key) and key not in result, "empty or duplicate environment key")
        result[key] = value
    return result


def changed(left: dict, right: dict) -> set:
    missing = object()
    return {key for key in left.keys() | right.keys() if left.get(key, missing) != right.get(key, missing)}


def host_config(container: dict) -> dict:
    result = deepcopy(container.get("HostConfig"))
    require(isinstance(result, dict) and bool(result), "missing host configuration")
    binds = result.get("Binds")
    require(isinstance(binds, list) and all(isinstance(x, str) for x in binds), "missing bind configuration")
    result["Binds"] = sorted(binds)
    return result


def mounts(container: dict) -> list:
    result = container.get("Mounts")
    require(isinstance(result, list) and bool(result), "missing runtime mounts")
    require(all(isinstance(m, dict) and isinstance(m.get("Destination"), str) for m in result), "invalid runtime mounts")
    require(len({m["Destination"] for m in result}) == len(result), "duplicate mount destination")
    return sorted(result, key=lambda m: m["Destination"])


def verify(reference_arm: str, arm: str, reference_container: dict, current_container: dict,
           reference_image: dict, current_image: dict) -> dict:
    require((reference_arm, arm) in TRANSITIONS, "unregistered arm comparison")
    for role, container, image in ((reference_arm, reference_container, reference_image), (arm, current_container, current_image)):
        require(container.get("Image") == image.get("Id") == IMAGES[role], "actual image identity differs from registered arm")
        require(image.get("Os") == "linux" and image.get("Architecture") == "arm64", "wrong image platform")
        require(isinstance(container.get("Config"), dict) and isinstance(image.get("Config"), dict), "missing launch configuration")
    require(current_container.get("State", {}).get("Running") is True, "current container is not running")
    require(type(current_container["State"].get("Pid")) is int and current_container["State"]["Pid"] > 0, "current container has no live PID")
    old_image_env, new_image_env = environment(reference_image["Config"]), environment(current_image["Config"])
    old_env, new_env = environment(reference_container["Config"]), environment(current_container["Config"])
    allowed = BUILD_KEYS if (reference_arm, arm) == ("incumbent", "control") else set()
    require(changed(old_image_env, new_image_env) == allowed, "image environment changes differ from registered comparison")
    require(changed(old_env, new_env) == allowed, "container environment changes differ from registered comparison")
    for key in allowed:
        require(key in old_image_env and key in new_image_env and key in old_env and key in new_env, "missing registered build setting")
        require(old_env[key] == old_image_env[key] and new_env[key] == new_image_env[key], "build setting is not the exact inherited image value")
    for field in LAUNCH_FIELDS:
        require(reference_image["Config"].get(field) == current_image["Config"].get(field), "image launch configuration differs")
        require(reference_container["Config"].get(field) == current_container["Config"].get(field), "container launch configuration differs")
    require(reference_container.get("Path") == current_container.get("Path") and isinstance(current_container.get("Path"), str), "container executable differs")
    require(reference_container.get("Args") == current_container.get("Args") and isinstance(current_container.get("Args"), list), "container arguments differ")
    require(host_config(reference_container) == host_config(current_container), "host configuration differs")
    require(mounts(reference_container) == mounts(current_container), "runtime mounts differ")
    return {"verified": True, "reference_arm": reference_arm, "arm": arm,
            "reference_image": IMAGES[reference_arm], "current_image": IMAGES[arm],
            "environment_equal": not allowed, "environment_changed_keys": sorted(allowed),
            "exact_image_inherited_changes": bool(allowed), "same_launch_configuration": True,
            "same_host_configuration": True, "same_mounts": True,
            "normalization": "Order of bind entries and mount records only; duplicate environment keys rejected",
            "scope": "Private Docker metadata comparison only; runtime source/argv/checkpoint/API/KV/speed gates remain separate"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-arm", choices=IMAGES, required=True)
    parser.add_argument("--arm", choices=IMAGES, required=True)
    for name in ("reference-container", "current-container", "reference-image", "current-image"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        inputs = [one(json.loads(path.read_text())) for path in
                  (args.reference_container, args.current_container, args.reference_image, args.current_image)]
        result = verify(args.reference_arm, args.arm, *inputs)
        result["verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except (ValueError, KeyError, TypeError, OSError):
        # Neither exception values nor source data are public output.
        print(json.dumps({"verified": False, "error": "private runtime comparison failed; inspect inputs locally"}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
