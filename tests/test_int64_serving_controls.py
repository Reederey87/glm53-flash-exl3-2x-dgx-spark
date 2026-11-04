from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def controls():
    path = Path(__file__).resolve().parents[1] / "experiments/dense-int64/verify_serving_controls.py"
    spec = importlib.util.spec_from_file_location("serving_controls", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = {"Id": module.IMAGES["incumbent"], "Os": "linux", "Architecture": "arm64",
             "Config": {"Env": ["SECRET=never-print-this", "PATH=/bin"] + [k + "=8" for k in sorted(module.BUILD_KEYS)],
                        "Entrypoint": ["bash"], "Cmd": ["entry.sh"], "User": "", "WorkingDir": "/app"}}
    container = {"Image": image["Id"], "Config": deepcopy(image["Config"]),
                 "State": {"Running": True, "Pid": 123}, "Path": "bash", "Args": ["entry.sh"],
                 "HostConfig": {"Binds": ["/host/b:/b", "/host/a:/a"], "NetworkMode": "host", "Privileged": False},
                 "Mounts": [{"Destination": "/b", "Source": "/host/b", "RW": True}, {"Destination": "/a", "Source": "/host/a", "RW": False}]}
    return module, container, image


def rebuilt(controls):
    module, container, image = controls
    current, built = deepcopy(container), deepcopy(image)
    current["Image"] = built["Id"] = module.IMAGES["control"]
    for config in (current["Config"], built["Config"]):
        config["Env"] = [row.split("=", 1)[0] + "=4" if row.split("=", 1)[0] in module.BUILD_KEYS else row for row in config["Env"]]
    return current, built


def test_exact_image_inherited_change_is_explicit_and_private(controls):
    module, old, old_image = controls
    new, new_image = rebuilt(controls)
    result = module.verify("incumbent", "control", old, new, old_image, new_image)
    assert result["environment_changed_keys"] == sorted(module.BUILD_KEYS)
    assert result["exact_image_inherited_changes"] and not result["environment_equal"]
    assert "never-print-this" not in json.dumps(result)
    assert "SECRET" not in json.dumps(result)


def test_unchanged_incumbent_and_paired_candidate(controls):
    module, old, old_image = controls
    new = deepcopy(old)
    new["HostConfig"]["Binds"].reverse()
    new["Mounts"].reverse()
    new["Config"]["Env"].reverse()
    assert module.verify("incumbent", "incumbent", old, new, old_image, old_image)["environment_equal"]
    control, control_image = rebuilt(controls)
    candidate, candidate_image = deepcopy(control), deepcopy(control_image)
    candidate["Image"] = candidate_image["Id"] = module.IMAGES["candidate"]
    assert module.verify("control", "candidate", control, candidate, control_image, candidate_image)["environment_equal"]


@pytest.mark.parametrize("fault", ["wrong_image", "tag_only", "platform", "stopped", "pid", "env_extra", "env_secret", "env_duplicate", "env_removed", "build_arbitrary", "build_override", "image_env", "image_cmd", "container_cmd", "argv", "host", "mount", "mount_duplicate", "missing_mounts"])
def test_runtime_drift_cannot_hide_behind_allowed_build_keys(controls, fault):
    module, old, old_image = controls
    new, new_image = rebuilt(controls)
    if fault == "wrong_image":
        new["Image"] = module.IMAGES["incumbent"]
    elif fault == "tag_only":
        new_image["Id"] = "vllm-node-b12x:latest"
    elif fault == "platform":
        new_image["Architecture"] = "amd64"
    elif fault == "stopped":
        new["State"]["Running"] = False
    elif fault == "pid":
        new["State"]["Pid"] = 0
    elif fault == "env_extra":
        new["Config"]["Env"].append("NEW_FLAG=1")
    elif fault == "env_secret":
        new["Config"]["Env"][0] = "SECRET=changed-private-value"
    elif fault == "env_duplicate":
        new["Config"]["Env"].append("PATH=/bin")
    elif fault == "env_removed":
        new["Config"]["Env"].pop(1)
    elif fault == "build_arbitrary":
        new["Config"]["Env"] = ["MAX_JOBS=999" if x.startswith("MAX_JOBS=") else x for x in new["Config"]["Env"]]
    elif fault == "build_override":
        old["Config"]["Env"] = ["MAX_JOBS=999" if x.startswith("MAX_JOBS=") else x for x in old["Config"]["Env"]]
    elif fault == "image_env":
        new_image["Config"]["Env"].append("EXTRA=1")
    elif fault == "image_cmd":
        new_image["Config"]["Cmd"] = ["different.sh"]
    elif fault == "container_cmd":
        new["Config"]["Cmd"] = ["different.sh"]
    elif fault == "argv":
        new["Args"].append("extra")
    elif fault == "host":
        new["HostConfig"]["Privileged"] = True
    elif fault == "mount":
        new["Mounts"][0]["RW"] = False
    elif fault == "mount_duplicate":
        new["Mounts"].append(deepcopy(new["Mounts"][0]))
    else:
        old.pop("Mounts")
        new.pop("Mounts")
    with pytest.raises(ValueError):
        module.verify("incumbent", "control", old, new, old_image, new_image)


def test_candidate_cannot_inherit_an_unregistered_environment_change(controls):
    module, _, _ = controls
    old, old_image = rebuilt(controls)
    new, new_image = deepcopy(old), deepcopy(old_image)
    new["Image"] = new_image["Id"] = module.IMAGES["candidate"]
    new["Config"]["Env"].append("UNREGISTERED=1")
    with pytest.raises(ValueError):
        module.verify("control", "candidate", old, new, old_image, new_image)


def test_cli_failure_does_not_export_private_input(controls, tmp_path):
    module, old, old_image = controls
    paths = []
    for name, data in (("old", old), ("new", old), ("old_image", old_image), ("new_image", old_image)):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps([data]))
        paths.append(str(path))
    command = [sys.executable, module.__file__, "--reference-arm", "incumbent", "--arm", "candidate"]
    for name, path in zip(("reference-container", "current-container", "reference-image", "current-image"), paths):
        command += ["--" + name, path]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1 and json.loads(result.stdout)["verified"] is False
    assert "SECRET" not in result.stdout + result.stderr
    assert "never-print-this" not in result.stdout + result.stderr
