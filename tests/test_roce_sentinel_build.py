from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def sentinel_bundle(tmp_path):
    path = Path(__file__).resolve().parents[1] / "experiments/roce-failure-sentinel/apply_kernel.py"
    spec = importlib.util.spec_from_file_location("sentinel_apply", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, bundle, output = (tmp_path / name for name in ("source", "bundle", "output"))
    (source / "tests").mkdir(parents=True)
    bundle.mkdir()
    (source / "kernel.py").write_text("failure = sequence\n")
    (source / "tests/test_kernel.py").write_text("# original regression\n")
    for args in (["init", "-q"], ["add", "."],
                 ["-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                  "commit", "-qm", "Pinned test source"]):
        subprocess.run(["git", *args], cwd=source, check=True, capture_output=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    patch = ("--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-failure = sequence\n+failure = 1\n"
             "--- /dev/null\n+++ b/control.py\n@@ -0,0 +1 @@\n+FLAG = 14\n")
    native = "# native regression source, not executed by this helper\n"
    (bundle / "kernel.patch").write_text(patch)
    (bundle / "test_wrap_sentinel_gpu.py").write_text(native)
    manifest = {"flashinfer_base": revision, "patch_sha256": sha(patch),
                "native_test_sha256": sha(native), "proxy_abi": {"control": 7, "candidate": 8},
                "files": [
                    {"path": "kernel.py", "base_sha256": sha("failure = sequence\n"),
                     "candidate_sha256": sha("failure = 1\n")},
                    {"path": "control.py", "base_sha256": None,
                     "candidate_sha256": sha("FLAG = 14\n")}]}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return module, source, bundle, output, manifest


def test_control_preserves_missing_file_and_candidate_adds_it(sentinel_bundle):
    module, source, bundle, output, _ = sentinel_bundle
    control = module.apply(source, bundle, output / "control", "control")
    assert not (source / "control.py").exists()
    assert (source / "kernel.py").read_text() == "failure = sequence\n"
    candidate = module.apply(source, bundle, output / "candidate", "candidate")
    assert (source / "control.py").read_text() == "FLAG = 14\n"
    assert (source / "kernel.py").read_text() == "failure = 1\n"
    assert control["files"]["control.py"] is None
    assert candidate["proxy_abi"] == 8 and control["proxy_abi"] == 7
    assert not candidate["gpu_qualified"]
    test_path = "kernel-tests/tests/experimental/b12x/comm/test_wrap_sentinel_gpu.py"
    assert (output / "control" / test_path).read_bytes() == (output / "candidate" / test_path).read_bytes()


@pytest.mark.parametrize("fault", ["patch", "native", "revision", "dirty", "new_collision", "scope", "duplicate", "bad_arm"])
def test_invalid_inputs_are_rejected_before_source_mutation(sentinel_bundle, fault):
    module, source, bundle, output, manifest = sentinel_bundle
    if fault == "patch":
        with (bundle / "kernel.patch").open("a") as handle:
            handle.write("unexpected\n")
    elif fault == "native":
        (bundle / "test_wrap_sentinel_gpu.py").write_text("different test\n")
    elif fault == "revision":
        manifest["flashinfer_base"] = "unreviewed"
    elif fault == "dirty":
        (source / "unrelated.py").write_text("unexpected source\n")
    elif fault == "new_collision":
        (source / "control.py").write_text("do not overwrite\n")
    elif fault == "scope":
        patch = (bundle / "kernel.patch").read_text() + "--- /dev/null\n+++ b/other.py\n@@ -0,0 +1 @@\n+OTHER = 1\n"
        (bundle / "kernel.patch").write_text(patch)
        manifest["patch_sha256"] = sha(patch)
    elif fault == "duplicate":
        manifest["files"].append(manifest["files"][0].copy())
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    before = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*.py")}
    with pytest.raises(ValueError):
        module.apply(source, bundle, output, "unknown" if fault == "bad_arm" else "candidate")
    after = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*.py")}
    assert after == before
    assert not output.exists()


def test_existing_export_is_not_overwritten(sentinel_bundle):
    module, source, bundle, output, _ = sentinel_bundle
    output.mkdir()
    receipt = output / ".roce-failure-sentinel.json"
    receipt.write_text("previous receipt\n")
    with pytest.raises(ValueError, match="already exists"):
        module.apply(source, bundle, output, "candidate")
    assert receipt.read_text() == "previous receipt\n"
    assert (source / "kernel.py").read_text() == "failure = sequence\n"
