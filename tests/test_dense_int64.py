from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def kernel_bundle(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "experiments/dense-int64/apply_kernel.py"
    spec = importlib.util.spec_from_file_location("dense_int64_apply", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, bundle, output = (tmp_path / name for name in ("source", "bundle", "output"))
    (source / "tests").mkdir(parents=True)
    bundle.mkdir()
    (source / "kernel.py").write_text("index = 32\n")
    (source / "tests/test_kernel.py").write_text("# regression source\n")
    patch = "--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-index = 32\n+index = 64\n"
    (bundle / "kernel.patch").write_text(patch)
    sha = lambda value: hashlib.sha256(value.encode()).hexdigest()
    manifest = {"flashinfer_base": "frozen-source", "patch_sha256": sha(patch), "files": [
        {"path": "kernel.py", "base_sha256": sha("index = 32\n"), "candidate_sha256": sha("index = 64\n")}
    ]}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "frozen-source\n")
    return module, source, bundle, output


def test_kernel_build_preserves_control_and_applies_only_candidate(kernel_bundle):
    module, source, bundle, output = kernel_bundle
    control = module.apply(source, bundle, output / "control", "control")
    assert (source / "kernel.py").read_text() == "index = 32\n"
    candidate = module.apply(source, bundle, output / "candidate", "candidate")
    assert (source / "kernel.py").read_text() == "index = 64\n"
    assert control["files"] != candidate["files"]
    assert not candidate["gpu_qualified"]
    assert (output / "candidate/kernel-tests/tests/test_kernel.py").read_text() == "# regression source\n"
    assert json.loads((output / "candidate/.dense-int64.json").read_text()) == candidate


@pytest.mark.parametrize("fault", ["patch", "base", "revision"])
def test_kernel_build_rejects_unreviewed_source_before_mutation(kernel_bundle, monkeypatch, fault):
    module, source, bundle, output = kernel_bundle
    if fault == "patch":
        with (bundle / "kernel.patch").open("a") as handle:
            handle.write("unexpected bytes\n")
    elif fault == "base":
        (source / "kernel.py").write_text("index = 16\n")
    else:
        monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "other-source\n")
    before = (source / "kernel.py").read_bytes()
    with pytest.raises(ValueError):
        module.apply(source, bundle, output, "candidate")
    assert (source / "kernel.py").read_bytes() == before
    assert not output.exists()
