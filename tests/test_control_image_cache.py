from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def guard(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "experiments/dense-int64"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("control_cache", folder / "release_control_image_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "TREATMENT_SHA", hashlib.sha256(b"receipt").hexdigest())
    monkeypatch.setattr(module, "DENSE_SHA", hashlib.sha256(b"source").hexdigest())
    return module


ROOT = "1 0 0:1 / / ro,relatime - overlay overlay rw,lowerdir=/layers"


def test_readonly_image_with_scratch_outside_scopes_is_allowed(guard):
    mounts = ROOT + "\n2 1 0:2 / /tmp rw - tmpfs tmpfs rw"
    guard.verify_container(mounts, b"receipt", b"source")


@pytest.mark.parametrize("mounts", [
    "", "broken", ROOT.replace("ro,relatime", "rw,relatime"),
    ROOT.replace("- overlay overlay", "- ext4 /dev/nvme"),
    ROOT + "\n" + ROOT,
    ROOT + "\n2 1 0:2 /host /usr/share/models ro - ext4 /dev/nvme rw",
    ROOT + "\n2 1 0:2 /host /workspace ro - ext4 /dev/nvme rw",
    ROOT + "\n2 1 0:2 /host /opt/cache rw - ext4 /dev/nvme rw",
])
def test_unsafe_or_ambiguous_image_mounts_are_rejected(guard, mounts):
    with pytest.raises(ValueError):
        guard.verify_container(mounts, b"receipt", b"source")


@pytest.mark.parametrize("receipt,source", [(b"candidate", b"source"), (b"receipt", b"changed kernel")])
def test_different_control_identity_is_rejected(guard, receipt, source):
    with pytest.raises(ValueError, match="wrong"):
        guard.verify_container(ROOT, receipt, source)


def test_candidate_identity_is_bound_to_exact_image(guard, monkeypatch):
    monkeypatch.setattr(guard, "CANDIDATE_TREATMENT_SHA", hashlib.sha256(b"candidate receipt").hexdigest())
    monkeypatch.setattr(guard, "CANDIDATE_DENSE_SHA", hashlib.sha256(b"candidate source").hexdigest())
    guard.verify_container(ROOT, b"candidate receipt", b"candidate source", guard.CANDIDATE_IMAGE_ID)
    for image, receipt, source in [
        (guard.CANDIDATE_IMAGE_ID, b"receipt", b"source"),
        (guard.CORRECTED_IMAGE_ID, b"candidate receipt", b"candidate source"),
        (guard.CANDIDATE_IMAGE_ID, b"candidate receipt", b"source"),
        ("sha256:unreviewed", b"candidate receipt", b"candidate source"),
    ]:
        with pytest.raises(ValueError):
            guard.verify_container(ROOT, receipt, source, image)
