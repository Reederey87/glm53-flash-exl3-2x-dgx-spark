from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(params=("dense-int64", "roce-single-block", "roce-failure-sentinel"))
def preparer(request):
    path = Path(__file__).resolve().parents[1] / "experiments" / request.param / "prepare_build.py"
    spec = importlib.util.spec_from_file_location("experiment_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = """FROM base AS vllm-builder
RUN uv pip install build-dependencies
FROM base AS flashinfer-builder
RUN nvcc frozen-kernel.cu
FROM ${CUDA_IMAGE} AS runner
RUN --mount=type=bind,from=flashinfer_wheels,target=/workspace/flashinfer-wheels \
    uv pip install *.whl --override /tmp/wheel-override.txt
RUN uv pip install extra-dependencies --override /tmp/torch-override.txt
"""


def test_runner_constraint_preserves_compiler_stages_and_both_solves(preparer):
    result = preparer.pin_runtime_dependencies(SOURCE)
    marker = "FROM ${CUDA_IMAGE} AS runner\n"
    assert result.split(marker)[0] == SOURCE.split(marker)[0]
    assert "COPY runtime-constraints.txt /tmp/glm53-runtime-constraints.txt" in result
    assert result.count("--constraint /tmp/glm53-runtime-constraints.txt") == 2
    assert result.index('COPY runtime-constraints.txt') < result.index('uv pip install *.whl')
    assert "--override /tmp/wheel-override.txt --constraint" in result
    assert "--override /tmp/torch-override.txt --constraint" in result


@pytest.mark.parametrize("old,new", [
    ("FROM ${CUDA_IMAGE} AS runner\n", "FROM other AS runner\n"),
    ("FROM ${CUDA_IMAGE} AS runner\n", "FROM ${CUDA_IMAGE} AS runner\nFROM ${CUDA_IMAGE} AS runner\n"),
    ("--override /tmp/wheel-override.txt", "--override /tmp/changed.txt"),
    ("--override /tmp/torch-override.txt", "--override /tmp/changed.txt"),
    ("--override /tmp/torch-override.txt", "--override /tmp/torch-override.txt --override /tmp/torch-override.txt"),
    ("RUN --mount=type=bind,from=flashinfer_wheels", "RUN --mount=type=bind,from=changed_wheels"),
])
def test_changed_dockerfile_contract_is_rejected(preparer, old, new):
    with pytest.raises(ValueError):
        preparer.pin_runtime_dependencies(SOURCE.replace(old, new))


def test_profile_matches_effective_control_versions(preparer):
    data = preparer.runtime_constraints()
    assert hashlib.sha256(data).hexdigest() == preparer.RUNTIME_CONSTRAINTS_SHA
    record = json.loads(preparer.RUNTIME_CONSTRAINTS_PATH.with_name("control-effective-packages.json").read_text())
    excluded = {"vllm", "flashinfer-python", "flashinfer-jit-cache", "flashinfer-jit-cache-sm121a"}
    expected = {name: version for name, version in record["effective"] if name not in excluded}
    lines = [line for line in data.decode().splitlines() if line and not line.startswith("#")]
    assert len(lines) == len(expected) == 218
    assert dict(line.split("==", 1) for line in lines) == expected
    assert expected["setuptools"] == "78.1.0"  # effective version, not the shadowed OS distribution
    assert expected["openai"] == "3.43.0"
    assert expected["nvidia-cudnn-frontend"] == "1.31.0"
    raw_path = Path(__file__).resolve().parents[1] / "docs/receipts/2026-11-03-int64-build/frontend131-packages.json"
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == record["raw_inventory_sha256"]
    assert len(json.loads(raw_path.read_text())) == record["raw_count"] == 223


@pytest.mark.parametrize("contents", [None, b"openai==3.44.0\n"])
def test_missing_or_changed_profile_fails_before_context_creation(preparer, tmp_path, monkeypatch, contents):
    profile = tmp_path / "constraints.txt"
    if contents is not None:
        profile.write_bytes(contents)
    monkeypatch.setattr(preparer, "RUNTIME_CONSTRAINTS_PATH", profile)
    destination = tmp_path / "context"
    with pytest.raises((FileNotFoundError, ValueError)):
        preparer.prepare(tmp_path / "unused-repository", destination)
    assert not destination.exists()
