from __future__ import annotations

import importlib.util
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
    result = preparer.pin_runtime_frontend(SOURCE)
    marker = "FROM ${CUDA_IMAGE} AS runner\n"
    assert result.split(marker)[0] == SOURCE.split(marker)[0]
    assert 'nvidia-cudnn-frontend==1.31.0' in result
    assert result.count("--constraint /tmp/glm53-runtime-constraints.txt") == 2
    assert result.index('RUN echo') < result.index('uv pip install *.whl')
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
        preparer.pin_runtime_frontend(SOURCE.replace(old, new))
