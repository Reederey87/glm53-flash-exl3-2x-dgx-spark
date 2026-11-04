#!/bin/bash
set -euo pipefail
arm=${1:?control or candidate}
[[ "$arm" == control || "$arm" == candidate ]]
[[ ! -e /receipts/container-start.txt ]]
date -u +%FT%TZ > /receipts/container-start.txt
unset UV_SYSTEM_PYTHON
export UV_PROJECT_ENVIRONMENT=/tmp/qualification-venv
export UV_CACHE_DIR=/tmp/qualification-cache
export B12X_COMPILE_CACHE_DIR=/receipts/cache/b12x
export CUTE_DSL_CACHE_DIR=/receipts/cache/cute
export TORCHINDUCTOR_CACHE_DIR=/receipts/cache/inductor
export TORCH_EXTENSIONS_DIR=/receipts/cache/extensions
export CUDA_CACHE_PATH=/receipts/cache/cuda
export FLASHINFER_WORKSPACE_BASE=/receipts/cache/flashinfer
export XDG_CACHE_HOME=/receipts/cache/xdg
export B12X_LOG_CUTE_COMPILES=1
export PYTHONUNBUFFERED=1
uv venv --system-site-packages --python /usr/bin/python3 "$UV_PROJECT_ENVIRONMENT"
uv sync --frozen --project /harness
uvrun=(uv run --no-sync --project /harness)
finish() {
    status=$?
    trap - EXIT
    set +e
    "${uvrun[@]}" python /checks/results.py "$arm"
    results_status=$?
    date -u +%FT%TZ > /receipts/container-end.txt
    printf '%s\n' "$status" > /receipts/test-command-exit.txt
    if [[ $status -ne 0 ]]; then exit "$status"; fi
    exit "$results_status"
}
trap finish EXIT
"${uvrun[@]}" python /checks/identity.py "$arm"
nvidia-smi -q > /receipts/nvidia-smi.txt
nvcc --version > /receipts/nvcc.txt
compute-sanitizer --version > /receipts/compute-sanitizer-version.txt
cd /opt/dense-int64-tests
find tests -type f -name '*.py' -exec sha256sum {} + > /receipts/test-source-sha256.txt
suites=(tests/experimental/b12x/gemm/test_blockscaled_a16.py
        tests/experimental/b12x/gemm/test_blockscaled.py
        tests/experimental/b12x/gemm/test_mxfp8_linear.py)
pytest_args=(-c /dev/null --rootdir .)
"${uvrun[@]}" python -m pytest "${pytest_args[@]}" "${suites[@]}" \
    -k 'not test_nvfp4_split_partials_exceed_int32_elements' --collect-only -q \
    > /receipts/collected-bounded.txt
"${uvrun[@]}" python -m pytest "${pytest_args[@]}" "${suites[@]}" \
    -k 'not test_nvfp4_split_partials_exceed_int32_elements' \
    --junitxml=/receipts/bounded.xml -ra
if [[ "$arm" == candidate ]]; then
    high=/checks/test_high_offset_sm121.py::test_nvfp4_split_partials_exceed_int32_elements
    "${uvrun[@]}" python -m pytest "${pytest_args[@]}" "$high" \
        --junitxml=/receipts/high-offset.xml -ra
    "${uvrun[@]}" compute-sanitizer --tool memcheck --target-processes all \
        --require-cuda-init yes --kill --error-exitcode 86 \
        --log-file '/receipts/memcheck-%p.log' \
        python -m pytest "${pytest_args[@]}" "$high" \
        --junitxml=/receipts/high-offset-memcheck.xml -ra
fi
