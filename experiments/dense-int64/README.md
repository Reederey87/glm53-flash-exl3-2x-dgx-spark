# Dense split-K Int64 candidate

Prepared, not deployed or GPU-qualified. `kernel.patch` backports the address
arithmetic from FlashInfer `f5f5341d711f8210ce91af5731a0841570f70e53` to the
incumbent's monolithic `dense_gemm.py` at `91271757f7746f3dfb58f7c2aeb44a663389de9e`.
The upstream regression is appended to the incumbent's existing test module,
whose preparation/workspace helpers retain the required API. `manifest.json`
binds both base and candidate bytes. The installed head kernel matches the
base SHA256 exactly on both head and worker (direct SSH verification).

Only output layout pitches and output coordinates become Int64. Weight
encoding, activation arithmetic, tile selection and launch geometry stay
unchanged. The four launch layouts and five output-coordinate sites match the
upstream correction. Python parsing and patch applicability pass locally;
these are not evidence of CUDA compilation or numerical correctness.

## Qualification before any serving trial

1. After the MTP window, stop production through its unit, verify both ranks
   are gone and wait for MemFree >=90 GiB on both nodes. Save the rollback.
2. Build from the pinned full source Docker context. Do not bind-mount this
   kernel into the running image or use a Python-only runtime overlay. Freeze
   the exact build context/arguments and dependency identities first. If a
   rebuild changes dependencies, qualify a matching unpatched control build
   rather than attributing that drift to Int64 arithmetic.
3. Use fresh, separate compiler/tuning caches for the candidate. Compile and
   execute native SM121a kernels on each GB10. Record physical GPU identity,
   source and compiled-artifact hashes, toolchain, and test commands.
4. Run the existing blockscaled A16, blockscaled and MXFP8 correctness/graph
   suites. Run the added
   `test_nvfp4_split_partials_exceed_int32_elements` on the candidate, including
   a Compute Sanitizer memcheck run. M=4096, N=34816, K=1024, split-K=16 makes
   FP32 partial scratch exceed 8 GiB. The test checks exact constant output,
   changes the input and replays the captured graph. Treat a skip as missing
   evidence. Do not deliberately run the known overflowing address pattern on
   the incumbent; use bounded shapes there for comparison.
5. Inspect small-shape and large-shape compilation for spills, allocation
   changes and graph-address stability. No invalid access, nonfinite output,
   replay allocation or new compilation during replay is allowed.
6. Only after these gates, preregister the full serving window with the built
   image identities and unchanged checkpoint/KV/context/launcher controls.
   Use the explicit-max harness and complete no-cache ladder at c1/2/4/8,
   three repetitions, plus functional smoke. Require no robust >2% regression
   in prefill, decode or acceptance. This is a correctness improvement; a
   throughput-neutral result can qualify, but a speedup must be measured.
7. Restore the incumbent and re-arm its watchdog pending final adoption.

The broader updated FlashInfer image remains a separate queue item. This
backport isolates the address fix from its extensive API/policy changes.

## Reproducible build preparation

`prepare_build.py` archives reviewed launcher commit
`0bdc9c94cd89843e1c8f7d7b33df2d5bc746c03d` from a local git object store into a
new directory. It verifies the original Dockerfile, pins the ARM64 CUDA 13.0.2
base digest and NCCL 2.32.3 commit, and adds the hash-checked source treatment
before FlashInfer's wheel build. It does not edit that checkout or production.

```bash
uv run --frozen --no-dev experiments/dense-int64/prepare_build.py \
  --upstream-repo /path/to/spark-vllm-docker \
  --out local/int64-build-context
```

Freeze the helper commit and generated `context-sha256.txt`, then stage this
context to an isolated build directory owned by `nvidia` on spark1. Only in
the registered idle window, run `bash build_pair.sh glm53-int64-UNIQUE-LABEL`
there. The wrapper refuses production-active/low-memory states, validates
both nodes directly over QSFP, and never replaces an existing image tag.
It uses the upstream full Dockerfile's vLLM and FlashInfer export stages,
with four build jobs and `12.1a`. One rebuilt vLLM wheel is shared by both
runners; only the FlashInfer kernel treatment differs. Both retain the exact
kernel/test hashes and test sources in the image. A wheel build alone does
not compile every lazy B12X kernel; the native GPU gates above remain mandatory.

The wrapper records image identities, wheel hashes and package inventories,
and rejects a package-version mismatch between arms. Unpinned transitive
dependencies can differ from the incumbent, so compare this matched rebuilt
control with the candidate, and separately qualify the final candidate against
the incumbent. Full bit-for-bit reproducibility of external apt/Python indexes
is not claimed. BuildKit cache reuse is allowed for common source build layers;
GPU qualification requires fresh, separate JIT/tuning caches for each arm.

Exa source review used the upstream
[Dockerfile](https://github.com/eugr/spark-vllm-docker/blob/0bdc9c94cd89843e1c8f7d7b33df2d5bc746c03d/Dockerfile)
and [build script](https://github.com/eugr/spark-vllm-docker/blob/0bdc9c94cd89843e1c8f7d7b33df2d5bc746c03d/build-and-copy.sh).
Newer `main` has a different standalone B12X path; it is not substituted for
the reviewed embedded implementation. Local vLLM architecture selection,
Exllamav3 compile/cache rules and TensorFold's native CUDA requirements were
reviewed before preparing this source-build path.

Research: Exa lookup covered the overflow and synchronization alternatives;
the exact commit is available in local git (Exa's commit-page crawl timed out).
Local vLLM, Exllamav3 and TensorFold were reviewed first. Their pool-address
and complete-state rules reinforce the high-offset/replay gate; no EXL3
checkpoint or different serving engine is introduced by this patch.

## Native test environment and evidence

The current image includes Compute Sanitizer but not pytest. A CPU-only,
256 MiB-limited disposable container verified the following environment setup
against the frozen serving-kit `pyproject.toml` and `uv.lock`:

```bash
unset UV_SYSTEM_PYTHON
export UV_PROJECT_ENVIRONMENT=/tmp/qualification-venv
export UV_CACHE_DIR=/tmp/qualification-cache
uv venv --system-site-packages --python /usr/bin/python3 "$UV_PROJECT_ENVIRONMENT"
uv sync --frozen --project /harness
```

Here `/harness` is a read-only copy of the frozen serving-kit project. The
check resolved pytest 9.1.1 inside the temporary venv, with Torch 2.13.0+cu130,
CUTLASS DSL 4.7.0 and FlashInfer 0.7.1 still coming from the image's system
packages. It did not import Torch or access a GPU. This qualifies environment
setup only; recheck the identities in each built image before native tests.
Do not install a standalone B12X package: the image's compatibility module
loads the embedded FlashInfer implementation.

Run GPU tests only after the idle/memory gate, separately for each node and
image arm. Use a new receipt directory and fresh compiler/tuning caches for
each arm. Set `B12X_COMPILE_CACHE_DIR`, `CUTE_DSL_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, `CUDA_CACHE_PATH`,
`FLASHINFER_WORKSPACE_BASE` and `XDG_CACHE_HOME` beneath that directory; do not
mount the serving caches. Record the collected test list and JUnit results.
The full source build retains these suites under `/opt/dense-int64-tests`:

```bash
cd /opt/dense-int64-tests
uv run --no-sync --project /harness python -m pytest -c /dev/null --rootdir . \
  tests/experimental/b12x/gemm/test_blockscaled_a16.py \
  tests/experimental/b12x/gemm/test_blockscaled.py \
  tests/experimental/b12x/gemm/test_mxfp8_linear.py \
  -k 'not test_nvfp4_split_partials_exceed_int32_elements' \
  --junitxml=/receipts/bounded.xml -ra
```

That bounded suite runs on both arms. On the **candidate only**, run the added
high-offset regression normally and under memcheck, retaining both results:

```bash
uv run --no-sync --project /harness compute-sanitizer --tool memcheck \
  --target-processes all --error-exitcode 86 --log-file '/receipts/memcheck-%p.log' \
  python -m pytest -c /dev/null --rootdir . \
  tests/experimental/b12x/gemm/test_blockscaled_a16.py::test_nvfp4_split_partials_exceed_int32_elements \
  --junitxml=/receipts/high-offset-memcheck.xml -ra
```

Inspect every skip, not just pytest's exit code. In particular, the MXFP8
suite skips if the installed DSL lacks `MmaMXF8Op`; that would leave coverage
missing, not prove correctness. Hash emitted native artifacts and inspect
their resource/spill reports as well as source hashes. TensorFold's SASS
identity checks reinforce that instruction encodings, including continuation
words, matter when comparing compiled code; Python AST checks are insufficient.

The [Compute Sanitizer manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)
documents the nonzero error exit code and child-process instrumentation.
Do not force blocking launches: doing so changes execution and can reduce
reported errors. The [uv environment guide](https://docs.astral.sh/uv/pip/environments/)
describes isolated environments; this setup was also executed in the actual
incumbent image. Local vLLM sanitizer tests, Exllamav3 architecture/cache
selection and TensorFold native-artifact tests were reviewed before this plan.

The reusable runner is in `qualification/`. Stage its four files together
with this experiment's `manifest.json` into a new isolated checks directory,
and write `sha256.txt` for those five files before transfer. The locked project
files are staged separately under
`/home/nvidia/glm53-native-tests/int64-20261103/harness`. Invoke `host-run.sh`
with the exact built image ID, arm (`control` or `candidate`) and a new
directory beneath `.../int64-20261103/results/`, only after the build completes
and the common both-node idle gate passes. Execute one arm/node at a time.
The runner refuses a live build/serving unit, existing result directory,
wrong source bytes or non-GB10 GPU. It does not start or configure production.

JUnit results, skip reasons, installed package/source identities, cache objects
and resource reports are retained. Missing/skipped tests and absent/nonzero
memcheck summaries fail the automated result check. Sanitizer communication
failure terminates its test process (`--kill`); CUDA initialization is required.
The [NVIDIA sanitizer manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)
documents these controls. The runner's automated result is only one gate:
inspect compiled artifacts and the [reported register/local-memory resources](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
before claiming native qualification. Resource-report collection alone does
not prove absence of spills, and none of these tests measures serving speed.

If a completed source phase leaves MemFree below the registered floor,
`release_build_cache.py` can advise away clean cache pages from the exact,
closed vLLM build workspace mounted read-only at `/completed-vllm`. Inspect
the terminal build status and identify that layer first; the helper is not
for active builds, model files or global cache purging. It preserves data and
reports actual memory separately from the sizes of files advised. Always
rerun the unchanged both-node memory gate before resuming a build.
