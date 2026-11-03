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
