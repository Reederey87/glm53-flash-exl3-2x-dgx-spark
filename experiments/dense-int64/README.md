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

Research: Exa lookup covered the overflow and synchronization alternatives;
the exact commit is available in local git (Exa's commit-page crawl timed out).
Local vLLM, Exllamav3 and TensorFold were reviewed first. Their pool-address
and complete-state rules reinforce the high-offset/replay gate; no EXL3
checkpoint or different serving engine is introduced by this patch.
