# Single-block RoCE candidate — queue 29

Prepared only. This is the exact kernel/test delta from FlashInfer
[`ad7027c`](https://github.com/local-inference-lab/flashinfer/commit/ad7027cafe8a000a265b0534b94824c4aea28700),
applied to incumbent `91271757`. Patch application and Python parsing pass;
`manifest.json` records the original and resulting hashes. The old utility
imports are retained: the later source's unrelated utility-module split is
not included. The base kernel hashes match the earlier direct inspection of
both deployed ranks.

The treatment skips staging and tail-arrival atomics only for a one-block
launch. It retains the system fence before the proxy doorbell, the multi-block
protocol, timeout/poison handling, and the host API. Kernel cache versions
advance (oneshot 6 to 7, all-gather 5 to 6). It does not change message thresholds,
HCAs, thread counts, NCCL fallback or model code.

The incumbent's actual `_grid_blocks` function, executed without CUDA, chooses
one block through 16 KiB with its default 512 threads. GLM BF16 hidden rows of
4096 elements are 8 KiB each, so one- and two-row reductions can use the path.
Four rows use two blocks; six verification rows use four. This establishes
potential relevance, not measured call coverage or model throughput.

Upstream reports two-host GB10 correctness and mixed-grid graph replay, with
component latency improvements around 0.8 microseconds for its tested small
messages. It explicitly does not establish end-to-end throughput improvement.
Those upstream results are not this cluster's receipts.

## Separate qualification window

1. Complete the current MTP window first. Freeze this candidate and preregister
   exact image identities, controls and thresholds before any serving restart.
2. Stop both production ranks through the unit, disarm the watchdog and require
   MemFree >=90 GiB on both. Compile from the pinned full source Dockerfile,
   with a matching unpatched control if dependencies differ from the incumbent.
   This CUDA launch-contract change is not a Python runtime overlay. Do not run
   the dense-Int64 build wrapper unchanged: it applies a different treatment.
3. Use separate fresh JIT/tuning caches. On both GB10s, compile native SM121a
   kernels and run the paired RoCE suite with the actual two-rail topology.
   Require both ranks' reports; a skipped distributed test is missing evidence.
4. Run the modified
   `test_all_gather_graph_replay_with_alternating_grid_sizes`, covering small
   reductions between larger gathers/reductions, changing inputs and checking
   every output against NCCL. Require unchanged allocated bytes and cumulative
   allocation counts during replay, plus successful `runtime.check_health()`.
5. Preserve bounded-timeout and poisoned-sequence behavior. Qualify transitions
   across grid sizes, repeated graph replay and sequence/counter wrap. Failed
   exchanges must not publish usable output or a healthy completion. Keep any
   independent timeout/sequence correction in a separate treatment.
6. Record coverage and paired latency distributions for 16 B, 8 KiB, 16 KiB,
   32 KiB and the actual GLM collective shapes. These microbenchmarks cannot
   substitute for the full explicit-max serving matrix and functional smoke.
   Keep exact weights, KV bytes/capacity, MTP depth and transport knobs fixed.
7. Require a repeatable >=2% decode gain outside control noise, with no >2%
   prefill/decode/draft-acceptance regression reproduced in two of three reps.
   Otherwise keep the incumbent. Restore it and re-arm the watchdog pending
   the user's final adoption decision.

Exa research included the primary upstream commit and NVIDIA's CUDA guide,
plus stronger timeout/sequence safeguards. Local vLLM's version-1 RoCE adapter,
Exllamav3's acquire/timeout path and TensorFold's
`docs/roce-gather.md` / `zig/kernels/cuda/roce_gather.cu` were reviewed before
preparation. TensorFold likewise requires checking host-visible failures after
synchronization; its transport is not substituted into this serving stack.
