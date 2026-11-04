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

## Full-source build preparation

`prepare_build.py` produces this candidate's isolated ARM64 source context
from reviewed launcher `0bdc9c94`. The CUDA base is pinned by digest; vLLM,
FlashInfer and NCCL by commit. Torch 2.13.0+cu130, DSL 4.7.0 and SM121a stay
fixed. It does not modify a source checkout or build an image:

```bash
uv run --locked --no-dev experiments/roce-single-block/prepare_build.py \
  --upstream-repo /path/to/spark-vllm-docker \
  --out local/roce-build-context
```

After the registered shutdown/idle window, build the generated context on
the head as `nvidia` with `bash build_pair.sh glm53-roce1-UNIQUE-LABEL`.
It creates one shared freshly compiled vLLM wheel, separate control/candidate
FlashInfer wheels and two full runner images. Only the candidate's three
manifest-listed files change. Image labels and
`/workspace/roce-single-block.json` retain treatment identity. Package,
Debian and wheel inventories are recorded; dependency drift requires a matched
rebuilt control comparison. No standalone b12x package is installed.

The wrapper refuses an active serving unit/watchdog, existing tags/artifacts,
changed context hashes, stale serving containers, insufficient MemFree on
either node, reserve mismatch or probe failure. Its read-only both-node helper
comes from the frozen startup candidate; no service change is installed.
Readiness is checked before each build stage.

Offline qualification used fresh pinned FlashInfer sources, not the already
patched dense-Int64 tree. Both arms returned the exact manifest hashes; all
eight shared patch-helper tests pass. This is not GPU qualification.

## Native test identity

Use **the same strengthened candidate test source against both image arms**.
Export the candidate's `/opt/roce-single-block-tests` and mount that directory
read-only into each test container. Do not overlay the installed kernel
package. Record test SHA256
`a9a90db81b6e851cf9b03f9182c00a2fe9523bfe4a465761eca3d33054218346`
on both ranks and retain both JUnit reports. The stronger test adds small
reductions between larger graph collectives and cumulative allocator checks.

Prepare the qualified locked temporary test venv with system site packages,
retaining installed Torch/DSL/FlashInfer. Run from the exported test root so
`tests.test_helpers.parametrize` resolves correctly. Use fresh per-arm/per-rank
compiler caches. Inside each idle qualification container with the actual
two HCAs, launch both ranks under bounded supervisors:

```bash
uv run --frozen --project /harness python -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=1 --node-rank="$RANK" \
  --master-addr=192.168.177.10 --master-port=29650 \
  -m pytest -x -v tests/experimental/b12x/comm/test_roce_oneshot_gpu.py \
  --junitxml="/receipts/roce-rank-$RANK.xml"
```

Require identical image IDs/test hashes and direct fabric rendezvous. Record
actual rail selection and the striping test, not just environment names.
Module-wide skipping fails qualification. A capacity skip must identify
missing coverage and cannot count as passing the full gate.

The incumbent host `check_health()` uses the failed sequence as a sentinel;
source inspection does not prove wrap-safe failure reporting at sequence zero.
The stronger host-visible completion and separate failure flag described in
upstream b12x PR438 are not bundled into this optimization. Existing timeout
tests do not prove wrap coverage. Resolve that qualification before declaring
this candidate fully safe; any protocol repair remains a separate treatment.
