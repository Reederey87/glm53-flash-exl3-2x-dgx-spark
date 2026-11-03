# Exact-checkpoint and kernel screening

This supplements the [experiment queue](2026-11-02-improvement-queue.md).
Initial November 2 screening changed no cluster configuration. Direct SSH was
then unavailable; the Mac's
active LAN interface has `10.0.0.244`, but the recorded head address
`10.0.0.214` has an incomplete ARP entry and SSH returns `No route to host`.
Both configured Spark mDNS names fail resolution. Saved NVIDIA Sync config
contains those same names; no replacement address was found. Tailscale is
excluded by the user's instruction.

November 3 update: both nodes are reachable directly after power-on. The live
checkpoint matches revision `76348ddb...` on both ranks. The incumbent matrix
is running, with its watchdog temporarily stopped; serving controls remain
unchanged. Live source inspection confirms the DSA publishing barrier is
already installed. The following earlier loader tests remain valid.

## NVIDIA checkpoint, not a substitute GLM quant

The public Hugging Face model API resolved
`nvidia/GLM-5.3-Flash-FP4-Local` to revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`. Only model metadata,
`config.json` and `hf_quant_config.json` were fetched, at that immutable
revision. This does not identify the deployed snapshot until SSH inspection.

The configuration has 494 quantized-layer declarations:

| Stored algorithm | Declarations | Example / storage contract |
|---|---:|---|
| MXFP8 | 205 | `lm_head`: `weight_block_size=[1,32]` |
| FP8_PB_WO | 170 | layer 10 shared-expert down projection: `[128,128]` block scales |
| W4A16_NVFP4 | 77 | MTP experts and vision modules; group size 16 |
| NVFP4 | 42 | routed experts; group size 16, `csf_32x2_v1` scales |

Vocabulary size is 154,880; the benchmark's generated IDs below 150,000
are within this vocabulary. The declaration counts are not parameter counts.

## Flare's current loader fails before GPU qualification

This section describes the original local checkout `47581cc`. The newer
upstream result below supersedes it as the candidate path; it does not erase
the recorded failures of that older source.

Executed the actual functions from
`flare-local-inference/ignition/preparation/quantization.py` using the downloaded
configuration, via `importlib.util.spec_from_file_location`. No Torch or CUDA
was loaded. Called `quantized_layers(config)`, then `linear_format` or
`expert_format` on one representative declaration of each algorithm.

| Check | Result |
|---|---|
| NVFP4-CSF expert declaration | Recognized as `nvfp4_csf`, `a4` |
| MTP W4A16_NVFP4 expert declaration | Recognized as `nvfp4`, `a16` |
| FP8_PB_WO shared projection | `ValueError: unsupported linear quantization` |
| MXFP8 output head | `ValueError: unsupported linear quantization` |

This is a real declaration-function test, not full loader, numerical or
serving qualification. MXFP8's declaration normalization is only one issue;
adding a group-size alias does not provide native block-FP8 projection support.
The built-in `glm-5.3-flash` alias in `ignition/recipes/glm53/recipe.py` points
to `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD`, revision
`dec48abd...`, not the requested NVIDIA checkpoint. It cannot be used to claim
a successful exact-checkpoint comparison. Its preparation also handles
the vocabulary/draft head separately, which needs numerical scrutiny before
claiming unchanged checkpoint precision.

Local receipt: `local/checkpoint-audit/flare-declarations.json`, with full
representative inputs, exception text, source hash and metadata identity.
Source SHA256: `53d0e2cbbb5e928cfcc50a9a40040cf374f10c1e28ebe023a32ed2b23db720ef`.
Downloaded config was saved as indented JSON; its hash is a local serialization
hash, not a claim about the original HTTP payload bytes.

### New upstream exact-checkpoint support

On November 3 fetched upstream objects without changing the user's checkout.
New head `cc696a6bf828deeb04b09533c64c5cc5b26b0ab5` contains
`12c6c0b4eb3a2e11514ac0db5f0eb2ebe2b9a528`, which retains NVIDIA projection
formats and makes the GLM alias point to the exact requested revision.
Executed that head's declaration functions against the saved NVIDIA config:
MXFP8, FP8_PB_WO, W4A16_NVFP4 and NVFP4-CSF all pass. Receipt:
`local/checkpoint-audit/flare-latest-declarations.json`; source SHA256
`424dde0e0ce627e43f39753fe5e6a9a8d91136af9cbf2a0f448e8bbadfa26498`.

Code review shows native block-FP8 with FP32 scales, mixed projection groups,
BF16 MLA absorption, and use of the MXFP8 head for draft logits. This is a
stronger candidate than the older sibling bridge kit's head-widening patches.
The lockfile pins FlashInfer `f5f5341d...`; preparation uses Torch 2.14.1+cu132.
It is a separate complete engine/toolchain window, not a patch to current vLLM.

[Upstream issue 73](https://github.com/NVIDIA/flare-local-inference/issues/73)
reports this exact checkpoint with MTP on two Sparks at an earlier newer
commit, but its 1,993,728-token pool is below our 2,350,210-token control.
That report is neither our measurement nor a capacity waiver. The old local
Flare kit's 1M capacity gate and alternate QAD profile are also ineligible.
Native GPU loader/graph/export tests, measured unchanged capacity and full
serving parity remain required. Do not infer a present loader blocker from
the older declaration test, or declare the new engine qualified from four
CPU declaration checks.

### Flare cache and measurement contracts

Further review at `cc696a6b` executed the actual dependency-free `RankCosts`
declaration and GLM `_costs` function. With TP2, context parallelism and MTP,
one logical page spans 512 tokens and costs 1,723,392 bytes per rank. The
2,350,210-token baseline requires at least 4,591 logical pages (2,350,592
rounded tokens), or 7,912,092,672 bytes before physical rounding, private draft
storage and continuation aliases. These are planning numbers, not measured
usable capacity. Receipt: `local/checkpoint-audit/flare-kv-geometry.json`.
The raw Hugging Face layer names differ from the canonical recipe names;
counting `indexed_attention` in the unnormalized HF config gives a wrong result.

The [runtime memory contract](https://github.com/NVIDIA/flare-local-inference/blob/cc696a6bf828deeb04b09533c64c5cc5b26b0ab5/docs/cli-design.md)
defines `--memory-limit` as a whole-engine budget, including weights and graphs,
and `--kv-cache-limit` as a history ceiling. Neither is a direct 17 GiB KV-byte
setting. Preserve the byte and usable-capacity controls with measured native
allocation evidence; do not reinterpret a 17G engine cap as the baseline KV
reservation. GLM preparation also requires page-aligned context: the smallest
TP2/DCP export context covering 1,000,000 is 1,000,448. Enforcing the exact
public 1,000,000-token cap remains a separate compatibility check.

The current Rust server supports token-ID completion prompts, `ignore_eos`,
and `include_usage` plus `continuous_usage_stats`; its encoder puts cumulative
usage on completion choices. Its counters are `flare_accepted_draft_tokens_total`,
`flare_draft_request_rounds_total`, `flare_verification_draft_tokens_total` and
`flare_cache_reused_tokens_total`, rather than the vLLM names. Their exact
end-of-budget clipping and round semantics must be checked before comparison.
The reviewed router has no vLLM `/tokenize` endpoint: retain and verify the
baseline's explicit-max prompt IDs before the separate engine window. This
requires a qualified benchmark adapter, not a silent alias of metric names.
None of these source checks qualifies the Flare GPU deployment.

## Concrete FlashInfer corrections to qualify

### Dense split-K output address width — queue 16

Commit `f5f5341d711f8210ce91af5731a0841570f70e53` promotes output pitches and
coordinates to Int64 in embedded B12X's `dense_gemm/{compile,kernel}.py`.
Its regression is
`tests/experimental/b12x/gemm/test_blockscaled_a16.py::test_nvfp4_split_partials_exceed_int32_elements`.
The test uses M=4096, N=34816, K=1024 and split-K=16, checks constant outputs,
then mutates the input and replays a CUDA graph. Scratch exceeds 8 GiB, so
run only after both serving ranks stop and the memory gate passes.
Recorded upstream validation is SM120, not GB10. First inspect the incumbent
for this exact correction; do not rebuild an already-fixed image.

Live inspection confirms the correction is absent. The installed dense kernel
is byte-identical to the manifest source (`313ed8cd...`). A backport and the
upstream large-output regression are prepared under
`experiments/dense-int64`, with full source hashes and qualification steps.
No candidate kernel has been compiled or run yet.

The separate TMA refill fence correction `8d65eb248...` is already present in
that same installed file: its predicate is `self.load_path == "tma"`, not
`self.a16`. Commit-range listings alone would have incorrectly suggested it
was missing, because this source lineage includes equivalent changes.

### DSA publishing order — queue 27

Commit `37f5c4c1447376ac935b5f7cdf5ef35155e2bcef` adds
`cute.arch.sync_threads()` before thread zero publishes grid-barrier arrival
in `_fused_group_barrier`. The inspected standalone b12x checkout lacks that
pre-arrival barrier; the inspected FlashInfer checkout contains it.
Relevant regression:
`tests/experimental/b12x/attention/test_indexer_grid_barriers.py::test_barrier_orders_delayed_warp_writes`.
Require this adversarial test, the GLM top-k oracle and graph-replay checks on
GB10, then the serving matrix with fresh caches. This is a correctness fix,
not a promised speedup or a cure for all Flare numerical faults.

### Exported selection state — queue 28

Commit `6d7f238d40da62075f355fa5b000b0b25bfeb700` derives uncapped DSA width
from page-table metadata when no device-computed width is supplied. Current
Flare uses that contract. Its defect record explains why a preparation-filled
scratch scalar disappeared during export and corrupted >16-row verification.
The latest page-table-extent form was tested on SM120; the record explicitly
says it has not been rerun on Spark. Older exported packages need re-export.
This is a Flare integration gate, not evidence that the current vLLM runtime
has the same exporter bug.

## Incumbent versus updated upstream vLLM

Fetched the exact deployed LIL fork commit `8244d1c44348ae8a5bbfc9f1b607021652283ed4`
for a local read-only comparison with upstream checkout `190df0b...`.
The incumbent lazily registers `B12xModelLoader` for `--load-format b12x`
and has `model_loader/csf_utils.py` plus `quantization/nvfp4_csf.py`.
All three files are absent in the updated upstream tree. Its support for
ordinary ModelOpt MXFP8 and FP8_PB_WO does not supply the missing CSF loader.
Do not replace the image with this upstream checkout and expect the unchanged
recipe to load. Porting the loader is a separate precision/packing contract
change requiring numerical qualification, not a launcher-only update.

The incumbent also already implements `adaptive_speculative_tokens_window`
and `num_speculative_tokens_per_batch_size`. Its V2 graph manager captures
reachable draft depths. Its scheduler still selects the depth from
`len(num_scheduled_tokens)`, including prefills. The reported mixed-load
threshold issue in [vLLM #49548](https://github.com/vllm-project/vllm/issues/49548)
therefore warrants a dedicated gate, although that report uses another model
and runner configuration. K=0 still needs draft-state synchronization;
skipping its forward pass blindly is not a valid optimization. Static K=3
is the first isolated trial; adaptive depth is a later arm.

## Existing startup preparation and persistent caches

November 3 direct inspection found persistent host mounts for B12X,
FlashInfer, vLLM, Triton and TileLang on **both** ranks. No container override
disables B12X's default disk cache or redirects its compile-cache directory.
The deployed `b12x/_lib/compiler.py` matches the pinned source on both nodes:
SHA256 `3a73fe9d56fe1bf6277b92f8ec7c28b9b0a7b6c02e1ca2c04106c9922e83694b`.
Its cache key includes the package's file contents, runtime toolchain versions,
device architecture/SM count, compile options/environment and call structure
or explicit compile specification. Persistence and basic version separation
are therefore existing features, not unimplemented speed improvements.

The actual boot log also records weight, state and binding preparation via
`b12x_prepare.py`, cached selections, and completed CUDA graph capture. The
unit's own post-start hook only waits for HTTP health, but that does **not**
mean the engine skips native preparation. Receipt:
`local/cluster-20261103/namespaced/startup-preparation.txt`. This evidence does
not prove every mixed-load signature is covered or quantify cold/warm boot
latency; those are the remaining tests for candidates 02/03.

Exa follow-up found FlashInfer's stronger
[compile-cache isolation policy](https://github.com/flashinfer-ai/flashinfer/blob/main/ci/COMPILE_CACHE.md):
immutable image identities, private experiment caches and actual dependency
fingerprints address different risks. It is CI infrastructure, not a claim
about the incumbent's serving behavior. Keep fresh isolated caches for kernel
qualification; do not add a redundant cache mount or alter the running
baseline. Local vLLM warmup, Exllamav3 build targeting and TensorFold complete
prefix-state handling were reviewed alongside the deployed implementation.

## Dependency closure follow-up (November 3)

GitHub reports three open default-branch `uv.lock` alerts. This branch's
deployment-kit lock contains neither affected package; the serving image has
its own dependencies. A read-only, network-disabled inspection of incumbent
image `a7f76ae9f18c7701c9cde5f72c606393d9a21ca94dd8e53b27589935f64fcbe2`
at 18:45 UTC found `certifi==2026.7.22` and `filelock==3.32.3`.

| Package | Applicable advisory | Patched version | Existing build-log observation |
|---|---|---|---|
| certifi | [GHSA-6q48-6c7w-9gw2](https://github.com/advisories/GHSA-6q48-6c7w-9gw2): bundled emSign root trusts compromised intermediate keys | 2026.10.18; advisory recommends 2026.10.19 or later | 2026.10.23 |
| filelock | [GHSA-v4x8-3wr5-3wq8](https://github.com/tox-dev/filelock/security/advisories/GHSA-v4x8-3wr5-3wq8): SQLite-backed reader/writer locks in directories writable by another local user | 4.2.1 | 4.2.4 |

The separate [Windows filelock advisory](https://github.com/tox-dev/filelock/security/advisories/GHSA-9vhv-jxww-xfm5)
does not describe these Linux nodes. The pinned vLLM weight loader uses
`filelock.FileLock`, and a search of pinned FlashInfer's embedded B12X found
no `ReadWriteLock` or `AsyncReadWriteLock` references. This is a limited
call-site review, not proof that every installed dependency avoids those APIs.
No exploitation is asserted.

The newer versions appear in step 9 of the original pinned build's log, not
in a completed runner-image inventory. Later installs can change them. Before
declaring this resolved, inspect `artifacts/packages-control.json` and
`packages-candidate.json` from the completed build and each native run's
`identity.json`. Both arms must retain matching inventories and the existing
rebuilt-control-versus-incumbent qualification. Treat this as candidate 15's
dependency-closure review; do not change the frozen build, upgrade a running
container, or attribute dependency effects to the Int64 patch.

Receipt: [incumbent versions and build-log lines](receipts/2026-11-03-dependency-review/incumbent-and-build.txt).
The incumbent image and active build were not modified by this review.

### Declared-platform warning in the incumbent

A separate CPU-only `uv pip check --python /usr/bin/python3` checked 220
packages and returned 1 for one incompatibility: `nvidia-cusparselt-cu13`
was declared for a different platform. Inspection found version 0.8.1 with
`Tag: py3-none-manylinux2014_sbsa` in its WHEEL metadata, while the actual
`libcusparseLt.so.0` is ELF machine 183 (AArch64), 274,143,825 bytes. Its SHA256
is `fbb03bbf8d2238670f24256c5452281c2a660162899b838bb4d0cc06dad027df`.
No library was loaded or executed by this inspection.

[PyTorch PR 202090](https://github.com/pytorch/pytorch/pull/202090) describes
the matching invalid-tag defect and reports the same library hash for the
corrected 0.8.1.post1 wheels. The original PR body was read with `gh` after
Exa's canonical fetch failed; the hash comparison is against that upstream
report, not an independently downloaded post-release wheel. The incumbent's
Torch metadata requires `nvidia-cusparselt-cu13==0.8.1` on Linux, so a plain
upgrade to the post-release would not satisfy its declared exact dependency.
The broader [uv issue 17711](https://github.com/astral-sh/uv/issues/17711)
illustrates filename/internal-tag disagreement, but concerns a different
package/platform and is not direct proof about this binary.

Keep this packaging defect separate from the Int64 kernel result. Compare
the exact final-arm metadata and retain each `uv pip check` result; do not
call dependency closure clean if this warning remains. Any repair must retain
verified native payloads and coherent Torch requirements in a separately
frozen change, rather than silently overriding a package in this experiment.
Retained receipts: [dependency check](receipts/2026-11-03-dependency-review/incumbent-uv-check-full.txt),
[wheel metadata and ELF identity](receipts/2026-11-03-dependency-review/incumbent-cusparselt-metadata.txt),
and [library hash](receipts/2026-11-03-dependency-review/incumbent-cusparselt-library.sha256).
No installed metadata was edited.

## Sources and limits

Exa lookup and local source inspection were both used. The specific dense
split-K commit could not be fetched by Exa (live-crawl timeout); its code and
regression are available in the local FlashInfer git history. No web result
for an unrelated MLA split-K bug was treated as evidence for that commit.

- [NVIDIA model and precision contract](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local).
- [Flare's primary defect record](https://github.com/NVIDIA/flare-local-inference/blob/main/docs/glm-batched-speculation-defect.md): distinct export-state and grid-barrier defects; TP4/c16 NaN investigation remains open.
- [FlashInfer MLA addressing fix](https://github.com/flashinfer-ai/flashinfer/pull/3136): corroborates the general large-pool offset hazard, but is a different kernel and fix.

No checkpoint conversion or Flare deployment has been performed. The separate
Int64 full-source build is active, with its shared vLLM wheel exported and
FlashInfer's native providers compiling. No completed arm image or dedicated
changed-kernel GPU regression pass is claimed. Follow the
[build-window record](gates/2026-11-03-dense-int64.md) for subsequent results.
