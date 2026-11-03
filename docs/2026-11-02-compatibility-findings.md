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

## Sources and limits

Exa lookup and local source inspection were both used. The specific dense
split-K commit could not be fetched by Exa (live-crawl timeout); its code and
regression are available in the local FlashInfer git history. No web result
for an unrelated MLA split-K bug was treated as evidence for that commit.

- [NVIDIA model and precision contract](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local).
- [Flare's primary defect record](https://github.com/NVIDIA/flare-local-inference/blob/main/docs/glm-batched-speculation-defect.md): distinct export-state and grid-barrier defects; TP4/c16 NaN investigation remains open.
- [FlashInfer MLA addressing fix](https://github.com/flashinfer-ai/flashinfer/pull/3136): corroborates the general large-pool offset hazard, but is a different kernel and fix.

No checkpoint conversion, Flare deployment, new image build or dedicated GPU
kernel regression test has been performed. Live source/package identity is
now established; isolated candidate windows follow the full incumbent matrix.
