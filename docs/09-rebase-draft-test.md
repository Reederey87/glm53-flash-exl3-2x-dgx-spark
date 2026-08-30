# The rebase draft, tested on real hardware

*2026-08-30. This is the field report for the plan in [07-rebase-plan.md](07-rebase-plan.md):
we built a draft image on the official day-0 GLM-5.3 base and ran it on the production
pair (with production stopped) through the full gate battery — decode, multi-agent
concurrency, prefix-cache replay, and long-context fills up to 500k tokens. Unless a
statement is explicitly attributed to upstream issues or prior field reports, the
benchmark values below were measured in this single test session; treat them as
one-boot observations, not converged baselines.*

## What we built

The draft image starts from the official day-0 GLM-5.3 arm64 image — which turns out
to be the very image the production fork was built from (same vLLM build, same
FlashInfer). On top of it:

1. **EXL3 as an out-of-tree plugin.** The fork's EXL3 quantization support was already
   written plugin-style; we packaged it as a separate pip package using vLLM's
   `register_quantization_config()` entry point. No vLLM source patches needed for the
   quant itself.
2. **exllamav3 v1.4.4 built for aarch64.** The public kernels do not build on ARM.
   Five build-blocker categories had to be patched: `__builtin_cpu_supports` guards
   (four files), `_mm_pause`/`__builtin_ia32_pause` mapped to ARM `yield` (two CUDA
   files), the AVX-512-VNNI CPU MoE GEMM stubbed out, the AVX2/AVX-512 CPU all-reduce
   files stubbed (two files), and — not an x86-ism but a build blocker — the cuSPARSE
   headers resolved from the `nvidia/cu13` wheel via `CPATH`. `cuobjdump` on the built
   extension shows native `sm_121` machine code (the GB10 arch), not just PTX.
3. **vLLM PR #53969** (the NoPE effective-topk validator), applied fail-closed.
4. **The DFlash2 speculative drafter, ported from the fork** — see below; it is not in
   the official tree.

### Tested artifacts (pin these to reproduce)

| Artifact | Identity |
|---|---|
| Base image | `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4` |
| vLLM in that image | `0.1.dev20051+g487ecf187` (FlashInfer 0.6.17, CUDA 13) |
| Model weights | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` snapshot `1ae6d70430a12d762917786696db06a7b4f9bbae` |
| Drafter weights | `incoai/GLM-5.3-Flash-DFlash2` snapshot `7d74cdd881ed7e32c31175984a67823127b66cfe` (BF16) |
| exllamav3 | tag `v1.4.4` + the ARM patch set above |
| DFlash2 port source | the production fork image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce6…` (files extracted, listed below) |

## The headline numbers

| Gate | No drafter¹ | With DFlash2 k=7¹ | Production (fork) |
|---|---|---|---|
| Structured decode | 14.0 tok/s | **60.5** | 72.8 |
| Prose decode | 13.8 | **19.8** | 27.9 |
| 4-way concurrent, aggregate | 44.5 | 56.4 | — |
| Cold prefill, 133k–300k | 898–938 tok/s² | 811–950² | ~941 |
| 30k replay speedup³ | **37×** (0.8 s warm) | 5.4× (5.6 s warm) | — (~97% cache-hit rate) |
| Acceptance (mixed prompts) | — | **3.11/step, 0.44 overall** | struct 1.0000 / prose ~0.32 |

¹ The two draft arms ran at different geometry, because the un-slot-shared drafter
caps the window (see the next section): no-drafter = 560k window, 1,418,876-token KV
pool; DFlash2 = 350k window, 372,233-token pool. Both: util 0.85, MNBT 3584, TP=2.
² Per-length no-drafter: 133k @898, 200k @938 (240k+ ran on the DFlash2 arm only:
240k @950, 300k @881; the no-drafter arm also passed a 500k fill in an earlier probe
at ~900). All numbers are single cold runs.
³ Speedup = cold prefill time ÷ immediate same-prompt replay time. The production
column reports a different metric (its measured cache-hit rate); a production replay
speedup was not measured in this session. DFlash2-arm replays at ≥200k read ~1× and
are **not** clean cache measurements — at the 350k geometry the 372k-token pool
cannot hold a ≥200k fill twice, so those replays are pool-bound; the 30k pair above
is the honest spec-vs-cache signal.

What the battery actually ran: needles retrieved at 8k, 30k, and inside every fill
(133/200/240/300k on the spec arm; 133/200/500k on the control) — 9 of 9 retrievals
passed; the 4-way concurrency probe completed with each lane's tagged response intact
(no response mixing observed in this run). Structured decode at 60.5 is 83% of
production on a cold-JIT boot with no warmup sweep.

On acceptance: the curve by position (0.77/0.59/0.46/0.38/0.33/0.30/0.27) meets or
beats published NVFP4-target curves at every tail position. This is an observation
from one configuration, not an isolated A/B of weight formats — but it is consistent
with the external evidence (see the watchlist) that draft acceptance is not the place
weight quantization hurts.

## What the official tree does NOT give you

The DFlash2 feature itself is fork-only, and on top of it three fork patches remain
load-bearing:

1. **The exact-fit KV slot-share.** Upstream charges the drafter's sliding-window
   layers full-length in the KV fit check — about 5.7 GiB at a 560k window — which
   caps the servable window near 358k on this hardware. The fork's slot-share
   (drafter pages riding the MLA pages) is what makes a 1M window coexist with
   speculative decoding at all.
2. **The prefix-cache fixes for speculative decoding.** With the drafter on, the 30k
   replay speedup dropped 37× → 5.4×. The fork fixes this class with a runtime
   overlay applied at container start; upstream PR #54163 (open, not merged, not
   tested here — its author reports validating it on an L20) is the designated
   successor.
3. **The xgrammar termination backports.** Production's 7-for-7 structured acceptance
   was measured only with them; the draft shows the pre-backport profile. (Matched
   A/B isolating the backports alone was not run; treat "the last ~17% of structured
   throughput" as attribution, not measurement.)

The DFlash2 port is eight files, all extractable from the fork image and
bind-mountable for iteration: `qwen3_dflash2.py` (drafter model), `dflash2/`
(speculator package), one registry line, the speculator routing hook in
`spec_decode/__init__.py`, the draft-KV-dtype guard in `dflash/utils.py` (dense
DFlash2 attention cannot ride `fp8_ds_mla` on SM121), subclass hooks in
`qwen3_dflash.py`, the glm5next aux-hidden-state capture (upstream `glm5next` still
has no `SupportsEagle3` — issue #54451), and `kv_cache_utils.py` (the drafter-group
machinery). When porting the glm5next file, keep the day-0 tree's #53969
`buffer_width` line — the fork's predates that fix.

## Traps we hit so you don't have to

- **Run `--no-async-scheduling` on this stack.** The tri-state default resolves to
  enabled. What we demonstrated on the draft: with the drafter's sliding-window
  layers on board, async inflated the KV fit check by ~6 GiB (the doubled
  SWA-family reservation, vLLM #47728 class) and cost three boot rounds to isolate.
  We did not reproduce a failure in the no-drafter launch — but production runs
  async-off after measuring the same reservation class on the fork at 262k, so
  carrying the flag everywhere is our operational recommendation, not a demonstrated
  requirement for every mode.
- **The snapshot's `index_topk=2048` is a pre-fix value.** With #53969's validator in
  the tree, boot with `--hf-overrides '{"index_topk":2044}'` (the official value);
  2048 + the kpool tail overflows the kernel's 2048-wide buffer arithmetic.
- **Public exllamav3 v1.4.4 reads config attributes the fork's older bundle didn't.**
  The plugin's namespace stub (which fakes `exllamav3.model.config` so `LinearEXL3`
  imports without flash_attn) must provide `NullConfig` whose attribute chains
  resolve to a falsy object — `LinearEXL3.forward` reads
  `config.infer_params.no_reconstruct` two levels deep, so a plain `None` one level
  up raises. Minimal working stub:

  ```python
  class _NullAttr:
      __slots__ = ()
      def __getattr__(self, name):
          return _NULL_ATTR
      def __bool__(self):
          return False

  _NULL_ATTR = _NullAttr()
  config.NullConfig = type(
      "NullConfig", (), {"__getattr__": lambda self, name: _NULL_ATTR}
  )
  ```
- **Multi-node quoting:** JSON args with commas inside braces get brace-expanded by
  the remote shell if you pass them through `ssh "docker run ..."` (our
  `--speculative-config` arrived as fragments; `--hf-overrides` survived only because
  it has no comma). Push the worker's launch command as a script file instead.
- **The worker rank needs `--headless`.** Without it, rank 1 builds a full engine core
  and dies at KV-cache init with "collective_rpc should not be called on follower node".
- **`enable_thinking: false` mis-renders on this image.** Request shape:
  `POST /v1/chat/completions` with
  `"chat_template_kwargs": {"enable_thinking": false}`, reasoning parser `glm45`,
  the chat template from the model snapshot above. Result: the model reasons anyway
  and the parser cannot strip it, so raw chain-of-thought floods `content`. Default
  and `enable_thinking: true` behave correctly (clean `content`, CoT in
  `message.reasoning`). Interim rule until root-caused: serve thinking-on only — do
  not expose this image to clients that send `enable_thinking: false`.

## Rebase watchlist (same-day upstream sweep; external sources, not measured here)

- **#52560 (merged Aug 22) broke DFlash2 draft loading on current main**
  (#53428/#53612) — any rebase past it must carry the re-fix.
- **#54163** (open): the upstream replacement for the fork's prefix-hit overlay;
  author-validated on an L20 per the PR thread, unverified by us.
- **#54282** (merged): draft/target gumbel stream decoupling on the V2 runner —
  cherry-pick candidate for temp-1.0 serving.
- **Keep the drafter BF16** until #51581/#52883/#53122 land — quantized-drafter
  weight handling bypasses quant dispatch today.
- In the field reports we reviewed (FP8/NVFP4/INT4 targets, one Q3→Q5 precision
  ladder, one llama.cpp draft-KV bug), no acceptance penalty was attributable to
  target weight quantization; the collapses on record traced to engine bugs. Our
  EXL3 numbers are consistent with that pattern.
