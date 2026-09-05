# Rebase plan — riding GLM-5.3-Flash into vLLM mainline

> **2026-09-05 update:** #53906 merged on September 3; the source-rebase trigger
> below has fired. Mainline now resolves GLM-5.3-Flash, but still needs our EXL3
> integration and qualification of every retained overlay. This is preparation,
> not approval to rebase production. See [the current survey](13-upstream-review-20260905.md)
> for #54374, packed-MLA alignment, GDN/null-gap tests and proposed gates.
> The August hardware claims below are historical and superseded: SM121 lacks
> tcgen05/TMEM, not all FP4 warp MMA; verify native sm_121a code generation rather
> than assuming architecture-specific sm_120a binaries are portable.

> **Status (2026-08-30): the first stage of this plan was executed.** The day-0
> base was field-tested on the pair (`09-rebase-draft-test.md`) and production now
> runs an image this repo builds from it (`10-selfbuild-production.md`). What
> remains of this plan is the *forward* rebase — moving onto vLLM main once
> #53906 merges — for which the survey below is the map.

Researched 2026-08-29 (web survey + a full inventory of a current vllm-project/vllm
main checkout, `cacc429f62`). Goal: the next major release of this kit sits on
**official upstream GLM-5.3-Flash support** and carries only the layers upstream
will never take, instead of a 142-commit private fork delta.

## What changed upstream

- **vLLM PR #53906 — "[Model] add GLM-5.3-Flash support"** (ZJY0516 + zai-org) is
  open: the `Glm5Next` model, KDA layers, kpool indexer and MHC path headed for
  mainline. Requires FlashInfer ≥ 0.6.17. Official images already exist
  (`vllm/vllm-openai:glm53-flash{,-arm64-cu130,…}`, 2026-08-26 vintage — the
  arm64-cu130 one is this kit's current FROM-base).
- **vLLM PR #53969** — NoPE (`qk_rope_head_dim=0`) support on the SM120 sparse-MLA
  backend + effective-topk width validation: this deployment's own fixes,
  submitted upstream from the 2×DGX-Spark production lineage.
- **FlashInfer #4802/#4791** — a *native* SM120 NoPE sparse-MLA path (D_CKV=512,
  D_ROPE=0). When integrated, the zero-padding workaround retires and DSA-layer
  KV records drop from 656 B toward ~528 B/token — a pool win to re-measure.
- **SM121 hardware facts** (field-confirmed): SM121 lacks `cvt.e2m1x2` — NVFP4
  kernels can never compile for it (the EXL3 choice was structural, not taste);
  sm_120 cubins run on sm_121 via forward compatibility; platform support
  (`is_blackwell_class()`, GB10 MoE configs, TRITON_PTXAS_PATH) is merged (#31740).

## What current main already ships (verified in-tree, `cacc429f62`)

| Area | State |
|---|---|
| **DFlash/DFlash2** | First-class: `"dflash"` `SpeculativeMethod`, `DFlashProposer`, V1/V2 speculator split (`DFlash2Speculator`), `DFlash{arch}` draft-model naming convention, multi-KV-group handling. Newer than the fork's. ⚠ Trap: a DFlash2 draft must route to the **V2** speculator (`_is_dflash2_draft`) or it **silently degrades to DFlash1**. |
| **Sparse MLA on SM12x** | `FLASHINFER_MLA_SPARSE_SM120` backend upstream and **auto-selected for compute-major 12** — no fork patch, no `--attention-config`. Requires packed `fp8_ds_mla` + `index_topk=2048` (matches this config). `DEEPSEEK_V32_INDEXER` present with correct sm_120 gating (fp8 K-cache; mxfp4 refused). Assume the fork's SM12x sparse-MLA fixes are superseded (esp. by #51395). |
| **Sparse mamba retention** | Post-#52216: `CacheConfig.prefix_cache_retention_interval` (argument, default 0), env var only an optional override. `MambaManager.reachable_block_mask` native with the tri-state this kit relies on. ⚠ Trap: the semantics moved from env-only to argument — set `prefix_cache_retention_interval` **explicitly** at rebase or sparse retention can silently change meaning. |
| **Eagle-group scoping** | `is_eagle_group` annotation + coordinator-derived `eagle_group_ids` — this kit's `patch_hybrid_prefix_hit.py` intent, generic. Needs the drafter to declare `non_causal_multi_token_decode`. |
| **Parsers / video** | `glm45`/`glm47` reasoning+tool parser aliases and the `glm46v` video backend are upstream — the kit's parser/video mappings come free. |
| **KDA infrastructure** | Exists as Kimi-K3's: `GatedDeltaNetAttention` base, fused KDA decode CUDA kernels, external FlashKDA build, `kda_state_shape`. GLM's KDA layer class is absent, but the fork's KDA kernels are swap candidates (numerical A/B, not blind swap). |
| **GLM-family model base** | `Glm5Next` absent; `GlmMoeDsaForCausalLM` (GLM-5.2-shaped, MLA+indexer, no KDA) exists as a much closer re-port base. DeepSeek-V3.2 relocated to `vllm/models/deepseek_v32/{common,nvidia,amd}/`. |
| **EXL3** | **Completely absent** — no trellis quant of any kind. The one big fork-only component, forever. |

## Trigger

**Either** the official `glm53-flash-arm64-cu130` tag refreshes post-#53906-merge,
**or** #53906 merges to main (any later nightly then works as a source base). The
weekly `check-updates` monitor covers both signals. Do not self-rebase the current
fork — its 142 commits shrink to a thin overlay once #53906 lands.

## Overlay inventory — carry vs drop at rebase

| Layer | Verdict |
|---|---|
| EXL3 quant (`overlay/exl3.py`, aarch64 ext stub, model-overrides) | **CARRY — as an out-of-tree plugin**: upstream exports `register_quantization_config()`, so EXL3 registers without patching upstream files. This is the single biggest rebase-cost reduction available; do it even before the rebase. |
| DFlash2 drafter overlays | **REPLACE with upstream framework** + a thin `DFlash{arch}`-convention GLM draft model class; verify V2-speculator routing explicitly. |
| NoPE sparse-MLA padding + kpool SM120 fixes | **DROP** (superseded in main; re-measure KV with FlashInfer native NoPE when integrated). |
| `patch_xgrammar_termination.py` | **DROP** — #52805/#53046 merged upstream. |
| `patch_hybrid_prefix_hit.py` | **DROP** — native `is_eagle_group` annotation subsumes it (declare `non_causal_multi_token_decode` on the drafter spec). |
| `patch_glm5_drafter_group.py` (slot-share) | **REWRITE, not rebase** — #51704/#51718 changed the backend↔spec KV-packing interface; the pool math (1.40×) must be re-derived on the new base. |
| Retention env (`=0`) | **CONVERT** to explicit `prefix_cache_retention_interval` config; re-verify the multi-session fix (2×68k probe) under mainline semantics. |
| `patch_suppress_stops_in_reasoning.py`, `patch_scheduler_decode_floor.py` | **CARRY**, re-anchor (fail-closed patches refuse on drift — that is the signal to re-derive). |
| Long-prefill threshold (1792), loopback bind, MERGE_NICS=0, KV pin | **CARRY** — deployment policy, not engine code. The pin *value* must be re-derived (pool geometry changes with the base). |
| GLM KDA kernels (fork) | **A/B against upstream FlashKDA** — swap only on numerical + throughput parity. |

## Validation battery (no re-pin without all of it)

1. Fail-closed patch preflights (every carried overlay anchors or refuses).
2. Acceptance 7/7 · serving 6/6 · toolcall 23/23.
3. Decode gates vs standing baselines (structured 70.4 @ acceptance 1.0, prose
   29.5) — converge 3–4 passes (post-churn passes read low).
4. Cache battery: 2×68k ≥97%, 4×60k×3 ≥95%, solo 110k ≥98% — **mandatory** given
   the retention-semantics change.
5. Cold-prefill reference (~893 tok/s with the threshold; 941 without).
6. KV pool line read + pin re-derivation; loopback bind check (`ss -ltn`).
7. Concurrent-hit soak (vLLM #54199 class — GB10 IMA behind concurrent hits).
8. DFlash2-not-DFlash1 check: verify the V2 speculator engaged (spec-decode
   counters show k-position acceptance, not single-position).

## Release path

The current measured deployment (1M window, both cache bugs fixed, HOL relief)
ships as a tagged release of this kit. The rebase lands as the next major version
when the trigger fires and the battery is green.
