# DFlash2 Triton housekeep — cluster smoke receipts

Cluster: spark1 (head, `spark-a183`) + spark2 (worker, `192.168.177.11`),
2×DGX Spark GB10 / sm_121, CUDA 13.0, vLLM pin `487ecf187`.
Date: 2026-09-20 (EDT).
Branch: `overlay/dflash2-triton-housekeep-20260920` (base `origin/main` = `cbc06b79`).

## 1. Candidate under test

| item | value |
|---|---|
| image | `glm53-selfbuild:dflash2-triton-housekeep` |
| image id | `sha256:ec9653adaa36f7b1d95828ce1357dd7c3f15ee38b8428872799455adc6fa95d6` |
| size | 21,493,755,662 B (≈21.5 GiB) |
| base | `glm53-selfbuild:e3-pipeline-f1s8-reuse` (`sha256:1b866e26af4d…`) |
| layer | `Dockerfile.dflash2-overlay-layer` — Python-only, no CUDA rebuild |
| rollback | `.env` last-wins flip back to `IMAGE=glm53-selfbuild:e3-pipeline-f1s8-reuse` |

Image identity matched on both nodes (`ec9653adaa36f7…`).

## 2. The independent variable

The overlay layer's only behavioural delta over the live base image is the
DFlash2 draft module:

| file | base | candidate |
|---|---|---|
| `vllm/model_executor/models/qwen3_dflash2.py` | 10,347 B, md5 `196c5504a9c924483e594d7eae3f0d45` | 14,357 B, md5 `504c5744960ff0e252a6a413b3f4428a` |

Base = pure-PyTorch grouped conv (`_grouped_conv` present, no
`dflash2_grouped_conv`). Candidate = Triton `_dflash2_grouped_conv_kernel`
behind the `dflash2_grouped_conv` custom op, with the eager CPU fallback.

The installed module path is unchanged (`qwen3_dflash2.py`) because
`_SPECULATIVE_DECODING_MODELS["DFlash2DraftModel"]` maps there. Everything
else in the layer was already present in the base and the installers no-op'd
idempotently:

| overlay | base state | layer action |
|---|---|---|
| `patch_model_overrides.py` (`models/glm5next/nvidia/model.py`) | already present | no-op |
| `patch_glm_eagle3.py` (`models/glm5next/nvidia/model.py`) | already present (md5 identical base vs candidate) | no-op |
| `patch_dflash2.py` (registry path + `qwen3_dflash.py` + `registry.py` + `utils.py` + `__init__.py`) | pin patches present | re-installed kit module only |

Note: the EAGLE3 target is
`vllm/models/glm5next/nvidia/model.py` (under `vllm/models/`, NOT under
`vllm/model_executor/models/`).

## 3. Boot

`local/prod-start.sh` guarded boot: validate → stop → settle wait → start.
The `IMAGE` change altered the config-shape hash, so the persistent Triton /
TileLang JIT caches were wiped on **both** nodes (expected cold-JIT boot).

| event | value |
|---|---|
| unit `ActiveState` / `SubState` | `active` / `exited` (oneshot), `NRestarts=0` |
| `/health` 200 | 540 s after start |
| weight load | 120 shards, ~163.58 GiB checkpoint |
| DFlash2 CUDA graph capture | 12/12 FULL graphs, 32 s |
| post-ready warmup | 20/20 requests OK in 30 s |
| `spec` line | `spec=DFlash2 k=7 (incoai/GLM-5.3-Flash-DFlash2)` |
| launch | `--speculative-config {"method":"dflash","num_speculative_tokens":7,…}` |

## 4. Runtime identity (inside the running head container)

Raw output: `identity-probe.log` (`IDENTITY PASS`, 27/27 checks).

The probe was corrected after a review pass, because four of its original checks
reported FAIL on a *correct* install. Each was a probe bug, not a deployment
defect, and they are recorded here so the checks are not "fixed" back:

| original check | why it was wrong | correct check |
|---|---|---|
| "pin lacks `get_top_k_tokens`" | substring match tripped on the module docstring, which *names* the API it deliberately does not call | assert no executable `.get_top_k_tokens(` call |
| "`is_causal` override" | the override lives in the **parent** `qwen3_dflash.py`, not the drafter module | inspect the parent module |
| "class `DFlash2DraftModel`" | `DFlash2DraftModel` is an architecture **key**, not a class | assert the registry mapping |
| "glm5next EAGLE3" | the target is `vllm/models/glm5next/nvidia/model.py`, not `model_executor/models/glm5next.py` | inspect the real path |

The last one is the dangerous class: the original check inspected a path that
does not exist in this pin, so it could never have caught a broken EAGLE3
install. The corrected probe asserts each path exists before inspecting it, and
exits non-zero on any failure.

| check | result |
|---|---|
| `qwen3_dflash2.py` md5 == `/opt/glm53/dflash2_model.py` | PASS (`504c5744…`) |
| kit marker `# [glm53-dflash2]` | PASS |
| `_dflash2_grouped_conv_kernel` | PASS |
| `direct_register_custom_op(op_name="dflash2_grouped_conv")` | PASS |
| CUDA/eager branch on `hidden_states.is_cuda` | PASS |
| `torch.topk` candidates; no executable `.get_top_k_tokens(` / `draft_logits_spec` | PASS |
| classes `DFlash2Qwen3ForCausalLM`, `DFlash2Qwen3Model`, `DFlash2Qwen3DecoderLayer` | PASS |
| registry `DFlash2DraftModel` → `('qwen3_dflash2', 'DFlash2Qwen3ForCausalLM')` | PASS |
| custom op resolves | `vllm::dflash2_grouped_conv` |
| parent `qwen3_dflash.py`: `is_causal` / `decoder_layer_cls` | PASS |
| EAGLE3 `EagleModelMixin` / `SupportsEagle3` / `aux_hidden_state_layers` on `models/glm5next/nvidia/model.py` | PASS |
| `DFlash2Speculator` present | PASS |

Live EAGLE3 arming (this is the target-model aux-hidden interface DFlash2
consumes, not a second speculator):

```
[exl3_utils] Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)
[kv_cache_coordinator] eagle_group_ids=[6]
```

## 5. On-GB10 grouped-conv parity (new receipt)

Triton CUDA op vs the kit's eager CPU fallback vs an fp64 reference, run on
the GB10 inside the head container:

| case | rows × channels | dtype | cuda−eager | cuda−fp64 | rel vs fp64 |
|---|---|---|---|---|---|
| small | 64 × 512 | bf16 | 1.250e-01 | 3.116e-02 | 2.03e-03 |
| odd block | 100 × 768 | bf16 | 1.250e-01 | 4.959e-02 | 2.52e-03 |
| small | 64 × 512 | fp16 | 1.172e-02 | 3.903e-03 | 2.55e-04 |
| 1024-aligned | 256 × 1024 | bf16 | 1.250e-01 | 5.223e-02 | 2.58e-03 |
| large | 512 × 2048 | bf16 | 1.250e-01 | 6.214e-02 | 2.59e-03 |
| empty rows | 0 × 512 | bf16 | — | — | shape `(0,512)` OK |

`PARITY PASS`. All residuals are inside the bf16 rounding floor (2⁻⁸ ≈
3.9e-03 relative); the constant 0.125 cuda−eager delta is one bf16 ULP at the
working magnitude. This confirms the CUDA branch executes on sm_121 and the
math matches the reference. (Consistent with the recorded P3 note that the
CPU fallback is not bit-identical.)

## 6. Acceptance

`local/acceptance.sh`: **7 passed / 0 failed — ACCEPTANCE PASSED**

thinking reasoning+content · tool call (glm47, auto) · production sampling
temp 1.0 + thinking · vision tower · long-context needle ~32k
(`CORMORANT-8815` at 36,040 prompt tokens).

## 7. KV pool

| metric | value |
|---|---|
| `kv_cache_size_tokens` | 1,396,551 |
| `kv_cache_max_concurrency` | 1.396551724137931 (1.40×) |
| `num_gpu_blocks` | 567 |
| `kv_cache_memory_bytes` | 15,414,698,763 |

Identical to the recorded reuse baseline. DFlash2 drafter KV line unchanged
(`padded slot-share block=64 mla_page=2351104`). Indexer workspace:
`mode=rightsize index_kpool=4 reclaimed_mib=4909.5`.

## 8. Error triage

| node | tracebacks / CUDA errors / IMA / assertion |
|---|---|
| head | 0 |
| worker | 0 |

## 9. Decode lanes (same protocol as the recorded reuse baseline)

Warmup 3 runs/lane → settle → 9-run measured round, temp 0, thinking off,
`--max-tokens 200`. Baseline = recorded reuse-image window (2026-09-20).

| lane | baseline median (min–max) | candidate round 1 | candidate round 2 | delta |
|---|---|---|---|---|
| structured | 74.19 (63.49–74.67) | 73.61 (25.82–74.62) | 73.88 (72.08–74.30) | **−0.4%** |
| hashmap prose | 33.03 (28.10–35.43) | 29.20 (11.79–35.38) | 31.53 (28.11–35.15) | **−4.5%** |
| hard essay | 25.92 (24.46–27.20) | 25.69 (16.25–27.41) | — | **−0.9%** |

Acceptance ratio (accepted/step):

| lane | baseline | candidate r1 | candidate r2 |
|---|---|---|---|
| structured | 1.0 / 7.0 | 1.0 / 7.0 | 1.0 / 7.0 |
| hashmap | 0.551 | 0.527 | 0.517 |
| essay | 0.436 | 0.432 | — |

No NaN on any lane; hashmap coherent on both rounds.

**Interpretation.** The round-1 hashmap figure (29.20, min 11.79) was taken
immediately after the JIT-cache wipe and the JIT monitor recorded Triton
compilations during inference (`_rejection_kernel`, `_resample_kernel`,
`_topk_topp_kernel`, `_fused_q_kv_rmsnorm_kernel`, `rotary_kernel`). The
post-warm round-2 repeat recovered to 31.53 with min 28.11. Two same-image
facts bound the noise: the baseline's own two arms differed by 0.496 vs 0.551
acceptance on identical code, and the candidate's round-2 min (28.11) matches
the baseline min (28.10). All three lanes therefore sit inside the
boot-to-boot variance of the unchanged image.

**The change is throughput-neutral.** This was expected: the grouped conv is a
small elementwise op inside the drafter, so the Triton port is an upstream
parity/housekeeping change, not a performance lever. No speedup is claimed.

## 10. Memory

| node | MemFree while serving |
|---|---|
| head | 3,799,140 kB (≈3.62 GiB) |
| worker | 4,528,292 kB (≈4.32 GiB) |

Same steady-state posture as the pre-change production pair (page cache holds
the remainder; the 2.5 GiB floor was respected).

## 11. State at hand-off

Production is **running the candidate** on both nodes, health 200, watchdog
timer re-armed (`active`), `NRestarts=0`. Pre-arm `.env` backup:
`.env.bak-dflash2-housekeep-20260920T142543`.

Cluster validation for this candidate is **complete**; the publication gate
(commit → `gh pr create` → review) is still open and separate.

## 12. Open items

- Decode lanes measured once each way; no interleaved A/B against the base
  image on the same session.
- Kit's `tests/bench_decode.py` on the Spark is older than the local copy
  (no `--essay`); the local version was staged to `/tmp` for the essay lane
  and the kit copy was left untouched.
- `Dockerfile.dflash2-overlay-layer` was added after the reviewed manifest
  freeze (`f11003b3…`) as the smoke vehicle and is not in that manifest.
