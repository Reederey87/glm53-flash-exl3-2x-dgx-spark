# 15 — Task 34 arm: FlashKDA fused chunked prefill (vLLM #55737)

**Status 2026-09-10: PREPARED, WIRED and boot-validated; the production A/B
window has NOT been run.** No production restart, no `.env` change, no image
change. The candidate overlay is default-off and byte-neutral when unarmed, and
`start.sh` now mounts and executes it, so the arm can be armed by setting the
knob alone.

## 1. Why this arm, and what it is not

Prefill is this deployment's measured weak lane (240k cold ~1408 tok/s, 60k
~1454) and the upstream PR is prefill-weighted: **#55737** replaces the ~15-kernel
Triton `chunk_kda_with_fused_gate` path with `vllm._flashkda_C`, reporting
1.7–3.8× on the KDA layer and TTFT **−7.9% to −13.2%**.

It is a **port, not a rebase** (task 33's scoping fact). The deployed vLLM is the
divergent fork `487ecf187` (`0.1.dev20051`), which carries **no
`glm5next/nvidia/ops/third_party/`** tree — so #55736's
`glm5next/nvidia/ops/third_party/kda/{fused_recurrent,kernels}.py` edits have **no
counterpart file here**, and #55736 is not a rebase of the reverted task-30
overlay either (that patched the *vendored FLA* path, a different file). #55738
touches shared MLA backends, not the GLM model path, and is scoped separately.

This document covers **#55737 only**, on the one file the deployed fork actually
has: `vllm/models/glm5next/nvidia/kda.py` (653 lines).

## 2. Feasibility, verified on the deployed image

| Check | Result |
|---|---|
| `vllm._flashkda_C` importable in the serving container | **yes** |
| `torch.ops._flashkda_C.get_workspace_size(1792, 32, 4)` | **51,314,688 B** (runs on GB10) |
| `current_workspace_manager` importable (`vllm.v1.worker.workspace`) | **yes** |
| `GatedDeltaNetAttention.get_state_dtype` | **yes** |
| `self.head_dim` / `self.local_num_heads` present | **yes** (128 / 32) |
| upstream auto-select gate accepts this device | **yes** — SM12x, bf16, head_dim 128, bounded gate (`linear_lower_bound`) |

The upstream gate is `capability.major in (9, 10, 12) and head_dim == 128 and
dtype == torch.bfloat16 and lower_bound is not None`. GB10 is `sm_121` → major
12, so the arm would engage.

## 3. The port — `overlay/patch_flashkda_prefill.py`

Three insertion points, each asserted to match **exactly once**; a drifted anchor
aborts the boot rather than serving a half-patched KDA layer.

1. **module helper block** before `class Glm5NextLinearAttention(...)` — imports
   `current_workspace_manager`, adds `_glm53_flashkda_supported(...)` (the upstream
   selection predicate, restated);
2. **`__init__` tail** after `self._conv_state_dim_first = is_conv_state_dim_first()`
   — capability assertion, `import vllm._flashkda_C`, and the three
   `get_simultaneous` buffer specs (final state, workspace, spec-step output);
3. **the non-spec prefill call** — the single `chunk_kda_with_fused_gate(q=_rearr(q_ns), …)`
   becomes an `if self._glm53_flashkda_prefill:` dispatch, with the original Triton
   call preserved verbatim in the `else` branch.

Plus `_flashkda_prefill(...)`, appended as a method of the same class.

`GLM53_KDA_PREFILL_BACKEND` gates it: unset or `triton` leaves the file
**byte-identical**; `flashkda` patches; anything else is refused.

### Offline and cluster validation (done)

- `tests/test_flashkda_prefill_patch.py` — 16 CPU tests: applies and compiles,
  idempotent by marker, helper lands before the class, `_flashkda_prefill` is a
  class method, the Triton call survives re-indented, the selection predicate is
  restated, and **each of the three anchors fails closed when drifted** (plus an
  ambiguous-anchor case and the missing-target case).
- Same assertions run against the **real deployed `kda.py`** when the gitignored
  live-image dump is present (`tests/fixtures/live-image-vllm/`).
- **Cluster smoke (in-container, on a temporary copy — production untouched):**
  unarmed sha256 `ec090aab…` unchanged; armed → `a4bdc543…`, parses, 6 markers;
  re-run idempotent (`a4bdc543…`); the production file stayed `ec090aab…`. The
  armed hash moved from the earlier `02b234e9…` when the review rounds added the
  marker-requires-completeness and class-boundary checks; `a4bdc543…` is the
  current candidate and is the value re-verified through the wired boot path.

## 4. Pre-registered A/B contract (not yet run)

Follows docs/13 §6 in full. One independent variable: `GLM53_KDA_PREFILL_BACKEND`.

- **Control (A):** current stack — `glm53-selfbuild:e3-w3-zfill`,
  `EXL3_FAT_GROUPED=1`, last-wins `EXL3_TEMP_ROWS_FUSED=32`,
  `GLM53_ADAPTIVE_K=ema`, pin `7d74cdd`, C4.
- **Candidate (B):** A + `GLM53_KDA_PREFILL_BACKEND=flashkda`.
- **Observations:** ≥5 cold-prefill samples per arm at **60k and 240k**; prose
  (hashmap + hard essay) and structured ≥9 per arm; acceptance 7/7, serving 6/6,
  toolcall 23/23.
- **Adopt if:** ≥5% cold-prefill gain at 60k **and** 240k, with prose/structured
  and concurrency non-inferior, and per-request correctness on saved prompts.
- **Correctness gate (required, not optional):** the fused kernel must be
  numerically comparable to the Triton chunk path on the same inputs. This is a
  **prefill kernel replacement**, so parity is checked before any speed claim is
  read; a divergence is a correctness finding, not a tuning result.
- **Abort on:** any CUDA/Xid/IMA, output corruption, preemption regression,
  worker failure, or either-node MemFree < 2.5 GiB.
- **Rollback:** unset the knob through the guarded unit (the overlay then leaves
  `kda.py` byte-identical on the next boot). No image rebuild, no `.env` edit
  beyond the knob.
- **Both-node shape-cache:** the KDA layer is a config-shape change only if the
  workspace sizing moves; the JIT stamp must be recorded before/after regardless.

### Wiring — DONE and boot-validated (2026-09-10)

`start.sh` mounts and executes the overlay the way the other `patch_*.py`
overlays are wired: host variable (`FLASHKDA_PREFILL_PATCH_HOST`) → worker
`scp` → both ranks' `docker -v` → `python3 -S` at boot, after the FLA/KDA
overlays. Nine sites: host var, knob default, `validate` enum check, preflight
`-f` check, both boot heredocs, worker `scp`, both `docker -v` mounts, and the
`nccl_common` env forward (`-e GLM53_KDA_PREFILL_BACKEND`, which reaches both
ranks).

Validated, without touching production:

| Check | Result |
|---|---|
| `bash -n start.sh`, `shellcheck -S warning start.sh` | clean |
| Both generated inner scripts extracted and `bash -n` | rc=0, block present and `-f`-guarded in each |
| `./start.sh validate` in a scratch kit on spark1 | `triton`, `flashkda`, empty → "configuration valid"; `bogus` and `FLASHKDA` → rejected with the enum message |
| Boot invocation in a throwaway container, knob `triton` | `kda.py` byte-identical (`ec090aab…`), 0 markers |
| Boot invocation in a throwaway container, knob `flashkda` | `a4bdc543…`, 6 markers, `py_compile` OK, re-run idempotent |
| Live production during all of the above | `ec090aab…` unchanged, container up, `/health` 200 on :8000 |

The default is `triton` (the stock spelling), so **no `.env` change is required
to deploy this**: mounting the overlay cannot alter production by itself.

Deliberately **not** added to `prod-start.sh`'s JIT shape hash. The hash guards
the persistent Triton cache against changed *launch parameters*; this arm
replaces the prefill call and adds no Triton specialization, so hashing it would
force a full cache wipe on every arm switch for no benefit. Record the JIT stamp
before/after the window regardless, as below.

## 5. Explicitly not done

- No production restart, no armed boot, no A/B measurement.
- The launcher wiring landed and was boot-validated in throwaway containers;
  production was never restarted and its `kda.py` stayed `ec090aab…`.
- No numeric-parity run of `_flashkda_C.fwd` vs `chunk_kda_with_fused_gate`.
- #55736 and #55738 were not attempted; #55736's files do not exist in this fork
  and the whole lineage migration is its own multi-component window.
