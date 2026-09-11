# 15 — Task 34 arm: FlashKDA fused chunked prefill (vLLM #55737)

**Status 2026-09-10: REVERTED — the numeric parity gate FAILED.** The arm was
wired, deployed to the cluster and put through the mandatory correctness gate;
the gate found a fatal arity defect and, once that was fixed, a numeric
divergence from the Triton path. The cluster deployment was reverted to the
exact bytes production runs, production was never restarted, and the A/B window
was **not** run. Receipt: `local/task34-parity-gate-20260910.txt`.

Do not arm `GLM53_KDA_PREFILL_BACKEND=flashkda`. The overlay is retained as the
port under study, not as a candidate.

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
| `torch.ops._flashkda_C.fwd` reproduces the Triton chunk path | **NO — see §5** |

The upstream gate is `capability.major in (9, 10, 12) and head_dim == 128 and
dtype == torch.bfloat16 and lower_bound is not None`. GB10 is `sm_121` → major
12, so the arm would engage. Note what that gate does and does not establish: it
is a *selection* predicate. The extension being importable, and its workspace
allocator returning a plausible size, says nothing about whether its output
matches the path it replaces — which is what the parity gate in §5 tests.

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

- `tests/test_flashkda_prefill_patch.py` — 37 CPU tests: applies and compiles,
  idempotent by marker, helper lands before the class, `_flashkda_prefill` is a
  class method, the Triton call survives re-indented, the selection predicate is
  restated, **each of the three anchors fails closed when drifted** (plus an
  ambiguous-anchor case and the missing-target case), and — added after §5 — the
  fused call's **arity is pinned to the deployed op's 14 arguments**, with the
  exact 16-argument defect rejected, a wrong-arity template refused before any
  write, and a keyword rewrite not counted as positional.
- Same assertions run against the **real deployed `kda.py`** when the gitignored
  live-image dump is present (`tests/fixtures/live-image-vllm/`).
- **Cluster smoke (in-container, on a temporary copy — production untouched):**
  unarmed sha256 `ec090aab…` unchanged; armed → parses, 6 markers; re-run
  idempotent; the production file stayed `ec090aab…`. The armed hash has moved
  twice as the port changed: `02b234e9…` → `a4bdc543…` when the review rounds
  added the marker-requires-completeness and class-boundary checks, then
  `a4bdc543…` → **`58ff323c…`** when §5 removed the two spurious arguments.
  `58ff323c…` is the current armed hash and was re-verified on the cluster
  against the deployed `kda.py`: 14 positional arguments to `_flashkda_C.fwd`,
  zero keywords, `py_compile` OK, idempotent, real file `ec090aab…` before and
  after. Note that the *unarmed* hash is unchanged in all of this — the fix only
  affects what the arm writes, which is why it never touched production.

## 4. Pre-registered A/B contract (registered, never executed)

Kept verbatim as registered. It was **not** run: §5's correctness gate — listed
here as required — failed first, which by the contract's own terms ends the
evaluation. No sample in this design was collected.

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
| Boot invocation in a throwaway container, knob `flashkda` | `a4bdc543…` (now `58ff323c…` after §5), 6 markers, `py_compile` OK, re-run idempotent |
| Live production during all of the above | `ec090aab…` unchanged, container up, `/health` 200 on :8000 |

The default is `triton` (the stock spelling), so **no `.env` change is required
to deploy this**: mounting the overlay cannot alter production by itself.

Deliberately **not** added to `prod-start.sh`'s JIT shape hash. The hash guards
the persistent Triton cache against changed *launch parameters*; this arm
replaces the prefill call and adds no Triton specialization, so hashing it would
force a full cache wipe on every arm switch for no benefit. Record the JIT stamp
before/after the window regardless, as below.

## 5. The correctness gate — FAILED (2026-09-10)

Run **without stopping production**: the GPU is reachable alongside the serving
container, so the gate was executed in throwaway containers from the deployed
image while `glm53-exl3-head` stayed up. Harness:
`local/flashkda-parity-check.py`; full numbers in
`local/task34-parity-gate-20260910.txt`.

The harness mirrors both call sites exactly — the candidate invocation is
`overlay/patch_flashkda_prefill.py::_flashkda_prefill` verbatim, the reference is
`glm5next/nvidia/kda.py`'s live prefill call verbatim — so it answers the A/B
question directly: what production would compute if armed, versus what it
computes now.

**Finding 1 — fatal arity defect.** The overlay passed **16** positional
arguments to `torch.ops._flashkda_C.fwd`; the deployed op declares **14**
(`q, k, v, g, beta, scale, out, workspace, A_log, dt_bias, lower_bound,
initial_state, final_state, cu_seqlens`). Arming raised
`RuntimeError: expected at most 14 argument(s) but received 16` at the first
prefill — the arm could not have served a single request. The first 14 arguments
matched the declaration in order, so the defect was two spurious trailing
`None`s. This is the failure mode the gate exists for: the earlier cluster smoke
test only proved the patched file *parses*, never that the op accepts the call.

Fixed in the overlay; `EXPECTED_FWD_ARITY = 14` now pins the emitted call, so
`apply_to` refuses a template with the wrong count and `is_complete` rejects an
installed file with one.

**Finding 2 — numeric divergence.** With the arity corrected the fused kernel
does not reproduce the Triton path. At `tokens=512, heads=32, head_dim=128,
lower_bound=-5.0`:

| | max abs diff | reference peak | reference mean abs | candidate mean abs | Pearson |
|---|---|---|---|---|---|
| `out` | 0.0735474 | 0.0722656 | 0.00314 | **2.095e-05** | **+0.008** |
| `final_state` | 1.6172 | 1.62175 | 0.03503 | **1.580e-04** | **−0.063** |

The candidate's mean magnitude is ~150× below the reference's and its output is
uncorrelated with it. The max difference exceeds the reference's own peak
because the candidate is near-zero almost everywhere: in a `T=256` probe every
one of the 256 token rows had mean `|.|` ≈ 2e-5 against the reference's ≈ 2.7e-3.
The kernel *does* run and *does* write — with both buffers pre-filled to a
sentinel, 100% of both were overwritten and 256/256 token positions written — so
this is a wrong result, not a skipped kernel or a stride mismatch.

Every variant tried also failed: kimi_k3's own Triton kernel as the reference;
`tokens=64`; pre-normalized q/k (byte-identical, so the kernel normalizes
internally); bf16 pre-sigmoided beta; a zeroed gate (`g = dt_bias = A_log = 0`).
A `lower_bound` sweep of {−1,−2,−3,−5} gave ratios of 130/116/100/87 — it does
**not** track `exp(lower_bound)`, so the factor is not a mis-applied bound.

Conventions ruled out, each by direct test: `beta` (the kernel requires bf16 and
rejects fp32: `flash_kda.cpp:55, beta must be bfloat16` — the overlay's raw-bf16
convention is right), `A_log` (the kernel requires `[H]` and rejects `(1,1,H,1)`,
`(H,1)`, `(H,1,1,1)`: `flash_kda.cpp:109, A_log must be [H]` — the overlay's
`.view(-1)` is the correct adaptation of GLM's 4-D parameter), `scale` (both use
`head_dim ** -0.5`), `l2norm`, and layout.

**Root cause not isolated.** The structural lead: the fused kernel is paired, in
this fork, with a *different* Triton kernel than GLM's.
`vllm/models/kimi_k3/nvidia/kda.py:180` is the only `_flashkda_prefill` in the
image; its Triton branch calls the kimi_k3 copy
(`chunk_kda_with_fused_gate(..., raw_beta, ..., lower_bound=None)`, no
`safe_gate`), while GLM calls the `vllm.third_party` copy (`..., beta, ...,
safe_gate, lower_bound=-5.0`), which takes **pre-sigmoided** beta. The two Triton
kernels are not the same kernel. Their outputs on these inputs are nearly
identical to each other (`ref_abs_max` 0.0722656 vs 0.0722656), which places the
mismatch in the fused op's own input expectations rather than in the choice of
reference. Isolating it needs the kernel source; the image ships only
`_flashkda_C.abi3.so`, and the error strings name a build-time path
(`/workspace/.deps/flashkda-src/csrc/flash_kda.cpp`) absent from the image.

## 6. Decision, and the cluster state

**REVERT.** Per §4's pre-registered contract, a divergence is a correctness
finding, not a tuning result: with the fused output uncorrelated with
production's, a cold-prefill comparison would measure a different computation.
The A/B window was therefore not started and **no speed claim is made**.

| | sha256 (first 16) |
|---|---|
| `start.sh` before task 34 (byte-identical to repo `6d071e4`) | `560a7ed5b8be5243` |
| `start.sh` as deployed for the window | `9abb832a567b5c86` |
| `overlay/patch_flashkda_prefill.py` as deployed | `a38a0ad4f6ec8f09` |
| backup retained: `start.sh.bak-20260910-task34` | `560a7ed5b8be5243` |
| **`start.sh` after the revert** | **`560a7ed5b8be5243`** |

The revert restored `start.sh` from the backup and removed the deployed overlay
file, returning the kit to the exact bytes the running container was started
from (`bash -n` clean, overlay absent). The two had to move together: the wired
launcher's `-f` preflight makes the overlay file mandatory, so removing only the
overlay would abort the next boot. Production was never restarted — container
`glm53-exl3-head` stayed up (21 h at the end of the run), `/health` 200 on
:8000, `kda.py` `ec090aab…` throughout.

## 7. Explicitly not done

- No production restart, no armed boot, no A/B measurement.
- No e2e prefill/acceptance/toolcall measurement — the numeric gate failed first.
- Root cause of the divergence was not isolated (needs the kernel source).
- #55736 and #55738 were not attempted; #55736's files do not exist in this fork
  and the whole lineage migration is its own multi-component window.
