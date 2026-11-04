# Mixed prefill/decode measurement preparation

This supplementary probe supports candidates 07/08 (chunk size) and 22
(solo/mixed scheduling). It has passed local HTTP/fault tests, not a live
cluster feasibility run. It changes no launcher setting and makes no adoption
decision. The frozen full speed matrix remains mandatory and unchanged.

The probe starts one or three short code requests, tokenized with explicit
`reasoning_effort=max`. After every request has delivered at least 32 generated
tokens and is still active, it submits a unique random-token prefill of 16k,
64k or 128k tokens with one output token. Short requests use a fixed 2048-token
generated budget, including reasoning. Three repetitions give 18 measured
cells. Separate 32k-prefill/256-decode warmups cover each decoder count; 32k
allows four chunks even in the planned 8192-token arm. Prompt namespaces differ
across warmups, cells and repetitions and match across future paired arms.

This supplemental grid stays within the four-sequence admission cap, leaving
a slot for the long request. It does not replace the required c1/2/4/8 full
matrix, including its above-cap points and 200-generated-token decode budget.

## Evidence and rejection rules

All request starts, ends and cumulative token-progress updates use one
monotonic origin per cell. Receipts retain prompt hashes, usage, finish state,
metric snapshots and incomplete-stream errors. Bundles may contain multiple
tokens or empty visible text; event counts and text length never replace usage.

A valid mixed cell requires each short request to show progress before long
submission, at least two updates inside the long request's start-to-first-update
interval, and progress after that interval. Queued-only, already-finished and
insufficiently observed overlaps fail. This is client-observed request overlap,
not proof of GPU co-execution or a shared engine batch. Effective scheduler
configuration and any required engine trace remain separate activation evidence.

Reported quantities are:

- Long TTFT and prompt tokens / TTFT, including observed scheduling delay.
- Per-decoder token-count increases observed inside the long-prefill interval
  divided by its duration. Within-bundle token boundaries are unknown.
- P95 and maximum gaps between consecutive usage updates intersecting that
  interval. Crossing gaps retain their full duration; these are not labeled
  individual-token latencies.
- Whole-cell speculative counter deltas/rates, including short decodes and
  the long request's one output token. These are not per-request rates.

Running/waiting gauges must be zero at cell boundaries. Successful-request
delta must equal the submitted count; prefix-hit delta must be zero. Missing,
nonfinite or reset counters, absent drafts and incomplete budgets invalidate
the cell. These guards support an isolated window; they do not replace service
isolation. Failed cells stop the grid and are retained atomically. Existing
receipt files cannot be overwritten.

## Future feasibility run

Run only in a separately recorded serving window on the head, with checkpoint,
image/recipe identity and actual KV capacity verified, and no competing build
or measurement. Preserve the working script's relative layout: it imports
`scripts/speed_matrix.py` helpers. Both hashes are stored in the receipt and
must be frozen before staging. The full-matrix helper is not modified.

```bash
uv run --frozen --no-dev experiments/mixed-prefill/measure.py \
  --base http://127.0.0.1:8000 --out local/mixed-prefill-pilot.json
```

The first live run is a feasibility pilot, not a performance gate. Confirm
timely metric publication, maximum-reasoning tokenization and overlap in all
18 cells. If the decode budget ends too early or observations are too sparse,
retain the failed pilot and freeze a revised common protocol before any
treatment comparison. Do not weaken validity checks after candidate results.
Register exact scheduler sub-arms, thresholds and rollback separately.

## Research and validation

Exa review used primary [vLLM tuning guidance](https://docs.vllm.ai/en/stable/configuration/optimization/),
[metrics documentation](https://docs.vllm.ai/en/stable/usage/metrics/) and the
[chunked-prefill RFC](https://github.com/vllm-project/vllm/issues/3130).
The pinned vLLM metrics logger confirms request-success and occupancy metrics.
The existing scheduler switch's prerequisites remain in the
[source review](../../docs/2026-11-03-prefill-policy-review.md).

TensorFold's `cuda_prompt_solo.zig` checks whether other streams are decoding
and guards state/ring boundaries. Exllamav3 reduces shared chunks and spaces
prefill rounds while decoding jobs remain active. The transferable technique
here is distinguishing observed solo and mixed workloads; neither engine or
quantization is substituted.

Local tests cover real HTTP coordination, maximum-reasoning tokenization,
bundled/empty-text updates, malformed/incomplete streams, false overlap, cache
hits, foreign request counts, receipt retention and admission-cap checks.
Live GLM qualification remains pending.
