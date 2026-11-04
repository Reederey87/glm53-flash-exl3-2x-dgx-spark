# Completed-answer quality preparation

Prepared before either activation arm. **Not run on the cluster.** This adds
the executable math component; code, long-context, tool/JSON and native
qualification remain required before candidate 18 can start. Passing this
component alone cannot qualify native A4 or authorize adoption.

## Research and local review

Exa research covered the canonical [GSM8K data and extraction convention](https://github.com/openai/grade-school-math),
[lm-evaluation-harness API guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md),
its [reasoning-token truncation issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/3044),
and [EvalPlus execution policy](https://github.com/evalplus/evalplus/blob/master/docs/execution.md).
The chat API supports generated-answer evaluation; it must not be treated as
a loglikelihood endpoint. Short generation budgets can leave reasoning models
without a final answer. EvalPlus is the stronger complementary code test;
code completion, timeout/OOM policy and sandboxed execution must be pinned
before that component runs. No generated code is executed by this runner.

After research, reviewed current local vLLM `tests/evals/gsm8k/gsm8k_eval.py`,
Exllamav3 `tests/e2e/test_moe_offload.py`, and TensorFold
`docs/recipes/cuda.md`'s NVFP4 precision comparison. vLLM provides a useful
full-split evaluation pattern, but its last-number parser is unsuitable for
this strict final-answer check. Exllamav3 and TensorFold motivate a stronger
teacher-forced comparison on identical positions, separately for prompt and
decode paths, with a measured numerical noise floor. That remains additional
coverage to prepare; top-k API logprobs cannot establish full-vocabulary KL.
Their thresholds and quality results for other models do not transfer to
this checkpoint. No alternate quant or serving engine is introduced.

## Frozen math component

Use the entire 1,319-row GSM8K test split, in source order, zero-shot, without
selecting easy or baseline-correct subsets. Source revision:
`3101c7d5072418e28b9008a6636bde82a006892c`.

- [Dataset bytes](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl):
  749,738 bytes; SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
- [MIT license](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE):
  SHA256 `86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b`.
- Protocol SHA256 `39aa687f1e094e47a2b9378ec2041674d8379f53dad999393861cf3f2bc9893a`.
- Ordered request-digest list SHA256 `46433ce1d78a4c8c9f0632a943bb6e274d700c041a912a7b04da58f2fb379f0a`.

The local `prepare` command validated all reference answers and these hashes;
it does not contact inference. The corpus and license are retained under
`local/quality-corpus-20261104/`, not copied into the public repository.

[`scripts/quality_gsm8k.py`](../../scripts/quality_gsm8k.py) freezes the full
request: explicit `reasoning_effort=max`, temperature 0, top_p 1, seed 2026110401,
8,192 generated tokens, concurrency 1, no stop strings or few-shot examples.
The user prompt contains the question and a fixed final-answer instruction;
reference solutions never enter the request. Run against the unchanged
NVIDIA revision, MTP5, context, KV pool and controls in the activation README.
Greedy sampling does not guarantee bit-identical GPU execution.

Score only a final standalone `#### <number>` line from assistant `content`.
Signed decimals and valid thousands separators are supported. Compare exact
decimal values; do not search reasoning fields or take the last incidental
number. This is a custom paired regression protocol, not a claim to reproduce
published GSM8K leaderboard settings or the vendor's quality evaluation.

Pre-registered component gate:

1. Both arms must include every ordered request exactly once, with identical
   payload digests, protocol and runner bytes. Distinct service invocations
   and separately qualified runtime receipts are required.
2. Require successful transport, the expected model, a single assistant
   choice, consistent positive token usage within budget, nonempty final
   content and `finish_reason=stop` for every request. Truncation, missing final
   content or malformed API receipts invalidate the component. They are
   retained and scored incorrect; never silently dropped or retried.
3. Incorrect or malformed numeric answers remain in the denominator. The
   baseline must score at least 95%; a weak baseline cannot validate the arm.
4. Report both accuracies, their difference, every lost and gained case ID,
   format failures and protocol faults. Require the one-sided 95% exact
   binomial upper bound on `baseline correct / candidate wrong` cases to be
   at most 1%. Gains do not cancel losses. This conservative tolerance is an
   experiment choice, not NVIDIA guidance. The statistical bound assumes
   exchangeable independent cases; this fixed public benchmark cannot prove
   quality on other distributions or absence of training contamination.

If the fixed budget is insufficient, preserve the failed component and
pre-register a new budget/protocol for **both** arms before running again.
Do not retry only failures, change extraction after seeing answers, or let
unfinished reasoning count as a successful answer. The 200-token speed
matrix remains unchanged and supplies no completed-answer accuracy evidence.

## Execution and evidence

Do not run during the current Int64 matrix. After the complete activation
qualification protocol is frozen and the cluster window is eligible, copy
the pinned corpus and committed runner to an isolated head-node directory.
Verify hashes before either arm. Capture and independently verify actual
image, checkpoint, selected activation plans, config/KV controls and service
invocation; the runtime-receipt hash recorded by the runner is provenance,
not a substitute for that verification.

Stop the finished matrix unit after retaining its terminal receipt: these
oneshot units remain active after completion. The quality runner refuses an
active/activating GLM matrix unit, checks the expected serving invocation
before and after each request, and uses loopback HTTP on spark1. Also ensure
there are no other clients or quality jobs; the unit check cannot discover
every possible external client. No service stop/start is performed by it.

```sh
uv run --frozen scripts/quality_gsm8k.py --dataset /path/to/gsm8k-test.jsonl prepare
uv run --frozen scripts/quality_gsm8k.py --dataset /path/to/gsm8k-test.jsonl run \
  --serving-invocation <verified-invocation-id> \
  --runtime-receipt /path/to/verified-runtime.json --out /new/arm-directory
uv run --frozen scripts/quality_gsm8k.py --dataset /path/to/gsm8k-test.jsonl compare \
  --control /path/to/control --candidate /path/to/candidate
```

Each new output directory records metadata, fsynced full response JSONL and
a terminal completion marker. An existing directory is never overwritten or
resumed. A transport error stops the run after preserving its receipt, with
no retry. Comparison recomputes answers from raw responses and rejects partial,
reordered, duplicated or mismatched requests. Publish comparison and manifests;
retain full raw responses for audit. No latency/speed decision comes from
this sequential quality run.

Local fault tests cover parsing, reasoning-only and truncated responses,
usage/model mismatches, loss bounds, loss/gain cancellation, weak baselines,
partial/duplicate/mismatched receipts, same-arm comparisons, active-matrix
refusal, transport failure without retry and overwrite refusal. Live protocol
validation, all remaining quality components and activation tests are pending.
