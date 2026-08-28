# Known issues

## 1. Co-batched prefill inserts nothing into the prefix cache (open, upstream)

**Image:** `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a…`
(vLLM `0.1.dev20051+g487ecf187`). **Status:** open as of 2026-08-28.

A request whose prefill overlaps any other in-flight request — including one that is
only decoding — gets **zero cache retention**. Isolation data (all with MNBT=3584,
async off, identical prompts, metrics deltas from `/metrics`):

| Scenario | Hit on replay |
|---|---|
| 30k solo, replayed solo | 25,088 tok (7/8 pages) ✓ |
| 100k solo, replayed solo | 93,184 tok (26/27 pages, 7 s vs 110 s) ✓ |
| 4×60k concurrent, 3 rounds | **0 across all rounds** ✗ |
| burst prompt replayed SOLO afterwards | **0** → insertion (not lookup) is the broken side |
| B prefilled during A's decode, B replayed | **0** ✗ |
| identical pair fired concurrently | second hits the first's live blocks ✓ (in-flight sharing works) |

Consequences for agentic serving: sequential sessions hit 84–93%; concurrent bursts
read exactly 0% savings and the engine degenerates to full re-prefill of 50–200k
tokens per turn.

**Mitigations:** client-side concurrency ~1 for long-prompt agents; sessions ≤ ~150k
(compact around 110k). Server-side `MAX_NUM_SEQS=1` would force correctness at the
cost of serializing decode for every client — we chose client discipline instead.

## 2. Requests ≳ 200k prefill fine but are not retained

The pool is 89 pages (318,640 tokens). A 200k request re-prefills from scratch every
time. Working as sized — noted so nobody chases it as a bug.

## 3. Fit-check refusal at 17.51 GiB when experimenting with MNBT

If you raise MNBT (or re-enable async) and the engine refuses to boot with
"To serve at least one request … 17.51 GiB KV cache is needed", that is the
async double-count described in `docs/02-parameters.md` — not a real memory shortage.
Check `--no-async-scheduling` is present before touching the pin or max-len.

## 4. Bench coherence checker false-negative

`tests/bench_decode.py`'s coherence probe marks "9.9 vs 9.11" wrong even when the
model answers correctly ("9.9 is greater") — string-match artifact, not a model
regression. Read the text, not just the flag.
