# Known issues

## 1. Co-batched prefill inserts nothing into the prefix cache — FIXED 2026-08-29

**Measured on:** the previously pulled image
(`ghcr.io/miaai-lab/…@sha256:9bb1557a…`, vLLM `0.1.dev20051+g487ecf187`); the
self-built image production runs since 2026-08-30 is the same lineage and vLLM
build, so every issue and fix here carries over unchanged. **Status: RESOLVED by
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`** (see `04-prefix-caching.md`, "The
retention fix") — the zero-insertion was a side effect of dense KDA retention
making cached pages unaffordable, not a scheduler defect. The isolation data below
is kept as the historical record and as the regression test set (re-run
`local/cache-burst.py --sessions 4 --ctx-tokens 60000 --rounds 3`; rounds 2–3 now
measure 95%).

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

**Mitigations:** client-side concurrency ~1 for long-prompt agents; compact sessions
around 300k (solo retention verified to ~340k on the 1M-window pool). Server-side
`MAX_NUM_SEQS=1` would force correctness at the cost of serializing decode for every
client — we chose client discipline instead.

## 2. Multi-session retention collapse under the 1M window — FIXED 2026-08-29

Since the slot-share + hybrid-prefix-hit merge that made 1M allocate, two sessions of
~68k+ each evicted each other to an exact 0% cross-session hit rate; only ~2×20k
coexisted. **Resolved by the same retention fix as issue 1**
(`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`): 2×68k now measures 97.8% cross-session
hits with solo retention held at 98%. The one-long-context-client operating rule is
provisionally retired (soak concurrent hits under real load first — see the #54199
caution in `04-prefix-caching.md`). The old ablation lever (pre-slot-share
`patch_glm5_drafter_group.py`) is obsolete.

## 3. Fit-check refusal at 17.51 GiB when experimenting with MNBT

If you raise MNBT (or re-enable async) and the engine refuses to boot with
"To serve at least one request … 17.51 GiB KV cache is needed", that is the
async double-count described in `docs/02-parameters.md` — not a real memory shortage.
Check `--no-async-scheduling` is present before touching the pin or max-len.

## 4. Bench coherence checker false-negative

`tests/bench_decode.py`'s coherence probe marks "9.9 vs 9.11" wrong even when the
model answers correctly ("9.9 is greater") — string-match artifact, not a model
regression. Read the text, not just the flag.
