# Bounded resident-tail scheduling

Adopted on 2026-10-03 in production on both ranks with the existing
`glm53-selfbuild:e3-armc-shm50` image. The arm is `GLM53_RESIDENT_TAIL=1` in
the runtime `.env` ledger; rollback is a later `GLM53_RESIDENT_TAIL=0` line.
The experiment changed only this Python scheduler overlay.

The patch protects one narrow case: a fresh request with at least one cached
3584-token page and at most 3584 uncached prompt tokens. If that tail is already
running, or is waiting while a sequence slot is free, the scheduler can defer a
cold running prefill for up to four steps. It removes the 1792-token
long-prefill cap from the qualifying tail so that tail can use the available
step budget. It does not defer decode, does not make a cold request into a
tail, does not prevent two cold prefills from sharing, and does not pin KV.
The scheduler patch runs after the prefix-admit overlay.

This is a targeted TTFT experiment. The upstream vLLM tuning guide describes
chunked prefill as prioritizing decode work before using remaining token budget
for prefill, with smaller batches favoring decode latency and larger batches
favoring time to first token. The resident-tail arm keeps that general policy
and changes priority only for a measured, already-cached short remainder; its
step cap bounds the cost imposed on a cold prefill. See the [vLLM optimization
guide](https://docs.vllm.ai/en/v0.19.1/configuration/optimization/).

## Gate and results

The control boot used the same image and controls, with prefix admit on and
cache yield off. Cache races used a 7,030-token follow-up after warming a
6,274-token prompt; the engine cache counter reported 3,584 cached tokens, so
the uncached tail was 3,446 tokens. Decode tests used five measured runs,
temperature 0, thinking off, and 200 output tokens.

| Measure | Flag off | Resident tail | Gate | Result |
|---|---:|---:|---:|---|
| Cached follow-up wall, two-race median | 6.723 s | 3.043 s | ≥1.0 s faster | Pass, 3.680 s faster |
| Follow-up cached tokens / uncached tail | 3,584 / 3,446 | 3,584 / 3,446 | hit ≥3,584; tail ≤3,584 | Pass |
| Cold request wall, race 1 / 2 | 7.647 / 7.837 s | 7.639 / 7.671 s | each ≤2× control pair | Pass |
| Structured decode median | 72.696 tok/s | 74.260 tok/s | ≥70.515 tok/s | Pass, +2.15%; accept 1.000 / 7.000 |
| Hashmap prose decode median | 31.589 tok/s | 34.482 tok/s | ≥30.642 tok/s | Pass, +9.16% |
| Hard essay decode median | 26.216 tok/s | 27.300 tok/s | ≥25.429 tok/s | Pass, +4.14% |
| NaN / health / KV pool | none / 200 / 567 blocks | none / 200 / 567 blocks | none / 200 / 567 blocks | Pass |

The head engine log recorded three `[glm53-resident-tail] defer` events in
each race. The overlay applied on both ranks. Local patch tests passed 13/13.
Raw run receipts are in `local/resident-tail-20261003/` on spark1; the `.env`
backup is `.env.bak-resident-tail-20261003`.
