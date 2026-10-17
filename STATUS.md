# Status

- 2026-10-17: Replaced the EXL3 kit with `nvidia/GLM-5.3-Flash-FP4-Local` on
  `eugr/spark-vllm-b12x:latest` (digest `sha256:03a956…`), driven by the pinned
  `eugr/spark-vllm-docker` launcher at `0bdc9c9`. The recipe pins a 17 GiB FP8
  KV pool, MTP-5, 500k context, TP2. All EXL3 code, Dockerfiles, overlays, and
  EXL3 receipts were removed from the tree; the old kit remains in git history.
- 2026-10-17: Deployed on spark-a183 + spark-7829. DeepSeek V4.1 Flash was
  stopped through its unit first; its rollback tree and units are intact.
  The checkpoint validated as 42 indexed files / 165.3 GiB on both nodes.
  Boot reports **2,078,358 FP8 KV tokens, 4.16× at 500k**; `/health` 200;
  smoke hard checks pass (tool call observed, reasoning parser observed,
  prefix-cache field unavailable -> WARN). Receipt:
  `docs/receipts/2026-10-17-fp4-bringup.md`.
- 2026-10-17: Speed matrix (unique no-cache prompts, 200-token decode):
  prefill aggregate ~2.2–2.3k tok/s from 8k to 128k and c=1–8; decode
  42.90/41.73/19.63/26.60 tok/s at c=1/2/4/8 (repeat 46.76/44.54/20.25/29.77).
  The c=4 dip reproduced; open. Receipt: `docs/receipts/2026-10-17-speed-matrix.md`.
- 2026-10-17: Fixed the false prefix-cache WARN. The cache was hitting
  100% on repeated 1k/4k/8k/16k prompts; `cached_tokens` was null because the
  recipe lacked `--enable-prompt-tokens-details`. The flag is now set, smoke
  falls back to `vllm:prefix_cache_hits_total`, and the probe reports
  `cached_tokens=6029` on the smoke pair. Receipt:
  `docs/receipts/2026-10-17-prefix-cache-fix.md`.
- 2026-10-17: Memory note for the 2M arm: ~4.9 GiB free / ~3.7 GiB available
  after boot. Do not co-locate builds, downloads, or another model on the pair
  while it is resident. Drop to `16G` (~1.956M) or the model-card `0.87`
  command (1.944M) if stability is more important than the last 3.9%.
