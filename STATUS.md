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
- 2026-10-17: Memory note for the 2M arm: ~4.9 GiB free / ~3.7 GiB available
  after boot. Do not co-locate builds, downloads, or another model on the pair
  while it is resident. Drop to `16G` (~1.956M) or the model-card `0.87`
  command (1.944M) if stability is more important than the last 3.9%.
