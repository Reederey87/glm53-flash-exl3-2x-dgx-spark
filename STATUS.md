# Status

- 2026-11-03: Prepared a full-source matched build pair for the Int64 candidate,
  pinning CUDA/NCCL and sharing one vLLM wheel. Actual pinned-source control and
  patched hashes pass offline. No image build or GPU qualification yet.

- 2026-11-03: Harness `05c6023` passed 46 helper tests and 24 live requests
  across neighboring seeds at c4/c8, with zero prefix hits. Started the full
  explicit-max baseline as `glm53-ab-namespaced-base-20261103.service`;
  runtime receipts are in `local/ab-20261103/namespaced`. No MTP arm activated.

- 2026-11-03: Excluded the incomplete token-usage baseline after its c4 warmup
  correctly detected 8192 cached tokens. Additive run-seed/request-index arithmetic
  reused earlier prompts. Replaced it with deterministic SHA256 tuple namespaces;
  maximum reasoning and continuous token timing are unchanged. New baseline and
  candidate must share this harness, seed 2026110304, and zero-hit prefill gate.

- 2026-11-03: Explicit-max baseline correctly stopped on an empty-text token.
  Live replay identified token 119 with valid usage; serving stayed healthy.
  Replaced text-only arrival timing with validated continuous token usage,
  retaining arrival evidence and maximum effort. Failed partial remains
  excluded; corrected baseline uses fresh seed 2026110303. All 45 helper tests
  pass, including HTTP invisible-token and invalid-progress cases.

- 2026-11-03: Prepared an isolated Int64 dense-output backport and upstream
  large-scratch regression under `experiments/dense-int64`; static application
  and parsing pass, native build/GPU gates pending. Verified the existing
  dense TMA fence is already installed; added single-block RoCE as candidate 29.
- 2026-11-03: New Flare upstream `cc696a6b` resolves the original checkout's
  NVIDIA-format blocker: four actual declaration tests pass. Source uses the
  exact checkpoint and preserves projection formats. GPU/export/capacity gates
  remain open; no Flare deployment or weight conversion performed.

- 2026-11-03: Per user instruction, changed benchmark tokenization to explicit
  `reasoning_effort=max`, added JSON/Markdown mode and token-scope labels, and
  verified live tokens. Stopped and archived the incomplete implicit-max run;
  restarting both-arm measurement protocol with fresh seed 2026110302.
  Benchmark helper tests: 38 passed, including real HTTP request/receipt checks.

- 2026-11-03: Verified exact tokenizer contract during baseline: this GLM
  template ignores `enable_thinking=false`; all three flag variants tokenize
  identically at default maximum effort. Corrected the MTP3 gate's mode label
  before candidate measurement; kept both arms' frozen prompt/harness bytes.

- 2026-11-03: Started the full incumbent matrix as persistent user unit
  `glm53-ab-base-20261103.service`, using harness commit `318edd6`; private
  snapshots and incremental receipts are in runtime `local/ab-20261103`.
  FP4 watchdog temporarily stopped for the measurement window. MTP3 recipe
  patch frozen separately; no serving treatment applied yet.
- 2026-11-03: Verified direct worker access with `ssh -o HostName=10.0.0.35
  spark2`. Confirmed incumbent already contains the DSA barrier fix; updated
  upstream vLLM lacks its B12X/CSF loader files. Recorded native adaptive-MTP
  controls and the mixed-prefill scheduling gate in compatibility findings.

- 2026-11-03: Direct access restored via `spark1`; worker reached over
  `192.168.177.11` (LAN address `10.0.0.35`). Incumbent cold boot healthy at
  1M context / 2,350,210 KV tokens; exact NVIDIA checkpoint revision verified.
- 2026-11-03: Reconciled the prior November 1 reverted image window and
  prepared real-text decode, per-cell warmup, prompt hashes and incremental
  receipts. Local suite: 73 passed. MTP3 trial preregistered in
  `docs/gates/2026-11-03-mtp3.md`; no serving treatment applied yet.

- 2026-11-02: Verified public FP4-Local metadata at `76348ddb...` against
  Flare's actual declaration functions: FP8_PB_WO and MXFP8 reject. Recorded
  separate split-K, indexer-barrier and export-state qualification gates in
  `docs/2026-11-02-compatibility-findings.md`; no GPU tests or runtime changes.

- 2026-11-02: Prepared an isolated NVFP4 improvement worktree with the
  repository/Exa audit and ordered candidates in
  `docs/2026-11-02-improvement-queue.md`. Direct SSH is unavailable; no live
  configuration, image, checkpoint, driver or kernel changed.
- 2026-11-02: Hardened `scripts/speed_matrix.py` before A/B testing: require
  complete SSE/token usage, reject reset/missing speculative counters and
  cached prefill, retain request results, and use three distinct seeded reps.
  Cluster measurements and PR delivery remain pending access.

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
- 2026-10-17: Exposed the loopback model to the tailnet because Tailscale
  Serve is disabled on this tailnet. `vllm-glm53fp4-tailscale.service` binds
  only `100.125.23.98:8000` and forwards to `127.0.0.1:8000`; health, models,
  chat (`TAILNET_OK`), and SSE were verified through the Tailscale IP. The
  other Mac uses `http://100.125.23.98:8000/v1`; tailnet ACL is the remaining
  gate. Receipt: `docs/receipts/2026-10-17-tailscale-exposure.md`.
- 2026-10-17: Installed and enabled `vllm-glm53fp4.service` and the
  `vllm-glm53fp4-watchdog.timer`; the previous DSV boot units are disabled,
  linger is on, and a cold start through the unit reached `/health` 200 with
  the 2,078,358-token pool. First watchdog run exited 0 healthy; the health-down
  branch was exercised without restart. No physical reboot was performed.
  Receipt: `docs/receipts/2026-10-17-systemd-boot-watchdog.md`.
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
