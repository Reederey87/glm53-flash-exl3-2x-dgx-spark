# Status

- 2026-11-03: Corrected control passed all 271 bounded native GPU cases on
  spark1, with no skips/missing cases. All 187 emitted B12X objects passed
  SM121a/identity inspection. Candidate, worker and serving comparisons remain
  pending; inference/watchdog remain off and production aliases unchanged.

- 2026-11-03: Corrected Int64 control `40f4c522...` built successfully after
  both-node memory gates. Only frontend changed versus the rejected control:
  1.32.1 to 1.31.0; DSL4.7, OS inventory and treatment bytes identical.
  The remaining cuSPARSELt warning is unchanged. Native GPU tests are pending.

- 2026-11-03: Exact-image cache recovery preserved identities and raised head
  MemFree from 76,428,296 to 87,620,872 KiB, still below the build gate.
  Prepared guarded advice for the identified idle BuildKit uv cache; 24
  cache guard/metadata tests pass. No corrected build or GPU run started.

- 2026-11-03: Prepared a common runner constraint for cuDNN frontend 1.31.0,
  preserving DSL 4.7 and all compiler-stage bytes. Added guarded, read-only
  cache advice for the completed control image; neither repair has run yet.
  Measurement/max-reasoning and recovery checks: 99 tests passed.

- 2026-11-03: Checkpointed control runner built successfully as `45ac705b...`.
  Final inventory exposed unpinned cuDNN frontend 1.32.1 requiring DSL >=4.8,
  conflicting with frozen 4.7. No serving selection or GPU qualification;
  preserve this image and repair the common dependency set before continuing.

- 2026-11-03: Control FlashInfer providers completed (2,920 outputs, 564
  libraries; three wheels retained). Build stopped before the runner image
  at the unchanged MemFree gate. Extended scoped recovery to the completed
  FlashInfer workspace; 12 helper tests pass. No arm image exists yet.

- 2026-11-03: Incumbent dependency check reports one cuSPARSELt platform-tag
  defect. Actual library is AArch64 and matches upstream's reported fixed-wheel
  payload hash; Torch pins the old version. Recorded as unresolved packaging
  evidence, without editing metadata or the frozen build.

- 2026-11-03: Candidate 22 source/snapshot review found that the solo-prefill
  switch needs an active decode-counting policy and smaller scheduled cap;
  baseline has neither. Recorded prerequisites and mixed-load evidence needed.
  No scheduling change or performance result is claimed.

- 2026-11-03: Prepared candidate 05's isolated MTP1 recipe patch and gate after
  Exa/local-source review. Actual temporary application changes exactly one
  line and matches the recorded hashes. No runtime change or GPU trial yet.

- 2026-11-03: Reviewed reported dependency alerts against the incumbent image:
  certifi 2026.7.22 and filelock 3.32.3. Existing build logs show newer versions;
  final-image inventory verification remains pending. No package/build change.

- 2026-11-03: Inspector `d888df7` is hash-verified on both nodes. Its real
  incumbent-object smoke passes without GPU access, preserving object/manifest
  bytes and producing matching SM121a resource/SASS reports. All 176 tests pass;
  changed-kernel GPU qualification still awaits the active source build.

- 2026-11-03: Found that direct cuobjdump misses CuTe cache-object fatbins.
  Added a read-only, symbol-bounded extractor with manifest/hash checks and
  retained resource/SASS reports; 17 rejection/identity tests pass. GPU tests
  and the running source build remain unchanged.

- 2026-11-03: Applied the verified explicit-SHA fetch repair to both prepared
  RoCE build generators. Regenerated contexts differ only in Dockerfile/manifest;
  kernel patches, qualification sources and gates are unchanged. No RoCE build ran.

- 2026-11-03: The continuation stopped at FlashInfer checkout: normal clone
  omitted the frozen commit. An isolated exact-SHA fetch succeeded and its
  dense kernel hash matches. Prepared explicit pin fetching in the build-context
  generator; source pins, treatment and native/serving gates remain unchanged.

- 2026-11-03: Scoped recovery raised head MemFree from 89,364,996 to
  101,452,696 KiB; wheel SHA unchanged. Both-node idle gate passed again.
  Hash-identical source build resumed in `dense-int64-2157b70-resume1/context`
  (invocation `809ff4c66070403f8ab94efc11b8a342`), now in FlashInfer phase.
  Original artifacts retained; inference remains off. All 159 local tests pass.

- 2026-11-03: The shared vLLM wheel exported successfully, then the source-build
  wrapper stopped at the next MemFree gate (88,979,216 KiB, below 90 GiB).
  Build unit is terminal; no serving image selected. Preparing file-scoped
  clean-cache recovery for the completed build layer, preserving all artifacts.

- 2026-11-03: Prepared the Int64 native-test runner with installed-source/GPU
  identity checks, fresh caches, strict skip/memcheck handling and retained
  native resource reports. Locked test-project hashes match on both nodes;
  GPU execution awaits the ongoing source build. All 157 repository tests,
  runner ShellCheck and syntax checks pass. Runner `f0ac844` is staged and
  hash-verified on both nodes; live low-memory refusal passed without a GPU
  container. Updated queue status.

- 2026-11-03: Int64 full-source build started on spark1 at 16:19:51 UTC as
  `glm53-build-dense-int64-20261103.service` after both-node snapshots and
  MemFree >90 GiB. Inference is stopped and watchdog disarmed for this window;
  original image/MTP5 recipe retained. Native qualification remains pending.

- 2026-11-03: Full MTP3 trial passes (96 cells + 40 warmups per arm); code
  decode medians improve 5.31–11.89% with no registered robust regression.
  MTP5 restored, all API/cache/KV/control checks pass, watchdog re-armed.
  Receipts: `docs/receipts/2026-11-03-mtp-depth/`; MTP3 adoption pending.
  Local validation: 145 tests, ShellCheck, Bash/Python syntax all pass.

- 2026-11-03: Clarified reciprocal-TPOT and normalized aggregate labels; added
  supplementary bundle-aware/full-cell rates from retained arrivals. Sixteen
  comparator tests pass. Live harness and preregistered MTP gate are unchanged.

- 2026-11-03: Corrected the prepared multi-block timeout test to accept both
  legal partial-write orderings while checking guarded peer output and exact
  poisoned replay. Runtime patch is unchanged; native execution remains pending.
  Frozen `da83fb8` context is staged separately on spark1 with all hashes matching.

- 2026-11-03: Extended sentinel context `a26fe32` staged separately on spark1;
  all transferred hashes match. Active-cluster build refusal passed (exit 2,
  no artifacts), with MTP3 still running unchanged. No source build has run.

- 2026-11-03: Added prepared two-shot/varlen, compiled C-proxy and mixed-ABI
  sentinel tests; native execution remains pending. Local suite 144 passed.
  Extended build context supersedes `e6015c6`, which was never built.

- 2026-11-03: Frozen sentinel build context `e6015c6` staged under
  `/home/nvidia/glm53-source-builds/roce-sentinel-e6015c6/context`; hashes match.
  Loaded-cluster refusal passed (exit 2, no artifacts); local suite 143 passed.

- 2026-11-03: Prepared candidate 30's pinned full-source build pair with a
  new-file-aware, clean-source applier. Ten rejection/application tests and
  actual fresh upstream control/candidate hash checks pass; no build has run.

- 2026-11-03: Prepared independent RoCE sentinel repair across all four kernel
  paths, host and C proxy (internal ABI 8, public API 1). Twelve exact-source
  CPU cases pass; 24 native cases prepared but not run. No runtime change.

- 2026-11-03: Reproduced the installed RoCE host failure-sentinel defect at
  sequence zero with extracted methods and synthetic records, without GPU work.
  Added candidate 30; older PR438 offsets conflict with this newer protocol.

- 2026-11-03: Staged frozen RoCE context `48e3156` under the head's
  `/home/nvidia/glm53-source-builds/roce-single-block-48e3156/context`.
  Hashes verified; active-cluster build refusal returned 2 with no artifacts.

- 2026-11-03: Prepared RoCE candidate 29's separate pinned full-source build
  pair and gate. Exact control/candidate source hashes verified in fresh trees;
  133 local tests pass. Native paired-rank and wrap-failure coverage remain open.

- 2026-11-03: Frozen readiness helper `5bc557b` correctly refused the loaded
  pair in 1.15s (exit 1, both containers observed); actual unit hash and MTP3
  benchmark PID unchanged. All 129 repository tests pass. No unit trial yet.

- 2026-11-03: Prepared candidate 26's read-only both-node startup gate in
  `experiments/startup-memory`, with an isolated service patch and preregistered
  window. Its 22 fault tests pass; live service remains unchanged during MTP3.

- 2026-11-03: Activated the isolated MTP3 trial after both-node snapshots,
  teardown and MemFree gates. Exact image/checkpoint/argv and unchanged KV pass;
  corrected functional/cache gates pass. Full matched matrix started at 14:19:56
  UTC as `glm53-ab-namespaced-mtp3-20261103.service`; adoption pending.

- 2026-11-03: All 107 repository tests pass after the reasoning-field fix.
  The full baseline and corrected functional receipts are preserved under
  `local/cluster-20261103/namespaced`.

- 2026-11-03: Full explicit-max baseline completed: 96 measured cells and 40
  warmups pass strict receipt validation (JSON SHA256 `9a6f72b4...`). Fixed the
  functional smoke to read modern `reasoning` and require actual reasoning;
  10 smoke tests pass. Functional requalification precedes the MTP3 switch.

- 2026-11-03: Verified both ranks already persist compiler caches and use the
  source/toolchain/device-keyed B12X cache. Recorded native startup preparation;
  candidates 02/03 now focus on coverage, isolation and boot-time evidence.

- 2026-11-03: Qualified a locked temporary pytest environment in a CPU-only
  disposable container, retaining the installed Torch/DSL/FlashInfer packages.
  Documented native bounded/high-offset and sanitizer commands; GPU gates pending.

- 2026-11-03: Corrected the Int64 build wrapper's `uv run ... python -c`
  package-inventory command before any build. Verified command dispatch in the
  live image without GPU work; replace the earlier staged context before building.

- 2026-11-03: Prepared the isolated single-block RoCE patch and graph regression
  at `experiments/roce-single-block`. Exact base/result hashes and patch parsing
  pass; native paired-GPU and serving qualification remain pending.

- 2026-11-03: Full repository validation passes: 100 Python tests, ShellCheck,
  Bash syntax and Python compilation. Added standard-library `collections`
  to the helper import allowlist; the full cluster baseline is still running.

- 2026-11-03: Audited Flare's TP2 cache geometry and streaming/metric contracts.
  Native allocation, the exact context cap and a benchmark adapter remain
  qualification gates; no Flare deployment or serving-control change made.

- 2026-11-03: Added strict comparison of the registered full MTP matrices,
  including prompt/control matching and raw-timing/counter consistency.
  Fifteen tests pass; the helper reports existing gates without adopting an arm.

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
