# Rebuilt control: functional qualification and live matrix

Control image `3b5da79c42ed4204440cfb07c6de1eba46441d5cba6b57d5428d44e60c10d0ed`
is healthy on both Sparks. Serving invocation
`6d2701bcf90146258ce7316c5661fe2a` remained unchanged from startup through
qualification. No image adoption or performance conclusion is claimed.

At **05:27:59 UTC on November 4**, all six API checks and all eight prefix
rows passed. The first request at each prefix length reused zero tokens;
the second reused all 1,024 / 4,096 / 8,192 / 16,384 tokens. Cache labels and
geometry match the incumbent exactly. The KV pool is **2,350,210 tokens**.
There were exactly 13 successful functional requests and none running or waiting.

Both nodes passed the frozen private runtime comparison again after health:
exact control image, unchanged containers, exact serving argv, NVIDIA checkpoint
revision and metadata, and installed dense-kernel source. The four registered
inherited build-worker environment differences are the only differences from
the incumbent; remaining environment, launch, host and mount controls match.
Private inspection values remain on the cluster.

The functional scripts are frozen at `e0d7869`; the speed harness remains
`05c6023`. Functional parser probes retain their original low-effort settings
to preserve the comparison. All measured chat-decode requests explicitly use
**`reasoning_effort=max`**, with completion counts including reasoning.

The full matched speed matrix started at **05:28:43 UTC** as
`glm53-int64-control-matrix-20261104.service`, invocation
`03687f011642433b8fd1b91ebb261c59`, initial PID374273. Its command is identical
to the completed incumbent's except for output paths: seed2026110304, three
repetitions, 200 generated tokens, code/prose/structured decode, the 8k–128k
uncached prefill ladder and concurrency1/2/4/8 with server cap4. It requires
96 measured cells and40 warmups. Do not run other inference or GPU work during
this matrix. A running unit is not a completed performance result.

- [Healthy runtime verification](head/runtime-healthy.json)
- [Functional start accounting](head/functional-start.json)
- [Six API checks](head/smoke.json)
- [Eight prefix rows](head/prefix.json)
- [Exact KV pool](head/kv.json)
- [Functional qualification result](head/functional-result.json)
- [Exact matrix command](head/matrix-launch.json)
- [Initial benchmark state](head/matrix-initial-state.json)

The parent receipt manifest covers these bytes. Earlier activation, snapshots,
cache preservation and image-selection receipts remain in `head/` and `worker/`.
Original caches and incumbent image are retained; watchdog is disarmed during
the window. Complete and validate this matrix against the incumbent before
testing the patched candidate. Final restoration and watchdog rearming remain
required at the end of the window.
