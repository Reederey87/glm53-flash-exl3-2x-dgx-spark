<h1 align="center">GLM-5.3 Flash EXL3 for 2x DGX Sparks</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

OpenAI-compatible vLLM serve of
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) as
**[Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)**
— a byte-identical public mirror of
[brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
snapshot `5ab363a8…` (uniform-K4 EXL3/TR3 routed-experts, 4 bpw, ~164 GiB, 120 shards)
so this recipe stays fetchable if the upstream Hub id moves. On a **2× NVIDIA GB10**
kit: tensor-parallel size 2 over CX7, native `sm_121a` cubins, API on `:8888`.
Served model id: **`GLM-5.3-Flash-EXL3`**. EXL3/TR3 quant by
[brandonmusic](https://huggingface.co/brandonmusic).

This is **EXL3 weights + fp8 KV** on GB10. Do not pass `--moe-backend marlin`.
The Hub card on brandonmusic (TP2/EP2/DCP2 + calibrated NVFP4 MLA KV) is the SM120 B12X
image (`verdictai/glm53-flash-exl3-k4:…-v84-dflash2`), not this overlay. Target KV
stays packed **`fp8_ds_mla`**. Speculator is **DFlash2 k=7**
([incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2));
draft attention is **FLASH_ATTN** (do not pin `TRITON_ATTN` — that mask is causal
inside the draft block on this image and collapses later-position accept).

## Decode (this kit, 2026-08-28)

Official numbers: sparkDash Decode bench, DFlash2 k=7, **Structured** (count 1→200) and **Code** (`clamp_00`…`clamp_49`) — same high-accept regime. Temp **0**, thinking **off**, 400 tokens, CUDA graphs, fused EXL3 MoE. Prompt types, not grammar / schema. Stream tok/s is per request; aggregate is all streams.

| Concurrency | TTFT | Stream tok/s | Aggregate tok/s |
|---|---:|---:|---:|
| **×1** | **719 ms** | **62.9** | **62.9** |
| **×2** | 6.62 s | 51.7 | 103.3 |
| **×4** | 6.30 s | 37.1 | 146.5 |

Serve recipe is `--max-model-len 900000` with KV pool **982,612** tokens (1.09× a full 900k request) at util 0.87. These runs are warm / empty KV — they do not need a filled 900k cache.

Lab `tests/bench_decode.py` on the same protocol (median of 5 × 400, C1): Structured **61.7** tok/s (0.918 accept / 6.43 per step); Prose (hash-map) **26.9** (0.332 / 2.33). Long context / mixed (~60–100k KV) 24–27. MTP k=2 baseline ~24.6.

Structured per-pos (lab median): **0.98 / 0.98 / 0.94 / 0.94 / 0.91 / 0.83 / 0.83**.
Prose per-pos: **0.75 / 0.58 / 0.41 / 0.28 / 0.16 / 0.09 / 0.06**.
Pinning `attention_backend=TRITON_ATTN` dropped structured to ~29 tok/s / 0.31 accept
(pos0 healthy, later positions collapsed).

Re-measure:

```bash
# structured (count 1→200)
python3 tests/bench_decode.py --phase structured --structured --runs 5 --max-tokens 400 --skip-coherence --out /tmp/glm53-structured.json
# prose (hash-map explanation)
python3 tests/bench_decode.py --phase prose --runs 5 --max-tokens 400 --skip-coherence --out /tmp/glm53-prose.json
```

## Quality (KLD)

Independent teacher-logit panel from
[malaiwah on the 4bpw discussion](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/discussions/1#6a9144846b0bdba943bfe86f):
KLD(teacher ‖ model), five cold runs, 25 sealed windows (51,175 positions). This
scores the **weights**, not this GB10 overlay. We serve the **4bpw** row.

| Model | Mean KLD (nats) | Size |
|---|---:|---:|
| TR3 K6 (6bpw) | 0.013723 | 254 GB |
| Official FP8 (cross-stack) | 0.020615 | 328 GB |
| **This checkpoint — EXL3 4bpw** | **0.024555** | **176 GB** |
| Official FP8 (brandonmusic stack, v44) | 0.024629 | 328 GB |
| NVFP4 (brandonmusic stack, v44) | 0.060535 | ~180 GB |

On the same stack, 4bpw matches official FP8 (~1.00× KLD) at **54%** of the bytes.
K6 (`malaiwah/GLM-5.3-Flash-TR3-6bpw`) is a different checkpoint.

## What runs

| Layer | Runtime |
|---|---|
| API | vLLM OpenAI (`/v1/chat/completions`) on the head, port **8888** |
| Weights | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (mirror of `brandonmusic/…` snapshot `5ab363a8…`) |
| Model id | `GLM-5.3-Flash-EXL3` (`--served-model-name`) |
| Image | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` FROM `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…` (arm64, CUDA 13.0) |
| Executor | `mp`, `--nnodes 2`, `--tensor-parallel-size 2` |
| Head | this machine, `HEAD_IP=10.0.0.1`, container `glm53-exl3-head` |
| Worker | `WORKER_USER@WORKER_IP` (this kit: `zurih@10.0.0.2`), `--headless`, `glm53-exl3-worker` |
| Fabric | CX7 QSFP: `enp1s0f1np1`/`rocep1s0f1` ↔ `enp1s0f0np0`/`rocep1s0f0`. Image NCCL (`USE_HOST_NCCL=0`) |
| Attention | `FLASHINFER_MLA_SPARSE_SM120` (NoPE MLA padded into GLM_NSA 576-wide) |
| KV | `--kv-cache-dtype fp8` → packed **`fp8_ds_mla`**. `--enable-prefix-caching` (block-aligned hits; see Prefix caching) |
| Experts | packed trellis + suh + svh + mcg, codebook MCG, **one fused `exllamav3_ext.exl3_moe` launch per layer** |
| Dense / shared / attn / embed / lm_head | native (unquantized) |
| Spec | **DFlash2 k=7** (`incoai/GLM-5.3-Flash-DFlash2`); draft KV `auto`/bf16, draft TP=1, FLASH_ATTN. Rollback `SPEC_METHOD=mtp` |
| Context | **900k** (`MAX_MODEL_LEN=900000`). Pool **982,612** tokens (~15.67 GiB fp8 MLA) at util 0.87 — 1.09× a full 900k request. Native 1M still does not allocate |
| Tools / reasoning | `--tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45` |
| Graphs | on (`ENFORCE_EAGER=0`) — MTP capture `1 2 3 4 6 8 12`; DFlash2 capture `1 2 4 8 16 24 32` |
| Vision | on (`LANGUAGE_MODEL_ONLY=0`) — image + video, `--limit-mm-per-prompt {image:4,video:1}`, `--skip-mm-profiling` |

Kernels: `TORCH_CUDA_ARCH_LIST=12.1a`. ExLlamaV3 pin `c5d9c657` (0.0.43) exposes
`exl3_moe` / `exl3_moe_max_concurrency`; aarch64 CPU allreduce stubs in
`overlay/patch_exl3_ext_aarch64.py`.

## Why the overlay exists

Stock `vllm/vllm-openai:glm53-flash-arm64-cu130` loads this checkpoint and dies on
the first forward: `pe_dim must be 64 for fp8_ds_mla`. GLM-5.3-Flash is **NoPE MLA**
(`qk_rope_head_dim=0`, `kv_lora_rank=512`). On SM12x the only sparse-MLA backend is
`FLASHINFER_MLA_SPARSE_SM120`, whose packed record is 512 NoPE + 16 B scales + 128 B
RoPE (656 B). The overlay zero-pads the 512-d latent into that GLM_NSA geometry
(RoPE pad is zeros; the QK dot is unchanged) and registers a real EXL3 method so
routed experts stay packed instead of expanding to BF16.

Registering the name `"exl3"` is not enough. Experts must stay **trellis + suh +
svh + mcg** and run Trellis/MCG. Shared experts, attention, embeddings, and
`lm_head` stay native. TP=2 shards gate/up **column-wise** and down **row-wise**;
the MoE runner all-reduces once per layer.

DFlash2 on this fork also needs three GLM-specific hooks the stock image lacks:
EAGLE3 aux capture at mHC (`hc_post` then `hc_contract` → 4096-wide, taps log as
`(6, 15, 25, 34, 43)`), drafter SWA **slot-sharing the MLA KV pages** (never
`page_size_padded` — the generic uniform-page path cannot serve this hybrid), and
checkpoint `is_causal: false` so draft attention is bidirectional inside the
block. Draft KV is forced `auto` because dense DFlash2 cannot use the target's
`fp8_ds_mla` layout and SM121 has no FA3/FA4 for plain FP8.

`overlay/patch_glm_video_placeholders.py` routes Glm5Next video timestamps through
the glm46v path and aligns placeholder blocks to encoder `grid_t`. The overlay
also disables GB10 `persistent_topk` so long-history decode uses
`top_k_per_row_decode`.

## KV cache

`--kv-cache-dtype fp8` is required. The SM12x sparse-MLA kernel only accepts packed
`fp8_ds_mla`. **bf16 KV has no sparse kernel** on this arch. With DFlash2 + vision
+ util **0.87**, the pool is **982,612 tokens** (~15.67 GiB fp8 MLA) at
`--max-model-len 900000` (1.09× a full 900k request). The drafter costs ~0 extra
pool (it slot-shares MLA tensors). Native 1M still does not fit; the previous
800k recipe at util 0.86 was **837,065** tokens (1.05×).

Keep **`SKIP_MM_PROFILING=1`** — a max-size image+video dummy profile OOMs this UMA.
`LIMIT_MM={"image":4,"video":1}`.

**NVFP4 KV is not available here.** FlashInfer’s SM12x NVFP4 kernels are dense MHA,
not sparse MLA. Do not confuse that with NVFP4 **weights** (`--moe-backend marlin`).

## Prefix caching (this kit, 2026-08-28)

`--enable-prefix-caching` is on. The OpenAI API is **stateless**: the client
resends the full history each turn; vLLM hashes that prefix. Concurrent chats
do **not** mix activations. `--max-num-seqs 4` is four **in-flight** generations,
not four parked sessions. This boot’s pool was **926,373** tokens (CUDA-graph
memory profiling; recipe table above is 982,612). MLA `KpoolTailManager`
disables **fine-grained** hits — only **block-aligned** tokens count.

Live A/B, temp **0**, thinking **off**, two distinct ~7.7k-token chats:

| Turn | Prefix hits | Compute | Prompt tok | TTFT |
|---|---:|---:|---:|---:|
| A cold | 0 | 7734 | 7734 | 10.5 s |
| A follow-up | **3584** | 4176 | 7760 | 11.4 s |
| B cold (different text) | 0 | 7734 | 7734 | 10.5 s |
| B follow-up | **3584** | 4176 | 7760 | 5.9 s |
| A again after B | **0** | 7786 | 7786 | 10.8 s |
| A+B concurrent | 3584 total | rest recomputed | 7748 each | A 9.5 s / B 16.1 s |

Isolation held: A answered `STILL_A` / `CONCUR_A`, B answered `CONCUR_B`.

A follow-up reused **46%** of the prompt (3584 / 7760), not the whole history.
After talking in B, A’s prefix was gone. Concurrent A+B: only one session hit.
Idle chats are not reserved in the pool. Expect prefill on later turns of an
old window; only the next turn of a still-hashed, block-aligned prefix skips
part of it.

## Abliteration (`ABLIT=1`)

Optional refusal-direction ablation, applied at weight-load time on top of the
EXL3 checkpoint — nothing is requantized or rewritten on disk. Artifacts live
in `ablit/` (two ~18 KB fp32 direction vectors + `LAYER_MAP.json`) and come
from
[drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock](https://huggingface.co/drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock).
The published recipe is **dealign-oproj-transplant**: every `self_attn.o_proj`
in layers **15–45** is orthogonalized against the refusal direction, layers
**0–14 stay stock** as safety anchors (keeps coherence while dropping
refusals), and the checkpoint MTP block (layer 45) is included.

The edit per o_proj (`W: [4096, in]`, `r`: fp32 direction, ‖r‖=1):

```text
W' = (I - alpha * r rᵀ) W      # default ABLIT_ALPHA=3.0 (alpha_ref)
```

Output components orthogonal to `r` are preserved exactly; the component
along `r` is scaled by `1 − alpha` (alpha 3 inverts it — stronger
over-projection; set `ABLIT_ALPHA=1.0` for the plain projection that zeroes
it). Since o_proj is native BF16 here and `RowParallelLinear` shards only the
input dim, the row-space edit is identical on both TP ranks with no
collectives. Applied by `overlay/ablit_runtime.py`, installed by
`overlay/patch_ablit.py` at the end of `Glm5NextModel.load_weights` /
`Glm5NextMTP.load_weights` — before CUDA-graph capture, after the loaders are
done. The DFlash2 drafter is never touched.

Enable:

```bash
ABLIT=1 ./start.sh restart          # or set ABLIT=1 in .env, then ./start.sh restart
./start.sh logs | grep ablit        # "orthogonalized layers.15.self_attn.o_proj …" per rank
```

Disable the same way (`ABLIT=0` / unset → hook is a no-op, stock weights).
No rebuild: the artifacts + hook are bind-mounted into both containers on
every `./start.sh`, so prebuilt GHCR images work too.

| Knob | Default | What |
|---|---|---|
| `ABLIT` | `0` | `1` = apply the o_proj orthogonalization at load (both ranks) |
| `ABLIT_DIRECTION` | `dealign` | `dealign` (published recipe) \| `bf_oproj` (blackfrost direction, `alpha_ref` 3.0) \| absolute path to a custom direction `.pt` |
| `ABLIT_LAYERS` | `15-45` | inclusive range; `45` is the checkpoint MTP block |
| `ABLIT_ALPHA` | `3.0` | projection scale. `1.0` = plain projection, >1 over-projects |
| `ABLIT_INCLUDE_MTP` | `1` | also edit the MTP block's o_proj when it loads (`SPEC_METHOD=mtp`) |

Caveats: the KLD quality panel above was measured **without** ablit; expect a
refusal-behavior change and re-tune `ABLIT_ALPHA` if coherence degrades
(lower it first). DFlash2 acceptance rates can shift a little (the drafter is
stock while the target's outputs change). Provenance is the NVFP4 ablit
checkpoint, but only the **direction vectors** are used here — the EXL3
serve keeps its own weights, and o_proj is unquantized in both.

## Quick start (2× Spark)

```bash
git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks.git
cd GLM-5.3-Flash-EXL3-2x-DGX-Sparks
cp .env.example .env          # edit HEAD_IP / WORKER_IP / WORKER_USER if needed
./download.sh                 # optional: EXL3 + DFlash2 into the head HF cache only
./start.sh                    # pull public GHCR :exl3, download if missing, rsync, launch TP=2
```

First run of `./start.sh` copies `.env.example` → `.env` if missing. Prefix env
wins over `.env` (`SPEC_METHOD=dflash SKIP_DOWNLOAD=1 ./start.sh restart`).

`./start.sh` downloads weights automatically when the HF cache is incomplete
(120 shards of `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, falling back to
`brandonmusic/GLM-5.3-Flash-tr3-4bpw` if the mirror is incomplete, plus DFlash2 when
`SPEC_METHOD=dflash`). `./download.sh` is the same Hub fetch **on this machine
only** — no docker, no SSH, no worker rsync. Use it to stage ~164 GiB before
the worker is ready. `REFRESH_WEIGHTS=1 ./download.sh` re-fetches.
Already present: both scripts skip. `./start.sh` still rsyncs the cache to the
worker unless `SKIP_SYNC=1`.

DFlash2 (`incoai/GLM-5.3-Flash-DFlash2`, ~2.3 GiB BF16, CC BY-NC-ND 4.0 research/eval)
is the default. Rollback:

```bash
SPEC_METHOD=mtp ./start.sh restart      # MTP k=2
```

`./start.sh` will:

1. Preflight docker/ssh/disk on both nodes
2. `docker pull` `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` (public; no login) on every start, then `docker save | ssh docker load` onto the worker if the digest changed. `SKIP_PULL=1` keeps a local copy.
3. Download the TR3 EXL3 repo into `$HF_HOME` / `~/.cache/huggingface` (~164 GiB, 120 shards) if missing. Same job as `./download.sh`, which stops here (head only).
4. `rsync` that cache to `${WORKER_HOME}/.cache/huggingface`
5. Start rank 1 `--headless` on the worker, rank 0 + API on the head
6. Poll `/health` (weight load + warmup is slow; `READY_TIMEOUT` default 3600s), then a **nonfatal** DFlash2/sampler shape sweep so the first client is not the first JIT on TP=2. `GLM53_BOOT_SHAPE_WARMUP=0` skips it.

The worker does not need GHCR access — only the head pulls, then ships the image over SSH.

```bash
./download.sh                              # head HF cache only (no worker); same as ./start.sh download
SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh     # weights already local on both nodes
SKIP_PULL=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh restart  # keep local image, no GHCR
BUILD=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh restart  # rebuild overlay from this repo + ship
./start.sh status
./start.sh logs                # head
./start.sh logs worker
./start.sh stop                # or ./stop.sh
```

Do not pull `glm53-flash-sm121:v8` — that is the older NVFP4/Ray kernel.

API: `http://127.0.0.1:8888/v1` (LAN: `http://10.0.0.1:8888/v1`).

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-5.3-Flash-EXL3",
    "messages": [{"role": "user", "content": "hello!"}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Thinking defaults on. Disable it with the **top-level** JSON field
`"chat_template_kwargs": {"enable_thinking": false}`. This closes the empty
thinking block in the generation prompt and omits the reasoning-effort hint.
Do not send a literal nested `extra_body` object over raw HTTP; `extra_body` is
an OpenAI Python SDK option that merges its contents into the top-level request.
The Hub `generation_config.json` stamps `temperature=1.0` / `top_p=0.95` unless
the request overrides. The launcher sets
`--chat-template /opt/glm53/chat_template.jinja` (checkpoint jinja is language-only).

Needs: Docker (no sudo) on both nodes, passwordless SSH head → worker,
`hf` / `huggingface-cli` + `curl` + `rsync` on the head, ~180 GiB free per
node for the first download. The GHCR image is public; login is only needed
if you hit anonymous pull rate limits (`GHCR_TOKEN` + `GHCR_USER`).
Mixed OS accounts: set `WORKER_USER` (this kit uses `zurih` on spark2).

NCCL cannot use the `10.0.0.x` loopback aliases — leave the CX7 pins unless
your cabling differs. `ncclCommInitRank` hangs without them.

## .env

| Knob | Default | What |
|---|---|---|
| `HEAD_IP` | `10.0.0.1` | this node, NCCL/vLLM master |
| `WORKER_IP` | `10.0.0.2` | other Spark |
| `WORKER_USER` | *(unset = `$USER`)* | SSH user on the worker |
| `WORKER_HOME` | `$HOME` if same user, else `/home/$WORKER_USER` | worker HF cache |
| `MODEL` | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | Hub repo into the HF cache (mirror) |
| `MODEL_FALLBACK` | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` | Used if the mirror 404s or has fewer than 120 shards |
| `SERVED_MODEL_NAME` | `GLM-5.3-Flash-EXL3` | OpenAI `model` id (`/v1/models`) |
| `IMAGE` | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` | public GHCR tag; pulled on every start. `SKIP_PULL=1` skips. `BUILD=1` rebuilds the overlay |
| `GHCR_TOKEN` / `GHCR_USER` | *(unset)* | optional login if anonymous GHCR pull is rate-limited |
| `PORT` | `8888` | OpenAI API on the head |
| `TP` / `NNODES` | `2` / `2` | do not change for this recipe |
| `QUANTIZATION` | `exl3` | overlay method; never `marlin` |
| `MTP_TOKENS` | `2` | MTP speculative tokens (`SPEC_METHOD=mtp`) |
| `SPEC_METHOD` | `dflash` | `dflash` / `mtp` / `none`. Rollback: `SPEC_METHOD=mtp ./start.sh restart` |
| `DFLASH_MODEL` | `incoai/GLM-5.3-Flash-DFlash2` | DFlash2 draft Hub repo (~2.3 GiB BF16) |
| `DFLASH_TOKENS` | `7` | DFlash2 speculative tokens (trained block 8) |
| `DFLASH_DRAFT_TP` | `1` | keep the 2.3 GiB drafter on rank 0 (no CX7 per draft step). Empty = inherit TP |
| DFlash2 draft KV | `auto` (bf16) | target stays `fp8`/`fp8_ds_mla`; dense draft has no MLA FP8 backend on SM121 |
| DFlash2 attention | *(unset)* | SM121 picks FLASH_ATTN for non-causal SWA. Do not pin `TRITON_ATTN` |
| `ENFORCE_EAGER` | `0` | CUDA graphs; MTP capture `1 2 3 4 6 8 12`, DFlash2 `1 2 4 8 16 24 32` |
| `EXL3_FUSED_MOE` | `1` | `exl3_moe` per layer; `0` = LinearEXL3 loop |
| `KV_CACHE_DTYPE` | `fp8` | packed `fp8_ds_mla`; not `nvfp4`, not bf16 |
| `GPU_MEM_UTIL` | `0.87` | GB10 UMA budget (DFlash2 + vision; live pool 982,612 tokens) |
| `MAX_MODEL_LEN` | `900000` | default context. The 982,612-token pool is **1.09×** a full 900k request. Native 1M does not allocate |
| `MAX_NUM_SEQS` | `4` | decode batch; MTP adds k+1 tokens/seq |
| `MAX_NUM_BATCHED_TOKENS` | `1024` | prefill chunk; 8192 oversubscribes GB10 indexer topk on long context |
| `GLM53_SUPPRESS_STOPS_IN_REASONING` | `1` | ignore client `stop` strings until `</think>` (thinking-on default) |
| `GLM53_BOOT_SHAPE_WARMUP` | `1` | after `/health`, burn DFlash2 BLOCK / sampler / kpool shapes (nonfatal) |
| `TRITON_HOST_CACHE` / `TILELANG_HOST_CACHE` | `$CACHE_ROOT/triton` / `tilelang` | persist JIT caches across container recreate |
| `LANGUAGE_MODEL_ONLY` | `0` | load vision tower (image + video) |
| `SKIP_MM_PROFILING` | `1` | skip max-size MM dummy at init (OOM otherwise) |
| `LIMIT_MM` | `{"image":4,"video":1}` | `--limit-mm-per-prompt` |
| `HEAD_CX7_IF` / `WORKER_CX7_IF` | `enp1s0f1np1` / `enp1s0f0np0` | NCCL sockets |
| `HEAD_CX7_IB` / `WORKER_CX7_IB` | `rocep1s0f1` / `rocep1s0f0` | NCCL HCAs |
| `USE_HOST_NCCL` | `0` | image nvidia-nccl; host preload duplicates DeepEP |

## Image / overlay

```bash
docker build -t glm53-flash-sm121:local .
# or: BUILD=1 ./start.sh
```

`./start.sh` **pulls** `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`
(public) on every start unless `SKIP_PULL=1`. `BUILD=1` rebuilds the overlay from
this Dockerfile instead. After CUDA compile, Python overlay edits
(`overlay/exl3.py`, tests) are a cheap layer so they do not rebuild
`exllamav3_ext`.

| Path | Role |
|---|---|
| `Dockerfile` | NoPE sparse-MLA patches + EXL3 install (`sm_121a`) + self-check |
| `overlay/exl3.py` | `Exl3Config` / packed load / TP shard / fused `exl3_moe` apply |
| `overlay/patch_exl3_ext_aarch64.py` | stub AVX CPU allreduce so the ext builds on GB10 |
| `overlay/patch_model_overrides.py` | `"exl3"` in ModelConfig overrides |
| `tests/test_exl3_overlay.py` | registry, TP shard, `sm_121a` cubin, fused vs loop GEMM, `EXL3_FUSED_MOE=0` |
| `tests/bench_decode.py` | streaming decode + coherence; `--structured` is the count-1→200 median |
| `start.sh` / `stop.sh` / `download.sh` | 2-node launch; Hub fetch on the head only |
| `files/chat_template.jinja` | GLM-5.3 MM template (`<|image|>` / `<|video|>`); checkpoint jinja is language-only |
| `overlay/qwen3_dflash2.py` | DFlash2 draft (grouped conv + candidate selector) |
| `overlay/dflash2_speculator.py` | DFlash2 selector walk (V2 speculator) |
| `overlay/patch_dflash2.py` | registry + `decoder_layer_cls` + speculator dispatch + draft KV `auto` on MLA/FP8 |
| `overlay/patch_glm_eagle3.py` | Glm5Next EAGLE3 aux-hidden layers (mHC `hc_post` + contract) |
| `overlay/patch_glm5_drafter_group.py` | keep GLM KV fast path; DFlash2 SWA slot-shares MLA pages (no padding) |
| `overlay/patch_glm_video_placeholders.py` | align video timestamp blocks to encoder `grid_t` |
| `overlay/ablit_runtime.py` | ABLIT: o_proj refusal-direction orthogonalization at load (`ABLIT=1`) |
| `overlay/patch_ablit.py` | install the ABLIT hook into `Glm5NextModel` / `Glm5NextMTP` load (idempotent) |
| `overlay/patch_suppress_stops_in_reasoning.py` | fail-closed detokenizer guard: client `stop` dormant until `</think>` |
| `scripts/boot-shape-warmup.sh` | post-`/health` DFlash2 k=7 BLOCK ladder + sampler/kpool arms |
| `ablit/` | direction vectors + `LAYER_MAP.json` from drowzeys' published ablit recipe |
| `tests/test_ablit.py` | recipe integrity, orthogonalization math, TP-shard equivalence, hook gating |

Image-build runs `EXL3_SELFCHECK_GPU=0`. `./start.sh` runs the GPU self-check
(`docker run --gpus all`) before shipping unless `SKIP_OVERLAY_VERIFY=1`.

## Do not

- Destroy HF weights, requantize, or `docker rm` HF caches. `REFRESH_WEIGHTS=1 ./download.sh` only if you intend to re-fetch
- `--moe-backend marlin`, NVFP4 weights, or `glm53-flash-sm121:v8` as this serve
- qemu / amd64 / `cstechdev/vllm:glm53-flash-nope-sm120-*` / verdictai SM120 B12X
- `--kv-cache-dtype nvfp4` or bf16 (no sparse-MLA kernel)
- `"attention_backend": "TRITON_ATTN"` in speculative-config (causal-in-block on this image)
- Point `ABLIT_DIRECTION` at another checkpoint's direction without checking `hidden_size`, or commit edits to the `.pt` artifacts
- Change TP, CX7 pins, or `USE_HOST_NCCL` unless you are re-plumbing NCCL
- Force-push

## License

This repository (serve scripts, overlay, docs) is **MIT**. The EXL3/TR3
checkpoint stays [ShapleyMCG License 1.0](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/blob/main/LICENSE)
(unmodified upstream LICENSE; also on
[brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)).
DFlash2 stays [CC BY-NC-ND 4.0](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).

## Credits

- **EXL3/TR3 weights:** [brandonmusic](https://huggingface.co/brandonmusic) —
  [GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
  (uniform-K4 routed-experts, ShapleyMCG License 1.0). Public mirror for this
  recipe: [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)
- **EXL3 format / kernels:** [turboderp](https://github.com/turboderp-org/exllamav3) (ExLlamaV3)
- **Base model:** [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
- **DFlash2 drafter:** [IncoAI](https://huggingface.co/incoai) —
  [GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  (CC BY-NC-ND 4.0, research/eval)
- **KLD panel:** [malaiwah](https://huggingface.co/malaiwah) —
  [discussion #1](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/discussions/1#6a9144846b0bdba943bfe86f)
- **Abliteration recipe / direction artifacts:** [drowzeys](https://huggingface.co/drowzeys) —
  [keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock](https://huggingface.co/drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock)

