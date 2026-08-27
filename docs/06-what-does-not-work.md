# What does not work

Approaches tried and abandoned, so nobody repeats them. Each cost real hours.

## Zero-padding the NoPE query in Python

**Idea.** `FLASHINFER_MLA_SPARSE_SM120` wants `pe_dim=64`; GLM is NoPE. Pad the
query and `k_pe` with 64 zeros so the packed 656-byte layout lines up. The dot
product is mathematically unchanged.

**Result.** Boots cleanly, serves, `/health` returns 200 — and every completion
is 2048 `!` characters with empty content.

**Why.** Two reasons, and the second is fatal:

1. The kpool buffer is **2176** wide (`index_topk` 2048 + always-selected tail,
   rounded to `BLOCK_N=128`). Slicing it to 2048 to match the compiled kernel
   silently drops selected tokens.
2. FlashInfer ships the sparse module as a **precompiled AOT `.so`** that takes
   precedence over its Python and CUDA sources. Editing Python cannot change
   which kernel runs. The specialization has to be *recompiled* — which is what
   `image/build.sh` does.

**Lesson.** A clean boot proves nothing about a kernel. Only generated text does.

## Running the sparse layers as dense MLA

**Idea.** Skip sparse attention entirely: null `index_topk` so vLLM treats the
model as full MLA and picks the arch-generic `TRITON_MLA`. Dense attention is a
superset of top-k, so it should be correct, just slower.

**Result.** It *works*. It passes the full correctness suite. But it needs a
patch to vLLM's MLA prefill backend to accept dims `(256, 0, 256)`, plus config
surgery to drop the indexer, and it measured **no faster** than the sparse path
at benchmark context lengths while producing a **2.1x smaller KV pool**
(389K vs 813K tokens).

**Why keep it out of this repo.** More moving parts, worse memory, no speed win,
and it diverges from how the model is designed to run. It is a legitimate
fallback if the sparse path ever breaks, not a recommendation.

## Marking `TRITON_MLA` as sparse

**Idea.** Get the dense Triton backend selected on the sparse path by having it
advertise `is_sparse=True`, then route prefill through the decode MQA kernel.

**Result.** Structurally wrong. It drops upstream's per-token
`block_table`/`seq_lens` expansion, so a multi-token batch indexes a per-request
table, and every prefill token would share one `seq_len` — i.e. non-causal
prefill. It also has to stub `prefill_backend.clone()` because there is no dense
MHA prefill for these dims.

## Chasing newer NVIDIA driver branches

Ubuntu's `multiverse` carries `nvidia-driver-595` and `-610`, and they have real
HWE kernel modules. Do not jump to them to "fix" GB10 issues:

- DGX OS pins a validated driver for this hardware; the vendor also pins
  `cuda-drivers*` to priority `-1` in `/etc/apt/preferences.d/`, which is a
  deliberate signal.
- Everything here is validated against driver **580.173.02** with CUDA 13.0, and
  the FlashInfer AOT module is compiled against that ABI.

The upgrade worth taking is a new **DGX OS release**, not a newer Ubuntu driver
branch.

## Raising `--gpu-memory-utilization` to make more KV

0.95 (the recipe value) claims 115.6 GiB of a 121.69 GiB *unified* pool. The
startup gate never clears. 0.88-0.90 flaps depending on what else is resident.
0.85 is the practical ceiling on this hardware; the KV pool is what it is.
