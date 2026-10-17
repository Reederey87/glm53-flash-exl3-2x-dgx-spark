# Troubleshooting

Start every diagnosis with the boot log. The first error is usually the cause;
later errors are fallout.

```bash
./launch.sh logs
docker logs vllm_node 2>&1 | grep -iE 'error|traceback|oom|killed|kv cache|blocks' | tail -60
```

## Launcher and setup

| Symptom | Likely cause | Fix |
|---|---|---|
| `CLUSTER_NODES still holds the env.example addresses` | `.env` was copied but not edited | Set the real head/worker CX7 IPs, or `ALLOW_EXAMPLE_NODES=1 ./launch.sh` for a deliberate example run. |
| `cannot ssh to worker` | Wrong user/key, or the worker address is on the 10G network | Use the CX7 IP and passwordless SSH as the same user. Test `ssh <worker> true` first. |
| `less than 180 GiB free` | The checkpoint + image do not fit | Free space under `$HOME`; the checkpoint alone is ~166 GiB. |
| `pinned upstream ref not found` | `.vendor` is a shallow or unrelated checkout | Remove `.vendor/spark-vllm-docker` and re-run, or set `SPARK_VLLM_REF` to a commit in that clone. |
| `PyYAML missing` warning, then `pip` error | System Python is externally managed | Install PyYAML with your package manager or `uv pip install --system pyyaml`; upstream needs it. |
| `docker pull` fails | Registry/network, or the arm64 manifest moved | Retry; confirm `docker manifest inspect eugr/spark-vllm-b12x:latest` shows arm64. Pin the recorded digest in `.env` if `latest` is unstable. |

## Image and model

| Symptom | Likely cause | Fix |
|---|---|---|
| `unknown quantization: modelopt_mixed` | Wrong image; upstream vLLM does not read this checkpoint | Use `eugr/spark-vllm-b12x:latest` (or the recorded digest). Do not build a stock vLLM image. |
| Error mentioning `csf_32x2_v1`, compressed scales, or an unknown `weight_scale_encoding` | Wrong image | Same fix. The format is only in the B12X container. |
| `CUDA error: no kernel image is available` or an sm_121 compile failure | Wrong `CUTE_DSL_ARCH` or a non-B12X image | Keep `CUTE_DSL_ARCH=sm_121a` from the recipe; do not reuse the old `TORCH_CUDA_ARCH_LIST=12.1a`. |
| Image IDs differ on head and worker | Worker has an older copy or a failed transfer | `./launch.sh setup` copies the matching image; if it persists, remove `vllm-node-b12x` on the worker and re-run. |
| Checkpoint download stalls or a shard is truncated | Network or disk | Re-run `./launch.sh setup`; then `python3 scripts/validate_hf_snapshot.py --repo ~/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local --expected-shards 40`. |

## Boot and memory

| Symptom | Likely cause | Fix |
|---|---|---|
| Container exits during weight load | Not enough free unified memory; another process holds pages | `./launch.sh down`, wait for memory to settle, confirm `free -g` shows the node idle, retry. |
| `GPU KV cache size` line missing | Boot failed before cache allocation | Search the log for the first traceback or OOM; a missing line is a symptom, not the cause. |
| `kv` says BELOW TARGET | `--kv-cache-memory-bytes` was overridden, or an old container is running | `./launch.sh render` to confirm `11G`; restart; see [04-kv-cache.md](04-kv-cache.md). |
| OOM with 11G that disappears at 10G | Too little headroom for activations/graphs | Lower to 10G and accept ~1.9M, or lower `MAX_MODEL_LEN` and keep 11G. |
| Very slow first boot | AOT/JIT compilation into cold caches | Let it finish once; keep `~/.cache/vllm`, `~/.cache/flashinfer`, `~/.cache/b12x`, `~/.triton`. |

## Network and NCCL

| Symptom | Likely cause | Fix |
|---|---|---|
| `NCCL ... IB ... not found`, `ibv_modify_qp failed`, or a hang at connect | `IB_IF` names are wrong or only one twin is listed | Use the live devices (`rocep1s0f1,roceP2p1s0f1` on this hardware); both twins, comma separated. |
| Hang after "waiting for worker" | Worker cannot reach the head's `MASTER_PORT` | Check the CX7 subnet and firewall; test `nc -vz <head> 29501` from the worker. |
| All-reduce slow or falls back | RoCE all-reduce disabled or PCIe path selected | Keep `VLLM_ENABLE_ROCE_ALLREDUCE=1`, `VLLM_ENABLE_PCIE_ALLREDUCE=0`. |
| `docker` says address already in use | Another `vllm_node` or an old server on `:8000` | `./launch.sh down`; `docker ps`; free the port or set `PORT`. |

## API behavior

| Symptom | Likely cause | Fix |
|---|---|---|
| `/health` works on the head but not the worker | Expected: the worker is headless | Use the head's `:8000` only. |
| `tool_parser` WARN in smoke | The model answered in prose instead of calling the tool | Check the raw response. If tool calls are required, tighten the prompt; the parser is `glm47` as in the vendor recipe. |
| `reasoning_parser` WARN | The answer had no `reasoning_content` (short/greeting prompts can skip it) | Not necessarily a defect; inspect `content`. |
| `prefix_cache` WARN (cached_tokens=0) | First requests raced, or the cache evicted the prefix | Re-run; for a controlled check, send the same long prompt twice with a pause. |
| `spec` says metrics missing | Speculative decoding is off or the wrong endpoint | Confirm the recipe's `--speculative-config` and scrape `/metrics` directly. |
| Model answers, but quality looks wrong | Wrong checkpoint revision or a different quantization | Confirm the model id and revision (`76348dd`); do not mix NVFP4, EXL3, and FP8 checkpoints. |

## Clean restart

If the state is unclear:

```bash
./launch.sh down
docker ps -a --filter name=vllm_node
ssh <worker> 'docker ps -a --filter name=vllm_node'
./launch.sh preflight
./launch.sh
```

Do not delete the HF cache or the compile caches to "fix" a startup problem
unless the log points at them; re-downloading 166 GiB is rarely the answer.
