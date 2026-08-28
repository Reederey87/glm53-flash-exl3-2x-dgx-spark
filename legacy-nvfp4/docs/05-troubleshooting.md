# Troubleshooting

Failure modes actually hit while building this, and what they mean.

## `Check failed: num_tokens > 64 (4 vs. 64)`

The `(32, 2176)` specialization is missing — you are running the TP=1 base image
rather than the TP=2 image from `image/build.sh`. See `03-tp2-kernel-fix.md`.

```bash
docker run --rm --entrypoint bash <image> -lc \
  'GLM53_GB10_PATCH_DISABLE=1 python3 /opt/glm53-tp2/verify_tp2.py'
```

## `cudaErrorNotPermitted`, apparently in the sampler

```
torch.AcceleratorError: CUDA error: operation not permitted
  ... gumbel_sample -> local_argmax.gather(...)
```

Almost certainly the MoE kernel, not the sampler. Set `MOE_BACKEND=marlin`.
A bad kernel poisons the CUDA context and the error surfaces at the next
synchronisation point. Do not debug the sampler.

## `pe_dim must be 64` at KV warmup

The sparse backend is being used without the NoPE adaptation — you are on a
stock vLLM image. Use the image built by `image/build.sh`.

## Boot hangs at `parallel_state` / NCCL init

A previous failed run left the **worker unit active**. It is headless and does
not exit when the head dies, and `systemctl start` on an active unit is a silent
no-op, so the new head rendezvouses with a stale worker that still holds ~90 GiB.

`start-cluster.sh` tears both ends down before starting, so re-running it is the
fix. If you start units by hand, stop **both** first.

## The startup memory gate never clears

`--gpu-memory-utilization 0.85` needs ~103 GiB of the 121.69 GiB pool free.
Anything else resident will block it: a remote IDE session, another model, a
leftover container. Check with `free -g` and `docker ps` on both nodes.

## Both nodes stop answering ssh (accept TCP, no banner)

Userspace starvation. Something is thrashing memory — usually two model
instances resident at once, or a crash-looping unit re-loading 90 GiB. There is
no remote fix; the nodes need a power cycle.

Prevent it: never run a second model service alongside this one, and if you have
another model's watchdog timer installed, **stop that timer** for the duration.
A watchdog that restarts a different model will fight you for the same memory.

## It serves, but the output is garbage

If completions come back as repeated punctuation with empty content, the
attention kernel is numerically wrong for this model — do not tune sampling.
Run `bench/smoke.sh`; it checks for exactly this.

## Slow first generations

Cold JIT. Several kernels compile on first use after any boot. Warm with
`bench/smoke.sh` before timing anything.
