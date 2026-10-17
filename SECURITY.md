# Security

## Reporting

Open a GitHub issue for anything affecting this kit's scripts, recipe, or
launcher. For vulnerabilities in the upstream projects (vLLM, FlashInfer, the
B12X image, `spark-vllm-docker`, or the checkpoint), report to those projects
directly.

## Design notes

- **The API binds loopback by default.** The recipe sets `--host 127.0.0.1`.
  vLLM has no authentication here, so exposing it with `HOST=0.0.0.0` puts an
  unauthenticated model on the network. If you need remote access, use an SSH
  tunnel (`ssh -N -L 8000:127.0.0.1:8000 <head>`). If you must bind a LAN
  address, put it behind a firewall and an authenticating reverse proxy.
- **`.env` is gitignored.** It holds your node addresses and may hold
  `HF_TOKEN`. Do not commit it. `HF_TOKEN` is optional because the checkpoint
  is public; prefer exporting it in your shell over writing it to disk.
- **No credentials are needed by the kit.** It uses your existing SSH setup.
  The upstream launcher uses passwordless SSH between the nodes; restrict that
  key to the cluster hosts.
- **Containers run privileged, with host networking and host IPC.** That is
  what multi-node vLLM with RDMA requires. Run this on hardware you control
  and do not co-locate untrusted workloads on the same nodes.
- **The image is pulled by tag.** `eugr/spark-vllm-b12x:latest` is a mutable
  community tag. `launch.sh` records the resolved digest in
  `local/image-digest.txt`; for a long-lived deployment, pin the digest in
  `IMAGE_REF` and re-validate before changing it.
- **The checkpoint is third-party quantized weights.** Treat it as data, not
  as trusted code. It is loaded by the B12X runtime; the model card is the
  authority for its provenance and license.

## Secrets scan

CI fails if a private key, GitHub token, or Hugging Face token pattern is
committed, and if `.env` is tracked.
