# Security

## Reporting

Open a GitHub issue for anything affecting this kit's scripts or image build.
For vulnerabilities in the upstream projects (vLLM, FlashInfer, the base image,
the model), report to those projects directly.

## Design notes you should know

- **The API binds loopback only.** `VLLM_HOST=127.0.0.1` is deliberate. Reach it
  remotely with an SSH port-forward, not by binding `0.0.0.0` — the server has
  **no authentication**, so a non-loopback bind exposes an unauthenticated model
  to your network.
- **`cluster.env` is gitignored.** It holds hostnames and paths for your
  environment. Keep it that way.
- **No credentials are needed by this kit.** It expects your existing SSH setup.
  If you add keys or tokens, do not commit them.
- **The container runs with `--gpus all`, host networking and host IPC**, which
  is what multi-node vLLM requires. Run it on hardware you control.
