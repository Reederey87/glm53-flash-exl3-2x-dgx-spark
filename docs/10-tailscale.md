# Tailscale access to the served model

The API stays bound to `127.0.0.1:8000` on `spark1`. Tailscale Serve is
**disabled on this tailnet**, so direct Serve/Funnel cannot be used yet. The
kit exposes the model to the tailnet through a small TCP proxy bound only to
the Spark's Tailscale IPv4.

## Topology

| Item | Value |
|---|---|
| Tailscale node | `spark-a183` |
| Tailscale IPv4 | `100.125.23.98` |
| MagicDNS | `spark-a183.tail0c5008.ts.net` |
| Local vLLM | `127.0.0.1:8000` |
| Tailnet listener | `100.125.23.98:8000` |
| Tailnet client URL | `http://100.125.23.98:8000/v1` |
| MagicDNS client URL | `http://spark-a183.tail0c5008.ts.net:8000/v1` |

The proxy is [scripts/tailscale_proxy.py](../scripts/tailscale_proxy.py). It
never binds `0.0.0.0` or the LAN interface. Tailscale encrypts the path with
WireGuard; the proxy does not add TLS.

## Install and arm

On `spark1` as `nvidia`:

```bash
cd /home/nvidia/glm53-fp4-local
install -m 644 systemd/vllm-glm53fp4-tailscale.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now vllm-glm53fp4-tailscale.service
```

Check:

```bash
systemctl --user status vllm-glm53fp4-tailscale.service --no-pager
ss -ltn | grep 100.125.23.98
curl -fsS http://100.125.23.98:8000/health
curl -fsS http://100.125.23.98:8000/v1/models
```

The unit is `Restart=always`; it waits for the Tailscale address at startup and
re-detects it on restart. It is enabled under `default.target`, so it starts at
boot with the cluster (linger is already on).

## Client configuration

On the other Mac, point an OpenAI-compatible client at:

```text
http://100.125.23.98:8000/v1
```

For DeepSeek Harness on that Mac, add a custom provider with:

```yaml
providers:
  glm-tailnet:
    displayName: GLM-5.3-Flash (tailnet)
    api: openai-completions
    baseURL: http://100.125.23.98:8000/v1
    headers:
      Authorization: Bearer local
    models:
      - id: nvidia/GLM-5.3-Flash-FP4-Local
        name: GLM-5.3-Flash-FP4-Local
        contextWindow: 480000
        maxTokens: 16384
        input: [text]
```

Keep the model's 480k/16k DSH limits: compaction reserves 65,536 output tokens,
and the server's `--max-model-len` is 500,000.

## Security

- The vLLM API has **no built-in authentication**. Tailscale ACLs are the only
  boundary. Restrict the tailnet ACL to the other Mac (and the Spark) rather
  than leaving the service open to every device on the tailnet.
- Do **not** enable Tailscale Funnel. Serve/proxy is tailnet-only; Funnel is
  public internet.
- When the tailnet admin enables Tailscale Serve, prefer it over this proxy:

  ```bash
  tailscale serve --bg 8000
  # https://spark-a183.tail0c5008.ts.net/v1
  systemctl --user disable --now vllm-glm53fp4-tailscale.service
  ```

  Serve gives HTTPS and a tailnet-managed access layer; the proxy is the
  no-admin fallback.

- The proxy forwards every request it receives; it does not add a bearer key.
  If you need auth, enable vLLM's `--api-key` and configure the client, or put
  an authenticating reverse proxy in front.

## Boot behavior

- `vllm-glm53fp4-tailscale.service` is enabled; `Linger=yes` is already set.
- The proxy starts before or after vLLM; it retries every 5 seconds until the
  Tailscale IPv4 is bindable, and systemd restarts it on any exit.
- If the Tailscale node gets a new IPv4, restart the unit; the proxy detects
  the new address at startup.
