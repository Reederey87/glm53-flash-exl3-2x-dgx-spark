# 2026-10-17 — Tailscale exposure receipt

## Why a proxy

Tailscale is running on `spark1` (`100.125.23.98`,
`spark-a183.tail0c5008.ts.net`), but `tailscale serve` reports **“Serve is not
enabled on your tailnet”**. vLLM binds loopback only, so no tailnet client
could reach the API.

A small TCP proxy now binds **only the Tailscale IPv4** and forwards to
`127.0.0.1:8000`. It does not touch `0.0.0.0`, the LAN address, or the
loopback-only posture of the model container.

## Installed state

```text
vllm-glm53fp4-tailscale.service   active (running)   enabled
listener: 100.125.23.98:8000      -> 127.0.0.1:8000
listener: 127.0.0.1:8000          (vLLM, unchanged)
```

Log:

```text
2026-10-17 21:52:03,745 INFO listening on 100.125.23.98:8000 -> 127.0.0.1:8000
```

The unit is enabled under `default.target`; linger is already on. It retries
until the Tailscale address is bindable and `Restart=always` re-arms it.

## Verification through the Tailscale IP

From `spark1` (the listener itself):

```text
GET  http://100.125.23.98:8000/health          -> 200
GET  http://100.125.23.98:8000/v1/models       -> nvidia/GLM-5.3-Flash-FP4-Local
POST /v1/chat/completions                      -> TAILNET_OK
POST /v1/completions (stream=true)             -> SSE data chunks
```

The model server itself is still healthy on `127.0.0.1:8000`, KV pool
**2,078,358 tokens**, and the systemd serving unit remains `active`.

## Client URL for the other Mac

```text
http://100.125.23.98:8000/v1
```

or, with MagicDNS:

```text
http://spark-a183.tail0c5008.ts.net:8000/v1
```

DeepSeek Harness on that Mac needs a custom provider with model
`nvidia/GLM-5.3-Flash-FP4-Local`, `contextWindow: 480000`,
`maxTokens: 16384`, and the dummy `Authorization: Bearer local` header. The
full example is in [10-tailscale.md](../10-tailscale.md).

## Remaining gate

The head-side listener and protocol path are verified. I cannot test from the
other Mac from here. If it fails, the remaining boundary is the **tailnet ACL**
or local firewall: allow TCP 8000 from that Mac to `spark-a183`. Do not enable
Funnel; the API has no authentication.

## Files

- `scripts/tailscale_proxy.py`
- `systemd/vllm-glm53fp4-tailscale.service`
- `docs/10-tailscale.md`
