#!/usr/bin/env python3
"""Expose the loopback vLLM API on the host's Tailscale IPv4 address.

Tailscale Serve is disabled on this tailnet, so this is a small userspace TCP
proxy. It binds only the Tailscale IPv4 address (never 0.0.0.0 or the LAN
interface), forwards to 127.0.0.1:8000, and exits when the address is absent;
the systemd unit restarts it. Tailscale already encrypts the path with
WireGuard; the API itself has no authentication, so the tailnet ACL is the
boundary.

Usage:
  python3 scripts/tailscale_proxy.py
  python3 scripts/tailscale_proxy.py --bind-ip 100.125.23.98 --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys

TARGET_HOST = "127.0.0.1"
TARGET_PORT = 8000
BIND_PORT = 8000
RETRY_SECONDS = 5

LOG = logging.getLogger("tailscale-proxy")


def parse_tailscale_ip(text: str) -> str:
    """Return the first IPv4 address in ``tailscale ip -4`` output."""
    for line in text.splitlines():
        candidate = line.strip()
        parts = candidate.split(".")
        if len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts):
            return candidate
    return ""


def detect_tailscale_ip() -> str:
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return parse_tailscale_ip(result.stdout)


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    finally:
        try:
            writer.write_eof()
        except (OSError, RuntimeError):
            pass


async def handle_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_host: str,
    target_port: int,
) -> None:
    peer = client_writer.get_extra_info("peername")
    try:
        remote_reader, remote_writer = await asyncio.open_connection(target_host, target_port)
    except OSError as exc:
        LOG.warning("target %s:%s unavailable for %s: %s", target_host, target_port, peer, exc)
        client_writer.close()
        try:
            await client_writer.wait_closed()
        except OSError:
            pass
        return

    try:
        await asyncio.gather(
            pipe(client_reader, remote_writer),
            pipe(remote_reader, client_writer),
        )
    except (OSError, asyncio.CancelledError):
        pass
    finally:
        for writer in (client_writer, remote_writer):
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, RuntimeError):
                pass


async def serve(bind_ip: str, port: int, target_host: str, target_port: int) -> None:
    server = await asyncio.start_server(
        lambda reader, writer: handle_client(reader, writer, target_host, target_port),
        host=bind_ip,
        port=port,
        reuse_address=True,
    )
    LOG.info("listening on %s:%s -> %s:%s", bind_ip, port, target_host, target_port)
    async with server:
        await server.serve_forever()


async def main_async(args: argparse.Namespace) -> int:
    while True:
        bind_ip = args.bind_ip or detect_tailscale_ip()
        if not bind_ip:
            LOG.warning("no Tailscale IPv4 address yet; retrying in %ss", RETRY_SECONDS)
            await asyncio.sleep(RETRY_SECONDS)
            continue
        try:
            await serve(bind_ip, args.port, args.target_host, args.target_port)
            return 0
        except asyncio.CancelledError:
            raise
        except OSError as exc:
            LOG.warning("cannot bind %s:%s (%s); retrying in %ss", bind_ip, args.port, exc, RETRY_SECONDS)
            await asyncio.sleep(RETRY_SECONDS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bind-ip", default="", help="Tailscale IPv4 (default: detect with tailscale ip -4)")
    parser.add_argument("--port", type=int, default=BIND_PORT, help="tailnet-facing port")
    parser.add_argument("--target-host", default=TARGET_HOST)
    parser.add_argument("--target-port", type=int, default=TARGET_PORT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
