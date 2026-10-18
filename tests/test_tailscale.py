from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "tailscale_proxy.py"
    spec = importlib.util.spec_from_file_location("tailscale_proxy_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_tailscale_ipv4() -> None:
    module = load_module()
    assert module.parse_tailscale_ip("100.125.23.98\n") == "100.125.23.98"
    assert module.parse_tailscale_ip("fd7a:115c:a1e0::1\n100.64.0.2\n") == "100.64.0.2"
    assert module.parse_tailscale_ip("not an ip\n") == ""
    assert module.parse_tailscale_ip("999.1.1.1\n") == ""
