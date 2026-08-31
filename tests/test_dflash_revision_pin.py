#!/usr/bin/env python3
"""Regression guard: the DFlash2 drafter resolves a pinned revision (kit PR #67 shape).

incoai/GLM-5.3-Flash-DFlash2 has shipped three different model.safetensors under
the same model ID (7d74cdd -> dc77ff1 -> bf582e4). Without a pin, a REFRESH_WEIGHTS
or fresh node silently pulls whatever Hub main means that day. This deployment's
receipts were all taken on 7d74cdd (sha256 8931dc52...), so that is the default.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"
PIN = "7d74cdd881ed7e32c31175984a67823127b66cfe"


def source() -> str:
    return START.read_text(encoding="utf-8")


def function(name: str) -> str:
    match = re.search(rf"(?ms)^{re.escape(name)}\(\) \{{\n.*?^\}}\n", source())
    assert match, f"missing function {name}"
    return match.group(0)


def test_pin_is_declared_and_threaded_through() -> None:
    text = source()
    assert f'DFLASH_REVISION="${{DFLASH_REVISION-{PIN}}}"' in text
    assert 'dflash_args+=(--revision "$DFLASH_REVISION")' in text
    assert '[ -f "$dir/model.safetensors" ]' in function("resolve_dflash_dir")
    assert ('sync_repo_to_worker "$DFLASH_PATH" "$DFLASH_CACHE_NAME" '
            '"DFlash2 draft" "$DFLASH_REVISION"') in text


def test_download_and_resolution_use_the_pin() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        args_file = Path(tmp) / "hf-args"
        script = f"""
set -euo pipefail
log() {{ :; }}
die() {{ printf 'DIE:%s\\n' "$*" >&2; exit 97; }}
{function("ensure_dflash_refs_main")}
{function("resolve_dflash_dir")}
{function("download_dflash")}
resolve_hf_bin() {{ HF_BIN_CMD=(mock_hf); }}
mock_hf() {{
    printf '%s\\n' "$@" > "$ARGS_FILE"
    mkdir -p "$DFLASH_PATH/snapshots/$DFLASH_REVISION"
    touch "$DFLASH_PATH/snapshots/$DFLASH_REVISION/config.json"
    touch "$DFLASH_PATH/snapshots/$DFLASH_REVISION/model.safetensors"
}}
ARGS_FILE={shlex.quote(str(args_file))}
DFLASH_PATH={shlex.quote(str(Path(tmp) / "dflash"))}
DFLASH_CACHE_NAME=models--incoai--DFlash
DFLASH_MODEL=incoai/DFlash
DFLASH_REVISION={PIN}
HF_CACHE_DIR={shlex.quote(tmp)}
SPEC_METHOD=dflash
SKIP_DOWNLOAD=0
REFRESH_WEIGHTS=0
download_dflash
printf 'resolved=%s\\n' "$(resolve_dflash_dir)"
"""
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert args_file.read_text(encoding="utf-8").splitlines() == [
            "download", "incoai/DFlash", "--revision", PIN]
        assert result.stdout.strip().endswith(f"/snapshots/{PIN}")


def test_pinned_snapshot_wins_over_stale_refs_main() -> None:
    """refs/main pointing at a never-materialised newer commit must not break boot (issue #52 shape)."""
    with tempfile.TemporaryDirectory() as tmp:
        script = f"""
set -euo pipefail
log() {{ :; }}
die() {{ printf 'DIE:%s\\n' "$*" >&2; exit 97; }}
{function("ensure_dflash_refs_main")}
{function("resolve_dflash_dir")}
DFLASH_PATH={shlex.quote(str(Path(tmp) / "dflash"))}
DFLASH_CACHE_NAME=models--incoai--DFlash
DFLASH_REVISION={PIN}
mkdir -p "$DFLASH_PATH/snapshots/$DFLASH_REVISION" "$DFLASH_PATH/refs"
touch "$DFLASH_PATH/snapshots/$DFLASH_REVISION/config.json" "$DFLASH_PATH/snapshots/$DFLASH_REVISION/model.safetensors"
printf 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef' > "$DFLASH_PATH/refs/main"
resolve_dflash_dir
"""
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith(f"/snapshots/{PIN}")


if __name__ == "__main__":
    test_pin_is_declared_and_threaded_through()
    test_download_and_resolution_use_the_pin()
    test_pinned_snapshot_wins_over_stale_refs_main()
    print("dflash revision-pin guard OK")
