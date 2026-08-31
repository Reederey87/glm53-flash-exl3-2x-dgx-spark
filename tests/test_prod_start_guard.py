#!/usr/bin/env python3
"""The JIT shape guard in local/prod-start.sh, run standalone with mocked docker/ssh.

Asserts: the wipe container is resolved from IMAGE= (self-built tag, no ghcr
digest); the stamp advances only when BOTH node wipes succeed; a failed wipe
on either node leaves the stamp unchanged; an unresolvable IMAGE leaves it
unchanged; the hash covers DFLASH_REVISION (from .env or the launcher default).
"""
from __future__ import annotations

import os
import stat
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROD_START = (ROOT / "local" / "prod-start.sh").read_text(encoding="utf-8")


def guard_block() -> str:
    begin = PROD_START.index("# --- JIT-cache config-shape guard")
    end = PROD_START.index("exec ./start.sh start")
    return PROD_START[begin:end]


def make_shim(bindir: Path, name: str, body: str) -> None:
    p = bindir / name
    p.write_text("#!/usr/bin/env bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def run_guard(tmp: Path, env_text: str, docker_rc: int, ssh_rc: int, stamp: str | None = None,
              start_sh_default: str = 'DFLASH_REVISION="${DFLASH_REVISION-abc}"\n') -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    work = tmp / "kit"; work.mkdir(exist_ok=True)
    (work / ".env").write_text(env_text)
    (work / "start.sh").write_text(start_sh_default)
    home = tmp / "home"; (home / ".cache" / "vllm-glm53-flash").mkdir(parents=True, exist_ok=True)
    stamp_path = home / ".cache" / "vllm-glm53-flash" / ".config-shape"
    if stamp is not None:
        stamp_path.write_text(stamp)
    bindir = tmp / "bin"; bindir.mkdir(exist_ok=True)
    log = tmp / "calls.log"
    make_shim(bindir, "docker", f'echo "docker $*" >> {log}\nexit {docker_rc}\n')
    make_shim(bindir, "ssh", f'echo "ssh $*" >> {log}\nexit {ssh_rc}\n')
    script = "set -uo pipefail\nWORKER_SSH=nvidia@worker\n" + guard_block()
    env = {"PATH": f"{bindir}:{os.environ['PATH']}", "HOME": str(home)}
    r = subprocess.run(["bash", "-c", script], cwd=work, env=env, capture_output=True, text=True)
    return r, stamp_path, log


ENV = "IMAGE=glm53-selfbuild:b5ab8091-w15a\nDFLASH_TOKENS=7\nDFLASH_REVISION=abc\n"


def test_image_resolved_from_IMAGE_line_and_stamp_written_on_success() -> None:
    with tempfile.TemporaryDirectory() as t:
        r, stamp, log = run_guard(Path(t), ENV, 0, 0, stamp="stale")
        assert r.returncode == 0, r.stderr
        calls = log.read_text()
        lines = calls.splitlines()
        assert "glm53-selfbuild:b5ab8091-w15a" in calls
        assert sum(l.startswith("docker ") for l in lines) == 1 and sum(l.startswith("ssh ") for l in lines) == 1
        assert stamp.read_text().strip() != "stale" and len(stamp.read_text().strip()) == 32


def test_head_wipe_failure_leaves_stamp() -> None:
    with tempfile.TemporaryDirectory() as t:
        r, stamp, _ = run_guard(Path(t), ENV, 1, 0, stamp="stale")
        assert stamp.read_text() == "stale" and "incomplete" in r.stdout


def test_worker_wipe_failure_leaves_stamp() -> None:
    with tempfile.TemporaryDirectory() as t:
        r, stamp, _ = run_guard(Path(t), ENV, 0, 1, stamp="stale")
        assert stamp.read_text() == "stale" and "incomplete" in r.stdout


def test_unresolvable_image_leaves_stamp_and_skips_wipe() -> None:
    with tempfile.TemporaryDirectory() as t:
        r, stamp, log = run_guard(Path(t), "DFLASH_TOKENS=7\n", 0, 0, stamp="stale")
        assert stamp.read_text() == "stale" and "could not resolve IMAGE" in r.stdout
        assert not log.exists()


def test_hash_tracks_revision_and_launcher_default() -> None:
    with tempfile.TemporaryDirectory() as t:
        _, s1, _ = run_guard(Path(t), ENV, 0, 0)
        h1 = s1.read_text()
    with tempfile.TemporaryDirectory() as t:
        _, s2, _ = run_guard(Path(t), ENV.replace("DFLASH_REVISION=abc", "DFLASH_REVISION=def"), 0, 0)
        h2 = s2.read_text()
    with tempfile.TemporaryDirectory() as t:
        _, s3, _ = run_guard(Path(t), ENV, 0, 0, start_sh_default='DFLASH_REVISION="${DFLASH_REVISION-zzz}"\n')
        h3 = s3.read_text()
    assert h1 != h2 and h1 != h3


if __name__ == "__main__":
    test_image_resolved_from_IMAGE_line_and_stamp_written_on_success()
    test_head_wipe_failure_leaves_stamp()
    test_worker_wipe_failure_leaves_stamp()
    test_unresolvable_image_leaves_stamp_and_skips_wipe()
    test_hash_tracks_revision_and_launcher_default()
    print("prod-start guard tests OK")
