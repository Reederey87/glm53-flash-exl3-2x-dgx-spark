#!/usr/bin/env python3
"""The JIT shape guard in local/prod-start.sh, run standalone with mocked docker/ssh.

Asserts: the wipe container is resolved from IMAGE= (self-built tag, no ghcr
digest); the stamp advances only when BOTH node wipes succeed; a failed wipe
on either node leaves the stamp unchanged; an unresolvable IMAGE leaves it
unchanged; the hash covers DFLASH_REVISION (from .env or the launcher default).
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROD_START = (ROOT / "local" / "prod-start.sh").read_text(encoding="utf-8")


def wipe_helper() -> str:
    """The image resolver the wipe block calls; it is defined above the guard."""
    begin = PROD_START.index("wipe_image_for() {")
    end = PROD_START.index("\n}\n", begin) + 3
    return PROD_START[begin:end]


def guard_block() -> str:
    begin = PROD_START.index("# --- JIT-cache config-shape guard")
    end = PROD_START.index("# --- start, with a bounded retry")
    return PROD_START[begin:end]


def retry_block() -> str:
    return PROD_START[PROD_START.index("# --- start, with a bounded retry") :]


def run_retry(
    tmp: Path,
    fail_count: int,
    rc: int = 17,
    preflight_ok: bool = True,
    max_attempts: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], int, list[str]]:
    """Run the bounded boot-retry loop with a fake start.sh.

    start.sh fails with `rc` for its first `fail_count` invocations of `start`,
    then succeeds. `preflight_ok=False` models a deterministic environment
    failure that no amount of retrying can fix. Returns the result, the number
    of `start` invocations, and every subcommand the loop issued in order.
    """
    work = tmp / "kit"
    work.mkdir(exist_ok=True)
    calls = tmp / "start-calls.log"
    issued = tmp / "issued.log"
    start = work / "start.sh"
    start.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$1" >> {issued}\n'
        'case "$1" in\n'
        f'  start) echo start >> {calls}\n'
        f'         n=$(wc -l < {calls} | tr -d " ")\n'
        f'         if [ "$n" -le {fail_count} ]; then exit {rc}; fi\n'
        "         exit 0 ;;\n"
        f'  preflight) exit {0 if preflight_ok else 1} ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    start.chmod(start.stat().st_mode | stat.S_IEXEC)
    bind = ""
    if max_attempts is not None:
        # Single-quoted so a hostile value (spaces, globs, `$(...)`) reaches the
        # loop as data rather than being expanded by the outer shell.
        bind = f"MAX_BOOT_ATTEMPTS='{max_attempts}'\n"
    script = (
        "set -uo pipefail\n"
        "log() { printf '[prod-start] %s\\n' \"$*\"; }\n"
        "settle_wait() { :; }\n" + bind + retry_block()
    )
    r = subprocess.run(
        ["bash", "-c", script], cwd=work, capture_output=True, text=True
    )
    n = len(calls.read_text().splitlines()) if calls.exists() else 0
    order = issued.read_text().splitlines() if issued.exists() else []
    return r, n, order


def make_shim(bindir: Path, name: str, body: str) -> None:
    p = bindir / name
    p.write_text("#!/usr/bin/env bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def run_prod_start(
    tmp: Path, env_text: str
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    work = tmp / "kit"
    local = work / "local"
    local.mkdir(parents=True)
    shutil.copy2(ROOT / "local" / "prod-start.sh", local / "prod-start.sh")
    calls = tmp / "start-calls.log"
    (work / ".env").write_text(env_text)
    start = work / "start.sh"
    start.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$1" >> {calls}\n'
        'if [ "$1" = validate ]; then\n'
        "  source .env\n"
        '  if [ "${SPEC_METHOD:-dflash}" = dflash ] '
        '&& [ "${DFLASH_TOKENS:-7}" != 7 ]; then exit 2; fi\n'
        "fi\n"
    )
    start.chmod(start.stat().st_mode | stat.S_IEXEC)
    home = tmp / "home"
    home.mkdir()
    bindir = tmp / "bin"
    bindir.mkdir()
    external = tmp / "external-calls.log"
    make_shim(
        bindir,
        "ssh",
        f'case "$*" in *MemFree*) echo 100 ;; *) echo "ssh $*" >> {external} ;; esac\n',
    )
    make_shim(bindir, "docker", f'echo "docker $*" >> {external}\n')
    result = subprocess.run(
        ["bash", str(local / "prod-start.sh")],
        env={
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "HOME": str(home),
            "NEED_GIB": "0",
            "SETTLE_TIMEOUT": "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return result, calls, external


def run_guard(
    tmp: Path,
    env_text: str,
    docker_rc: int,
    ssh_rc: int,
    stamp: str | None = None,
    start_sh_default: str = 'DFLASH_REVISION="${DFLASH_REVISION-abc}"\n',
    inspect_ok: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    work = tmp / "kit"
    work.mkdir(exist_ok=True)
    (work / ".env").write_text(env_text)
    (work / "start.sh").write_text(start_sh_default)
    home = tmp / "home"
    (home / ".cache" / "vllm-glm53-flash").mkdir(parents=True, exist_ok=True)
    stamp_path = home / ".cache" / "vllm-glm53-flash" / ".config-shape"
    if stamp is not None:
        stamp_path.write_text(stamp)
    bindir = tmp / "bin"
    bindir.mkdir(exist_ok=True)
    log = tmp / "calls.log"
    # The wipe resolves an image the node actually has before running it, so
    # `image inspect` / `images` must succeed and print a tag; only the wipe
    # container itself honours the injected return code. inspect_ok=False
    # models the task-35 failure: the requested tag is not on the node yet.
    inspect_rc = 0 if inspect_ok else 1
    inspect_out = 'echo "glm53-selfbuild:b5ab8091-w15a"' if inspect_ok else ":"
    make_shim(
        bindir,
        "docker",
        f'echo "docker $*" >> {log}\n'
        'case "$1" in\n'
        f'  image) {inspect_out}; exit {inspect_rc} ;;\n'
        '  images) echo "glm53-selfbuild:b5ab8091-w15a"; exit 0 ;;\n'
        f'  *) exit {docker_rc} ;;\n'
        "esac\n",
    )
    make_shim(
        bindir,
        "ssh",
        f'echo "ssh $*" >> {log}\n'
        'case "$*" in\n'
        # The remote resolver is ONE command holding both the requested-tag
        # inspect and its fallback, so it reports a tag either way; only the
        # wipe container honours the injected return code.
        '  *"docker image inspect"*) echo "glm53-selfbuild:b5ab8091-w15a"; exit 0 ;;\n'
        f'  *"docker run"*) exit {ssh_rc} ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n",
    )
    script = "set -uo pipefail\nWORKER_SSH=nvidia@worker\n" + wipe_helper() + "\n" + guard_block()
    env = {"PATH": f"{bindir}:{os.environ['PATH']}", "HOME": str(home)}
    r = subprocess.run(["bash", "-c", script], cwd=work, env=env, capture_output=True, text=True)
    return r, stamp_path, log


ENV = "IMAGE=glm53-selfbuild:b5ab8091-w15a\nDFLASH_TOKENS=7\nDFLASH_REVISION=abc\n"


def test_invalid_dflash_length_exits_before_stop_or_cache_handling() -> None:
    with tempfile.TemporaryDirectory() as t:
        result, calls, external = run_prod_start(
            Path(t),
            "SPEC_METHOD=dflash\nDFLASH_TOKENS=4\n",
        )
        assert result.returncode == 2
        assert calls.read_text().splitlines() == ["validate"]
        assert not external.exists()
        assert "production left untouched" in result.stdout


def test_native_dflash_and_non_dflash_continue_after_validation() -> None:
    for env_text in (
        "SPEC_METHOD=dflash\nDFLASH_TOKENS=7\n",
        "SPEC_METHOD=none\nDFLASH_TOKENS=4\n",
    ):
        with tempfile.TemporaryDirectory() as t:
            result, calls, _ = run_prod_start(Path(t), env_text)
            assert result.returncode == 0, result.stderr
            assert calls.read_text().splitlines() == ["validate", "stop", "start"]


def test_image_resolved_from_IMAGE_line_and_stamp_written_on_success() -> None:
    with tempfile.TemporaryDirectory() as t:
        r, stamp, log = run_guard(Path(t), ENV, 0, 0, stamp="stale")
        assert r.returncode == 0, r.stderr
        calls = log.read_text()
        lines = calls.splitlines()
        assert "glm53-selfbuild:b5ab8091-w15a" in calls
        # one resolve (image inspect) + one wipe (docker run) per node
        assert sum(line.startswith("docker ") for line in lines) == 2
        assert sum(line.startswith("ssh ") for line in lines) == 2
        assert stamp.read_text().strip() != "stale"
        assert len(stamp.read_text().strip()) == 32


def test_wipe_falls_back_to_a_locally_present_tag() -> None:
    """Task 35: the requested tag may not be on the node yet (start.sh ships it
    later), so the wipe must fall back to any local glm53-selfbuild image —
    otherwise the worker wipe dies with `pull access denied` and half-wipes."""
    with tempfile.TemporaryDirectory() as t:
        r, stamp, log = run_guard(Path(t), ENV, 0, 0, stamp="stale", inspect_ok=False)
        assert r.returncode == 0, r.stderr
        calls = log.read_text()
        assert "glm53-selfbuild:b5ab8091-w15a" in calls
        assert "incomplete" not in r.stdout
        assert stamp.read_text().strip() != "stale"
        # both nodes wiped with the fallback tag, none with the requested one
        assert calls.count("docker run") == 2
        assert "glm53-selfbuild:b5ab8091-w15a" in calls


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


def test_retry_loop_exhaustion_returns_the_failure_status() -> None:
    """Regression: `rc=$?` placed after the completed `if` read the
    if-statement's own status, which is 0 when the condition failed and no
    branch ran — so exhausting the retries exited 0 with production down."""
    with tempfile.TemporaryDirectory() as t:
        r, n, _ = run_retry(Path(t), fail_count=99, rc=17)
        assert n == 3, n
        assert r.returncode == 17, (r.returncode, r.stdout, r.stderr)
        assert "production left down" in r.stdout


def test_retry_loop_stops_at_the_first_success() -> None:
    with tempfile.TemporaryDirectory() as t:
        r, n, _ = run_retry(Path(t), fail_count=1)
        assert r.returncode == 0, r.stderr
        assert n == 2, n
        assert "production left down" not in r.stdout
        assert "retrying" in r.stdout


def test_each_retry_cleans_up_the_partial_launch_first() -> None:
    """A failed attempt can leave containers running (start.sh only removes
    them inside launch_cluster, so a failure before that point leaks them,
    holding unified memory and the API/master ports). Every retry must tear the
    pair down again before trying."""
    with tempfile.TemporaryDirectory() as t:
        r, n, order = run_retry(Path(t), fail_count=1)
        assert n == 2, n
        # start -> stop -> preflight -> start
        assert order == ["start", "stop", "preflight", "start"], order
        assert "cleaning up any partial launch" in r.stdout


def test_a_deterministic_failure_is_not_retried() -> None:
    """An unresolvable RoCE GID or a downed RDMA port fails identically every
    time. Burning the remaining attempts on it produced the confusing
    "3 attempts failed" report from a single configuration problem."""
    with tempfile.TemporaryDirectory() as t:
        r, n, order = run_retry(Path(t), fail_count=99, rc=17, preflight_ok=False)
        assert n == 1, f"should abort after the first attempt, ran {n}"
        assert order == ["start", "stop", "preflight", "preflight"], order
        assert r.returncode == 17
        assert "not a transient one" in r.stdout
        assert "not retrying" in r.stdout


def test_cleanup_still_runs_before_the_deterministic_abort() -> None:
    """The abort path must not skip cleanup: leaving the partial launch running
    is exactly what made the next manual start fail preflight."""
    with tempfile.TemporaryDirectory() as t:
        _, _, order = run_retry(Path(t), fail_count=99, rc=17, preflight_ok=False)
        assert order.index("stop") < order.index("preflight")


def test_cleanup_failure_does_not_abort_the_retry() -> None:
    """`stop` returning non-zero is reported but not fatal — the retry may still
    succeed, and the final state is reported either way."""
    with tempfile.TemporaryDirectory() as t:
        work = Path(t) / "kit"
        work.mkdir(exist_ok=True)
        calls = Path(t) / "start-calls.log"
        start = work / "start.sh"
        start.write_text(
            "#!/usr/bin/env bash\n"
            'case "$1" in\n'
            f'  start) echo start >> {calls}\n'
            f'         n=$(wc -l < {calls} | tr -d " ")\n'
            '         if [ "$n" -le 1 ]; then exit 17; fi\n'
            "         exit 0 ;;\n"
            "  stop) echo 'stop refused' >&2; exit 5 ;;\n"
            "  *) exit 0 ;;\n"
            "esac\n"
        )
        start.chmod(start.stat().st_mode | stat.S_IEXEC)
        script = (
            "set -uo pipefail\n"
            "log() { printf '[prod-start] %s\\n' \"$*\"; }\n"
            "settle_wait() { :; }\n" + retry_block()
        )
        r = subprocess.run(["bash", "-c", script], cwd=work, capture_output=True, text=True)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert "cleanup stop returned non-zero" in r.stdout


def test_exhaustion_still_cleans_up_the_partial_launch() -> None:
    """Regression (review round five): the exhaustion branch exited BEFORE the
    cleanup, so the last failed attempt's containers were left running. The
    report said "production left down" while the wreckage held unified memory
    and the API/master ports, which is what made the next manual start fail."""
    with tempfile.TemporaryDirectory() as t:
        r, n, order = run_retry(Path(t), fail_count=99, rc=17, max_attempts="1")
        assert n == 1, n
        assert order == ["start", "stop"], order
        assert r.returncode == 17, (r.returncode, r.stdout)
        assert "cleaning up any partial launch" in r.stdout
        assert "production left down" in r.stdout


def test_a_non_numeric_max_boot_attempts_does_not_disable_the_bound() -> None:
    """Regression (review round five): `[ "$attempt" -ge "$MAX_BOOT_ATTEMPTS" ]`
    is false for a non-numeric right-hand side, so `MAX_BOOT_ATTEMPTS=bogus`
    removed the bound entirely and spun forever on a failure that still passed
    preflight. A bad value now falls back to the default of 3."""
    with tempfile.TemporaryDirectory() as t:
        r, n, _ = run_retry(Path(t), fail_count=99, rc=17, max_attempts="bogus")
        assert n == 3, f"expected the default of 3 attempts, ran {n}"
        assert r.returncode == 17
        assert "not a positive integer" in r.stdout


def test_a_sub_one_max_boot_attempts_falls_back_to_the_default() -> None:
    with tempfile.TemporaryDirectory() as t:
        r, n, _ = run_retry(Path(t), fail_count=99, rc=17, max_attempts="0")
        assert n == 3, n
        assert r.returncode == 17
        assert "below 1" in r.stdout


def test_hash_tracks_revision_and_launcher_default() -> None:
    with tempfile.TemporaryDirectory() as t:
        _, s1, _ = run_guard(Path(t), ENV, 0, 0)
        h1 = s1.read_text()
    with tempfile.TemporaryDirectory() as t:
        _, s2, _ = run_guard(
            Path(t),
            ENV.replace("DFLASH_REVISION=abc", "DFLASH_REVISION=def"),
            0,
            0,
        )
        h2 = s2.read_text()
    with tempfile.TemporaryDirectory() as t:
        _, s3, _ = run_guard(
            Path(t),
            ENV,
            0,
            0,
            start_sh_default='DFLASH_REVISION="${DFLASH_REVISION-zzz}"\n',
        )
        h3 = s3.read_text()
    assert h1 != h2 and h1 != h3


if __name__ == "__main__":
    test_invalid_dflash_length_exits_before_stop_or_cache_handling()
    test_native_dflash_and_non_dflash_continue_after_validation()
    test_image_resolved_from_IMAGE_line_and_stamp_written_on_success()
    test_wipe_falls_back_to_a_locally_present_tag()
    test_head_wipe_failure_leaves_stamp()
    test_worker_wipe_failure_leaves_stamp()
    test_unresolvable_image_leaves_stamp_and_skips_wipe()
    test_retry_loop_exhaustion_returns_the_failure_status()
    test_retry_loop_stops_at_the_first_success()
    test_each_retry_cleans_up_the_partial_launch_first()
    test_a_deterministic_failure_is_not_retried()
    test_cleanup_still_runs_before_the_deterministic_abort()
    test_cleanup_failure_does_not_abort_the_retry()
    test_exhaustion_still_cleans_up_the_partial_launch()
    test_a_non_numeric_max_boot_attempts_does_not_disable_the_bound()
    test_a_sub_one_max_boot_attempts_falls_back_to_the_default()
    test_hash_tracks_revision_and_launcher_default()
    print("prod-start guard tests OK")
