#!/usr/bin/env python3
"""Launcher contract for `GLM53_PROMPT_TOKENS_DETAILS` (W43 attribution).

The knob is report-only, but it must still hold the kit's launcher rules: a
strict 0/1 value that an explicitly empty string does not satisfy, the same
argv element in the head and worker inner scripts, the value forwarded into
both containers, and no route into `EXTRA_ARGS` (API-server args must stay out
of the engine's JIT shape hash).

The interesting test is the last one: the generated inner scripts carry the
flag behind a runtime guard, so a static grep proves nothing about which value
arms it. These tests generate the real inner scripts through the launcher's own
`write_inner_scripts` and then *execute* the extracted guard block under both
values, which is the only check that can fail when the guard is inverted.

Run:  python3 tests/test_prompt_tokens_details_wiring.py   (or pytest)
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"
ENV_EXAMPLE = ROOT / "env.example"

KNOB = "GLM53_PROMPT_TOKENS_DETAILS"
FLAG = "--enable-prompt-tokens-details"
BEGIN = "# LOCAL: W43 attribution (begin)"
END = "# LOCAL: W43 attribution (end)"
INNER = {
    "head": ".glm53-exl3-head.inner.sh",
    "worker": ".glm53-exl3-worker.inner.sh",
}
HOST_TOOLS = ("docker", "ssh", "scp", "rsync", "curl", "ip", "nvidia-smi")
STUB = "#!/usr/bin/env bash\nexit 0\n"


def bashes() -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for cand in (shutil.which("bash"), "/bin/bash", "/opt/homebrew/bin/bash", "/usr/local/bin/bash"):
        if not cand or not Path(cand).is_file() or not os.access(cand, os.X_OK):
            continue
        real = os.path.realpath(cand)
        if real in seen:
            continue
        seen.add(real)
        found.append(cand)
    assert found, "no bash found"
    return found


class Kit:
    """Throwaway launcher copy with every host tool stubbed off PATH."""

    def __init__(self, tmp: Path) -> None:
        self.repo = tmp / "repo"
        self.repo.mkdir()
        shutil.copy2(START, self.repo / "start.sh")
        shutil.copy2(ENV_EXAMPLE, self.repo / "env.example")
        text = (self.repo / "start.sh").read_text()
        assert text.rstrip().endswith('\nmain "$@"'), 'start.sh must end with main "$@"'
        (self.repo / "start.fn.sh").write_text(
            text.rstrip()[: -len('main "$@"')] + '"$@"\n'
        )
        self.bin = tmp / "bin"
        self.bin.mkdir()
        for tool in HOST_TOOLS:
            p = self.bin / tool
            p.write_text(STUB)
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def env(self, **extra: str) -> dict[str, str]:
        env = dict(os.environ)
        env["PATH"] = f"{self.bin}{os.pathsep}{env.get('PATH', '')}"
        env.update(extra)
        return env

    def write_env(self, body: str) -> None:
        (self.repo / ".env").write_text(body)

    def run(self, bash: str, args: list[str], **extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [bash, *args], cwd=self.repo, env=self.env(**extra),
            capture_output=True, text=True,
        )

    def write_inner_scripts(self, bash: str) -> None:
        proc = self.run(bash, ["start.fn.sh", "write_inner_scripts"])
        assert proc.returncode == 0, proc.stderr
        for name in INNER.values():
            assert (self.repo / name).is_file(), f"{name} was not generated"

    def guard_block(self, name: str) -> str:
        text = (self.repo / name).read_text()
        assert BEGIN in text, f"{BEGIN} missing from {name}"
        assert END in text, f"{END} missing from {name}"
        after = text.split(BEGIN, 1)[1]
        return after.split(END, 1)[0]

    def run_guard(self, bash: str, name: str, value: str | None) -> list[str]:
        """Execute the generated guard block and return the argv it produced."""
        script = self.repo / "guard.sh"
        script.write_text(
            "say() { :; }\nARGS=()\n"
            + self.guard_block(name)
            + '\nprintf "%s\\n" ${ARGS[@]+"${ARGS[@]}"}\n'
        )
        env = dict(os.environ)
        env.pop(KNOB, None)
        if value is not None:
            env[KNOB] = value
        proc = subprocess.run([bash, str(script)], capture_output=True, text=True,
                              env=env)
        assert proc.returncode == 0, proc.stderr
        return [line for line in proc.stdout.splitlines() if line]


def test_knob_is_documented_and_defaulted_on() -> None:
    assert f"{KNOB}=" in ENV_EXAMPLE.read_text(), f"{KNOB} must be documented in env.example"
    text = START.read_text()
    assert f'{KNOB}="${{{KNOB}-1}}"' in text, (
        "the launcher default must be 1 and unset-only, so an explicitly empty "
        "value stays a value the validator can reject"
    )


def test_flag_is_guarded_once_per_inner_script() -> None:
    text = START.read_text()
    armed = f"ARGS+=({FLAG})"
    assert text.count(armed) == 2, (
        f"{armed} must appear exactly once per inner script (head and worker); "
        f"found {text.count(armed)}"
    )
    for marker in (BEGIN, END):
        assert text.count(marker) == 2, f"{marker} must appear once per inner script"
    for line in text.splitlines():
        if FLAG in line and "EXTRA_ARGS" in line:
            raise AssertionError(
                "the flag must not be routed through EXTRA_ARGS: EXTRA_ARGS feeds "
                "the JIT shape hash and this is an API-server argument, not an "
                f"engine one ({line.strip()})"
            )


def test_knob_is_forwarded_to_both_containers() -> None:
    text = START.read_text()
    assert f'-e "{KNOB}=${KNOB}"' in text, "head container env must carry the knob"
    assert f" {KNOB}; do" in text, (
        "the worker serve_env loop must carry the knob; the worker builds its own "
        "argv from its own environment"
    )


def test_knob_is_in_the_strict_bool_validator() -> None:
    text = START.read_text()
    line = next((ln for ln in text.splitlines()
                 if ln.strip().startswith("for _v in GLM53_KV_CAPACITY_LOG")), "")
    assert KNOB in line, f"{KNOB} must be in the strict-bool validate loop"


def test_generated_guard_arms_the_flag_only_for_1() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        kit = Kit(Path(tmp))
        kit.write_env(f"{KNOB}=1\n")
        for bash in bashes():
            kit.write_inner_scripts(bash)
            for name in INNER.values():
                assert kit.run_guard(bash, name, "1") == [FLAG], (
                    f"{name}: the generated guard must add {FLAG} when the knob is 1"
                )
                assert kit.run_guard(bash, name, "0") == [], (
                    f"{name}: the generated guard must add nothing when the knob is 0"
                )
                assert kit.run_guard(bash, name, None) == [FLAG], (
                    f"{name}: an unset knob must default to armed (the container "
                    "env is set explicitly by the launcher, but a bare boot must "
                    "still match the documented default)"
                )


def test_validate_accepts_only_zero_or_one() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        kit = Kit(Path(tmp))
        bash = bashes()[0]
        for value, expected in (("0", 0), ("1", 0), ("", 2), ("bogus", 2), ("2", 2)):
            kit.write_env(f"{KNOB}={value}\n")
            proc = kit.run(bash, ["start.sh", "validate"])
            assert proc.returncode == expected, (
                f"{KNOB}={value!r}: expected rc={expected}, got {proc.returncode}: "
                f"{proc.stderr.strip()}"
            )
        kit.write_env("")
        proc = kit.run(bash, ["start.sh", "validate"])
        assert proc.returncode == 0, (
            f"an unset knob must validate through its default; got rc={proc.returncode}"
        )


def test_caller_export_wins_over_env_file() -> None:
    """The kit's `prefix env wins over .env` rule applies to this knob too."""
    with tempfile.TemporaryDirectory() as tmp:
        kit = Kit(Path(tmp))
        kit.write_env(f"{KNOB}=1\n")
        bash = bashes()[0]
        proc = kit.run(bash, ["start.sh", "validate"], **{KNOB: "0"})
        assert proc.returncode == 0, proc.stderr
        kit.write_env(f"{KNOB}=0\n")
        proc = kit.run(bash, ["start.sh", "validate"], **{KNOB: "bogus"})
        assert proc.returncode == 2, (
            "a caller-provided invalid value must be validated, not shadowed by .env"
        )


def main() -> int:
    tests = [
        test_knob_is_documented_and_defaulted_on,
        test_flag_is_guarded_once_per_inner_script,
        test_knob_is_forwarded_to_both_containers,
        test_knob_is_in_the_strict_bool_validator,
        test_generated_guard_arms_the_flag_only_for_1,
        test_validate_accepts_only_zero_or_one,
        test_caller_export_wins_over_env_file,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
