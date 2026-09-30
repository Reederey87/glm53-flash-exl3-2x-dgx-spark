"""CI must inspect shipped files and fail closed without printing secrets."""
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("scan_status,passes", [(0, False), (1, True), (128, False)])
def test_secret_guard_checks_status_and_requires_quiet_scan(tmp_path, scan_status, passes):
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    code = textwrap.dedent(workflow.split(
        "      - name: no secrets or private config committed\n", 1
    )[1].split("        run: |\n", 1)[1])
    fake_git = tmp_path / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        '[ "$1" = grep ] && [ "$2" = -q ] || exit 128\n'
        f"exit {scan_status}\n"
    )
    fake_git.chmod(0o755)
    result = subprocess.run(
        ["bash", "-e", "-c", code], cwd=tmp_path,
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True, text=True, check=False,
    )
    assert (result.returncode == 0) is passes
    assert ("secret scan OK" in result.stdout) is passes


def test_shell_ci_scope_is_tracked_files_not_installed_dependencies():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "git ls-files -z '*.sh' | xargs -0 shellcheck" in workflow
    assert "git ls-files -z '*.sh' | xargs -0 -n1 bash -n" in workflow
    assert "find . -name '*.sh'" not in workflow
