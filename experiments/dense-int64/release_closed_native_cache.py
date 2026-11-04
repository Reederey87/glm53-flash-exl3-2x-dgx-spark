#!/usr/bin/env python3
"""Advise clean pages from four exact, terminal native-test cache directories.

Host must verify every build/test/inspection and both serving ranks are stopped.
This preserves all artifacts, including the failed split-K16 qualification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from release_build_cache import meminfo, release_tree

ROOT = Path('/completed-native')
SOURCE = '/home/nvidia/glm53-native-tests/int64-20261103/results'
SUMMARIES = {
    'head-candidate-runtimepin-bad0f200': '7370927724ccbf665a5c34cf16f7cb62ef6230180abf7690711f07a32ef86a6c',
    'head-candidate-split8-a5901ed': '8abe0172390f387bfd50008cb29f86df58bba28c1c4a6a6a43cab4103c7d3e4d',
    'head-control-frontend131-40f4c522': '22a6174d22725f1d2edf8f65367374be93401b64e8dc733fbf1cdb7b2b862dda',
    'head-control-runtimepin-3b5da79c': '9f92cc7d3a2a3097714503e8c00d8bbcdbc36e6faed83e7648264c841d30929c',
}


def verify_mount(text: str) -> None:
    rows = [line.split() for line in text.splitlines()]
    matches = [r for r in rows if len(r) >= 10 and r[4] == str(ROOT)]
    if (len(matches) != 1 or matches[0][3] != SOURCE
            or 'ro' not in matches[0][5].split(',')
            or any(len(r) >= 5 and r[4].startswith(str(ROOT) + '/') for r in rows)):
        raise ValueError('requires the exact completed results mounted read-only without nested mounts')


def release(root: Path) -> dict:
    # Validate ALL identities before advising ANY cache.
    originals = {}
    for name, expected in SUMMARIES.items():
        folder = root / name
        summary, cache = folder / 'native-summary.json', folder / 'cache'
        if (folder.is_symlink() or summary.is_symlink() or cache.is_symlink()
                or not cache.is_dir() or folder.resolve() != folder):
            raise ValueError('linked or missing completed result/cache')
        raw = summary.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('native qualification receipt differs')
        originals[name] = raw
    before = meminfo()
    reports = {name: release_tree(root / name / 'cache') for name in SUMMARIES}
    if any((root / name / 'native-summary.json').read_bytes() != raw
           for name, raw in originals.items()):
        raise RuntimeError('qualification receipt changed')
    return {'summaries': SUMMARIES, 'receipts_unchanged': True, 'caches': reports,
            'meminfo_before_kib': before, 'meminfo_after_kib': meminfo()}


if __name__ == '__main__':
    verify_mount(Path('/proc/self/mountinfo').read_text())
    print(json.dumps(release(ROOT), indent=2))
