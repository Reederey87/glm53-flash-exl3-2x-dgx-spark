#!/usr/bin/env python3
"""Reapply the existing reserve to clear accumulated kernel watermark boosts.

Run only after the host verifies BOTH nodes idle. The sole writable host mount
is the exact min_free_kbytes proc file. No configured numeric value changes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

TARGET = Path('/host-min-free-kbytes')
SETTINGS = {'min_free_kbytes': '1048576', 'watermark_scale_factor': '10',
            'watermark_boost_factor': '15000'}


def verify_mount(text: str, apply: bool) -> None:
    rows = [line.split() for line in text.splitlines()]
    matches = [r for r in rows if len(r) >= 10 and r[4] == str(TARGET)]
    if len(matches) != 1:
        raise ValueError('requires exactly one reserve-file mount')
    row = matches[0]
    if ('-' not in row or len(row) - row.index('-') < 4
            or row[row.index('-') + 1] != 'proc'
            or row[3] != '/sys/vm/min_free_kbytes'
            or ('rw' if apply else 'ro') not in row[5].split(',')):
        raise ValueError('requires the exact proc reserve file and expected mount mode')
    if any(len(r) >= 5 and r[4].startswith(str(TARGET) + '/') for r in rows):
        raise ValueError('nested reserve mount')


def zones(text: str) -> dict:
    result = {}
    current = None
    for line in text.splitlines():
        match = re.fullmatch(r'Node\s+(\d+), zone\s+(\w+)', line)
        if match:
            current = ':'.join(match.groups())
            if current in result:
                raise ValueError('duplicate zone')
            result[current] = {}
        match = re.fullmatch(r'        (boost|min|low|high|managed)\s+(\d+)', line)
        if match and current is not None:
            if match[1] in result[current]:
                raise ValueError('duplicate zone field')
            result[current][match[1]] = int(match[2])
    if not result or any(set(x) != {'boost', 'min', 'low', 'high', 'managed'}
                         for x in result.values()):
        raise ValueError('incomplete zone watermarks')
    return result


def verify_settings(settings: dict) -> None:
    if settings != SETTINGS:
        raise ValueError('existing reserve/scale/boost settings differ; refusing change')


def verify_after(before: dict, after: dict) -> None:
    verify_settings(after['settings'])
    if before['settings'] != after['settings'] or before['zones'].keys() != after['zones'].keys():
        raise ValueError('configured settings or zone set changed')
    for name, old in before['zones'].items():
        new = after['zones'][name]
        if old['managed'] != new['managed']:
            raise ValueError('managed zone size changed')
        if any(old[key] - old['boost'] != new[key] - new['boost']
               for key in ('min', 'low', 'high')):
            raise ValueError('base watermarks changed')
        if new['boost'] != 0:
            raise ValueError('watermark boost was not cleared; do not start serving')


def snapshot() -> dict:
    wanted = {'MemFree', 'MemAvailable', 'Cached', 'SReclaimable'}
    return {
        'settings': {key: (Path('/proc/sys/vm') / key).read_text().strip()
                     for key in SETTINGS},
        'zones': zones(Path('/proc/zoneinfo').read_text()),
        'meminfo_kib': {line.split(':', 1)[0]: int(line.split()[1])
                        for line in Path('/proc/meminfo').read_text().splitlines()
                        if line.split(':', 1)[0] in wanted},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    identity = os.uname()
    if (identity.sysname, identity.machine, identity.release) != (
            'Linux', 'aarch64', '6.17.0-1031-nvidia'):
        raise ValueError('requires the qualified ARM64 Spark kernel')
    status = Path('/proc/self/status').read_text().splitlines()
    caps = [line.split()[1] for line in status if line.startswith('CapEff:')]
    if len(caps) != 1 or int(caps[0], 16) != 0:
        raise ValueError('container must drop all capabilities')
    verify_mount(Path('/proc/self/mountinfo').read_text(), args.apply)
    before = snapshot()
    verify_settings(before['settings'])
    flags = (os.O_RDWR if args.apply else os.O_RDONLY) | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(TARGET, flags)
    try:
        if os.read(fd, 32).strip() != b'1048576':
            raise ValueError('mounted reserve value differs')
        if args.apply:
            os.lseek(fd, 0, os.SEEK_SET)
            if os.write(fd, b'1048576\n') != 8:
                raise RuntimeError('incomplete reserve write')
            os.lseek(fd, 0, os.SEEK_SET)
            if os.read(fd, 32).strip() != b'1048576':
                raise RuntimeError('reserve changed after same-value write')
    finally:
        os.close(fd)
    after = snapshot()
    if args.apply:
        verify_after(before, after)
    print(json.dumps({'ok': True, 'applied': args.apply, 'before': before,
                      'after': after, 'configured_values_unchanged': before['settings'] == after['settings']}, indent=2))


if __name__ == '__main__':
    main()
