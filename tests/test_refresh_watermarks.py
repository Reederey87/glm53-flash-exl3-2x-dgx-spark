import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('refresh_watermarks', Path(__file__).resolve().parents[1] / 'experiments/startup-memory/refresh_watermarks.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_same_base_watermarks_with_boost_removed():
    old = {'boost': 586588, 'min': 847295, 'low': 912471, 'high': 977647, 'managed': 31725389}
    new = dict(old, boost=0, min=260707, low=325883, high=391059)
    module.verify_after({'settings': module.SETTINGS, 'zones': {'0:Normal': old}},
                        {'settings': module.SETTINGS, 'zones': {'0:Normal': new}})


@pytest.mark.parametrize('fault', ['reserve', 'scale', 'factor', 'base', 'boost', 'zone', 'managed'])
def test_refuses_changed_controls_or_incomplete_reset(fault):
    before = {'settings': dict(module.SETTINGS), 'zones': {'0:Normal': {'boost': 10, 'min': 110, 'low': 120, 'high': 130, 'managed': 1000}}}
    after = copy.deepcopy(before)
    after['zones']['0:Normal'].update(boost=0, min=100, low=110, high=120)
    if fault in ('reserve', 'scale', 'factor'):
        key = {'reserve': 'min_free_kbytes', 'scale': 'watermark_scale_factor', 'factor': 'watermark_boost_factor'}[fault]
        after['settings'][key] = '0'
    elif fault == 'zone':
        after['zones']['0:Other'] = after['zones'].pop('0:Normal')
    else:
        key = {'base': 'low', 'boost': 'boost', 'managed': 'managed'}[fault]
        after['zones']['0:Normal'][key] += 1
    with pytest.raises(ValueError):
        module.verify_after(before, after)


@pytest.mark.parametrize('fault', ['wrong_file', 'wrong_fs', 'readonly', 'duplicate', 'missing'])
def test_refuses_wrong_write_target(fault):
    line = '6495 5207 0:26 /sys/vm/min_free_kbytes /host-min-free-kbytes rw,nosuid,nodev,noexec - proc proc rw'
    if fault == 'wrong_file': line = line.replace('/sys/vm/min_free_kbytes', '/sys/vm/drop_caches')
    if fault == 'wrong_fs': line = line.replace('- proc proc', '- ext4 disk')
    if fault == 'readonly': line = line.replace('rw,nosuid', 'ro,nosuid')
    if fault == 'duplicate': line += '\n' + line
    if fault == 'missing': line = ''
    with pytest.raises(ValueError): module.verify_mount(line, True)


def test_parse_only_zone_fields_and_reject_missing_or_duplicate():
    text = 'Node 0, zone   Normal\n        boost    10\n        min      110\n        low      120\n        high     130\n        managed  1000\n              high: 99\n'
    assert module.zones(text)['0:Normal']['high'] == 130
    with pytest.raises(ValueError): module.zones(text.replace('        low      120\n', ''))
    with pytest.raises(ValueError): module.zones(text + '        low      120\n')


def test_readonly_preflight_requires_readonly_exact_mount():
    line = '6495 5207 0:26 /sys/vm/min_free_kbytes /host-min-free-kbytes ro,nosuid,nodev,noexec - proc proc rw'
    module.verify_mount(line, False)
    with pytest.raises(ValueError): module.verify_mount(line.replace('ro,nosuid', 'rw,nosuid'), False)
