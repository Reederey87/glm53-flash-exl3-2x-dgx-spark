import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def cache_helper(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / 'experiments/dense-int64'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('closed_cache', folder / 'release_closed_native_cache.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('fault', [None, 'writable', 'wrong_source', 'nested', 'duplicate'])
def test_exact_readonly_result_mount(cache_helper, fault):
    m = cache_helper
    line = f'2 1 0:2 {m.SOURCE} {m.ROOT} ro - ext4 disk rw'
    if fault == 'writable': line = line.replace(' ro -', ' rw -')
    if fault == 'wrong_source': line = line.replace(m.SOURCE, m.SOURCE + '-other')
    if fault == 'nested': line += f'\n3 2 0:2 /other {m.ROOT}/child ro - ext4 disk rw'
    if fault == 'duplicate': line += '\n' + line
    if fault:
        with pytest.raises(ValueError): m.verify_mount(line)
    else: m.verify_mount(line)


@pytest.mark.parametrize('fault', [None, 'last_hash', 'linked_cache'])
def test_all_receipts_checked_before_any_advice(cache_helper, tmp_path, monkeypatch, fault):
    m = cache_helper
    summaries = {}
    for name in ('first', 'last'):
        folder = tmp_path / name; folder.mkdir(); (folder / 'cache').mkdir()
        raw = name.encode(); (folder / 'native-summary.json').write_bytes(raw)
        summaries[name] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(m, 'SUMMARIES', summaries)
    calls = []
    monkeypatch.setattr(m, 'release_tree', lambda path: calls.append(path) or {'metadata_unchanged': True})
    monkeypatch.setattr(m, 'meminfo', lambda: {})
    if fault == 'last_hash': (tmp_path / 'last/native-summary.json').write_text('changed')
    if fault == 'linked_cache':
        (tmp_path / 'last/cache').rmdir()
        (tmp_path / 'last/cache').symlink_to(tmp_path / 'first/cache')
    if fault:
        with pytest.raises(ValueError): m.release(tmp_path)
        assert not calls
    else:
        result = m.release(tmp_path)
        assert result['receipts_unchanged'] and len(calls) == 2
        assert (tmp_path / 'first/native-summary.json').read_bytes() == b'first'
