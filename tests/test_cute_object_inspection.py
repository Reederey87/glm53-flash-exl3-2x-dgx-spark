from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest


@pytest.fixture
def inspector():
    path = Path(__file__).resolve().parents[1] / "experiments/dense-int64/inspect_cute_object.py"
    spec = importlib.util.spec_from_file_location("cute_object_inspector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def object_bytes(*, symbol_value=16, symbol_size=None, section_type=1, duplicate=False):
    blob = struct.pack("<IHHQ", 0xba55ed50, 1, 16, 8) + b"device!!"
    # The decoy magic before the named symbol must never be scanned/extracted.
    rodata = b"\x50\xed\x55\xba" + b"x" * 12 + blob + b"trailing-data"
    assert rodata[16:40] == blob
    names = b"\0kernels_binary\0"
    symbols = bytes(24) + struct.pack("<IBBHQQ", 1, 1, 0, 1,
                                     symbol_value, len(blob) if symbol_size is None else symbol_size)
    if duplicate:
        symbols += symbols[-24:]
    ro_offset = 64
    names_offset = ro_offset + len(rodata)
    symbols_offset = names_offset + len(names)
    table_offset = symbols_offset + len(symbols)
    header = struct.pack("<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01\x01" + bytes(9),
                         1, 183, 1, 0, 0, table_offset, 0, 64, 0, 0, 64, 4, 0)
    sections = bytes(64)
    sections += struct.pack("<IIQQQQIIQQ", 0, section_type, 2, 0, ro_offset, len(rodata), 0, 0, 8, 0)
    sections += struct.pack("<IIQQQQIIQQ", 0, 3, 0, 0, names_offset, len(names), 0, 0, 1, 0)
    sections += struct.pack("<IIQQQQIIQQ", 0, 2, 0, 0, symbols_offset, len(symbols), 2, 0, 8, 24)
    return header + rodata + names + symbols + sections, blob, ro_offset + 16


def test_extracts_only_named_symbol_not_decoy_or_whole_section(inspector):
    data, blob, offset = object_bytes()
    assert inspector.extract_fatbin(data) == (blob, offset)


@pytest.mark.parametrize("fault", ["short_header", "short_sections", "wrong_host", "outside_section",
                                  "duplicate", "nobits", "fatbin_length", "wrong_symbol_offset"])
def test_rejects_ambiguous_truncated_and_out_of_bounds_objects(inspector, fault):
    kwargs = {"outside_section": {"symbol_size": 500}, "duplicate": {"duplicate": True},
              "nobits": {"section_type": 8}, "wrong_symbol_offset": {"symbol_value": 0}}
    data, _, offset = object_bytes(**kwargs.get(fault, {}))
    if fault == "short_header":
        data = data[:30]
    elif fault == "short_sections":
        data = data[:-1]
    elif fault == "wrong_host":
        data = data[:18] + struct.pack("<H", 62) + data[20:]
    elif fault == "fatbin_length":
        data = data[:offset + 8] + struct.pack("<Q", 999) + data[offset + 16:]
    with pytest.raises(ValueError):
        inspector.extract_fatbin(data)


@pytest.fixture
def cached_object(inspector, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    source = cache / "fixture.o"
    data, _, _ = object_bytes()
    source.write_bytes(data)
    semantic = {"target": "dense", "spec": {"split_k": 16}}
    manifest = {"schema": "b12x._lib.compile_manifest.v3", "object_sha256": inspector.sha(data),
                "object_bytes": len(data), "semantic_payload": semantic,
                "semantic_key": inspector.sha(json.dumps(semantic, sort_keys=True,
                                                          separators=(",", ":")).encode())}
    source.with_suffix(".json").write_text(json.dumps(manifest))
    return source, tmp_path / "inspection"


@pytest.mark.parametrize("fault", ["object_hash", "semantic_hash", "inside_cache"])
def test_identity_errors_create_no_inspection(inspector, cached_object, fault):
    source, output = cached_object
    if fault == "object_hash":
        source.write_bytes(source.read_bytes() + b"changed")
    elif fault == "semantic_hash":
        path = source.with_suffix(".json")
        manifest = json.loads(path.read_text())
        manifest["semantic_payload"]["spec"]["split_k"] = 4
        path.write_text(json.dumps(manifest))
    else:
        output = source.parent / "inspection"
    with pytest.raises(ValueError):
        inspector.inspect_object(source, output)
    assert not output.exists()


@pytest.mark.parametrize("fault", [None, "tool_failure", "missing_sass", "wrong_arch", "source_changed"])
def test_inspection_requires_matching_native_reports_and_unchanged_source(
        inspector, cached_object, monkeypatch, fault):
    source, output = cached_object
    before = source.read_bytes()

    def run(command, **_kwargs):
        option = command[1]
        arch = "sm_120a" if fault == "wrong_arch" else "sm_121a"
        reports = {"--list-elf": f"ELF file 1: kernels.{arch}.cubin\n",
                   "--dump-resource-usage": f"arch = {arch}\n Function dense_kernel:\n  REG:95 STACK:0 LOCAL:0\n",
                   "--dump-sass": "Function : dense_kernel\n"}
        if fault == "source_changed":
            source.write_bytes(before + b"changed")
        return SimpleNamespace(returncode=1 if fault == "tool_failure" else 0,
                               stdout="" if fault == "missing_sass" and option == "--dump-sass" else reports[option])

    monkeypatch.setattr(inspector.subprocess, "run", run)
    report = inspector.inspect_object(source, output)
    assert report["inspection_complete"] is (fault is None)
    assert (output / "inspection.json").exists()
    if fault != "source_changed":
        assert source.read_bytes() == before
    assert "GPU correctness" in report["qualification"]
