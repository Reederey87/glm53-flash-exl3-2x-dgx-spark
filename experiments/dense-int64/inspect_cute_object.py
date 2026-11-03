#!/usr/bin/env python3
"""Inspect a CuTe ARM64 cache object without loading it or initializing CUDA.

The exported host ELF keeps a fatbin in the kernels_binary data symbol, which
cuobjdump does not discover directly. Extract that symbol, never a magic-byte
scan or the whole rodata section. Original cache files are read-only inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_fatbin(data: bytes) -> tuple[bytes, int]:
    def span(offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset + size > len(data):
            raise ValueError("ELF range exceeds the object")
        return data[offset:offset + size]

    header = struct.unpack("<16sHHIQQQIHHHHHH", span(0, 64))
    if header[0][:7] != b"\x7fELF\x02\x01\x01" or header[1:4] != (1, 183, 1):
        raise ValueError("requires an ELF64 little-endian ARM64 relocatable object")
    table, entry_size, count = header[6], header[11], header[12]
    if header[8] != 64 or entry_size != 64 or not 0 < count < 0xff00:
        raise ValueError("unsupported ELF header or extended section table")
    span(table, entry_size * count)
    sections = [struct.unpack("<IIQQQQIIQQ", span(table + i * 64, 64))
                for i in range(count)]
    matches = []
    for section in sections:
        if section[1] != 2:  # SHT_SYMTAB
            continue
        if section[9] != 24 or section[5] % 24 or not 0 < section[6] < count:
            raise ValueError("invalid ELF symbol table")
        strings_section = sections[section[6]]
        if strings_section[1] != 3:
            raise ValueError("symbol names must come from SHT_STRTAB")
        strings = span(strings_section[4], strings_section[5])
        symbols = span(section[4], section[5])
        for pos in range(0, len(symbols), 24):
            name, info, _, index, value, size = struct.unpack_from("<IBBHQQ", symbols, pos)
            if name >= len(strings) or strings.find(b"\0", name) == -1:
                raise ValueError("invalid ELF symbol name")
            if strings[name:strings.index(b"\0", name)] != b"kernels_binary":
                continue
            if info & 15 != 1 or not 0 < index < count or size < 16:
                raise ValueError("kernels_binary must be a bounded data object")
            owner = sections[index]
            # ET_REL symbol values are section-relative offsets, not addresses.
            if owner[1] != 1 or owner[2] & 7 != 2 or value + size > owner[5]:
                raise ValueError("kernels_binary exceeds read-only allocated PROGBITS")
            span(owner[4], owner[5])
            blob = span(owner[4] + value, size)
            magic, version, header_bytes, body_bytes = struct.unpack_from("<IHHQ", blob)
            if (magic != 0xba55ed50 or version != 1 or header_bytes != 16
                    or header_bytes + body_bytes != len(blob)):
                raise ValueError("unsupported or truncated CUDA fatbin")
            matches.append((blob, owner[4] + value))
    if len(matches) != 1:
        raise ValueError("expected exactly one kernels_binary symbol")
    return matches[0]


def inspect_object(source: Path, output: Path, cuobjdump: str = "cuobjdump") -> dict:
    manifest_path = source.with_suffix(".json")
    if (source.is_symlink() or manifest_path.is_symlink()
            or not source.is_file() or not manifest_path.is_file()):
        raise ValueError("cache object and manifest must be regular, direct paths")
    if output.resolve().is_relative_to(source.parent.resolve()):
        raise ValueError("inspection output must be outside the cache object's directory")
    data, manifest_bytes = source.read_bytes(), manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema") != "b12x._lib.compile_manifest.v3"
            or manifest.get("object_sha256") != sha(data)
            or type(manifest.get("object_bytes")) is not int
            or manifest["object_bytes"] != len(data)):
        raise ValueError("object does not match its B12X cache manifest")
    semantic = json.dumps(manifest["semantic_payload"], sort_keys=True,
                          separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    if manifest.get("semantic_key") != sha(semantic):
        raise ValueError("semantic key does not match its payload")
    blob, offset = extract_fatbin(data)
    output.mkdir(parents=True, exist_ok=False)
    binary = output / "kernels.fatbin"
    binary.write_bytes(blob)
    receipt = {
        "source_object": str(source), "object_sha256": sha(data),
        "manifest_sha256": sha(manifest_bytes), "semantic_key": manifest["semantic_key"],
        "semantic_payload": manifest["semantic_payload"], "target": manifest.get("target"),
        "package_fingerprint": manifest.get("package_fingerprint"),
        "symbol": "kernels_binary", "object_offset": offset,
        "fatbin_bytes": len(blob), "fatbin_sha256": sha(blob),
        "launch_metadata": manifest.get("launch_metadata"), "tools": {},
        "qualification": "Artifact inspection only; GPU correctness, paired resource review and serving gates remain required.",
    }
    texts = {}
    for name, option in (("elf", "--list-elf"), ("resources", "--dump-resource-usage"),
                         ("sass", "--dump-sass")):
        try:
            result = subprocess.run([cuobjdump, option, str(binary)], text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            text, status = result.stdout, result.returncode
        except (OSError, subprocess.SubprocessError) as error:
            text, status = str(error), None
        (output / f"{name}.txt").write_text(text)
        texts[name] = text
        receipt["tools"][name] = {"exit_code": status, "report": f"{name}.txt",
                                  "sha256": sha(text.encode())}
    resources = set(re.findall(r"^ Function (\S+):", texts["resources"], re.M))
    disassembly = set(re.findall(r"Function\s*:\s*(\S+)", texts["sass"]))
    arches = set(re.findall(r"arch\s*=\s*(\S+)", texts["resources"]))
    unchanged = source.read_bytes() == data and manifest_path.read_bytes() == manifest_bytes
    binary_unchanged = binary.read_bytes() == blob
    receipt.update(architectures=sorted(arches), resource_kernels=sorted(resources),
                   sass_kernels=sorted(disassembly), originals_unchanged=unchanged,
                   extracted_binary_unchanged=binary_unchanged,
                   inspection_complete=bool(resources) and resources == disassembly
                   and arches == {"sm_121a"} and unchanged and binary_unchanged
                   and all(item["exit_code"] == 0 for item in receipt["tools"].values()))
    (output / "inspection.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="new directory outside the source cache")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    args = parser.parse_args()
    report = inspect_object(args.object, args.out, args.cuobjdump)
    print(json.dumps({key: report[key] for key in
                      ("object_sha256", "fatbin_sha256", "architectures", "inspection_complete")}, indent=2))
    raise SystemExit(0 if report["inspection_complete"] else 1)
