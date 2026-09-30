#!/usr/bin/env python3
"""Add only token-cache wiring to a drifted runtime launcher, atomically."""
from __future__ import annotations

import argparse
import os
import stat
import tempfile
from pathlib import Path

EDITS = (
    ('_glm53_cli_sparsemiss_val="${GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC-}"\n',
     '_glm53_cli_rolecache_set="${GLM53_ROLE_TOKEN_CACHE+a}"\n'
     '_glm53_cli_rolecache_val="${GLM53_ROLE_TOKEN_CACHE-}"\n'),
    ('[ -n "${_glm53_cli_sparsemiss_set}" ] && GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC="$_glm53_cli_sparsemiss_val"\n',
     '[ -n "${_glm53_cli_rolecache_set}" ] && GLM53_ROLE_TOKEN_CACHE="$_glm53_cli_rolecache_val"\n'),
    ('PREFIX_CACHE_SPARSE_MISS_PATCH_HOST="${PREFIX_CACHE_SPARSE_MISS_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_prefix_cache_sparse_miss_metric.py}"\n',
     'ROLE_TOKEN_CACHE_PATCH_HOST="${ROLE_TOKEN_CACHE_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_role_token_cache.py}"\n'
     'ROLE_TOKEN_CACHE_MODULE_HOST="${ROLE_TOKEN_CACHE_MODULE_HOST:-$SCRIPT_DIR/overlay/role_token_cache.py}"\n'),
    ('GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC="${GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC-1}"\n',
     'GLM53_ROLE_TOKEN_CACHE="${GLM53_ROLE_TOKEN_CACHE-0}"\n'),
    ('    [ -f "$PREFIX_CACHE_SPARSE_MISS_PATCH_HOST" ] || die "$PREFIX_CACHE_SPARSE_MISS_PATCH_HOST missing"\n',
     '    [ -f "$ROLE_TOKEN_CACHE_PATCH_HOST" ] || die "$ROLE_TOKEN_CACHE_PATCH_HOST missing"\n'
     '    [ -f "$ROLE_TOKEN_CACHE_MODULE_HOST" ] || die "$ROLE_TOKEN_CACHE_MODULE_HOST missing"\n'),
    ('if [ -f /opt/glm53/patch_prefix_cache_sparse_miss_metric.py ]; then\n'
     '    python3 -S /opt/glm53/patch_prefix_cache_sparse_miss_metric.py\nfi\n',
     'if [ -f /opt/glm53/patch_role_token_cache.py ]; then\n'
     '    python3 -S /opt/glm53/patch_role_token_cache.py\nfi\n'),
    ('    scp -q -o BatchMode=yes "$PREFIX_CACHE_SPARSE_MISS_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_prefix_cache_sparse_miss_metric.py"\n',
     '    [ -f "$ROLE_TOKEN_CACHE_PATCH_HOST" ] || die "missing $ROLE_TOKEN_CACHE_PATCH_HOST"\n'
     '    scp -q -o BatchMode=yes "$ROLE_TOKEN_CACHE_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_role_token_cache.py"\n'
     '    [ -f "$ROLE_TOKEN_CACHE_MODULE_HOST" ] || die "missing $ROLE_TOKEN_CACHE_MODULE_HOST"\n'
     '    scp -q -o BatchMode=yes "$ROLE_TOKEN_CACHE_MODULE_HOST" "${WORKER_SSH}:/tmp/role_token_cache.py"\n'),
    ('        -e "GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC=$GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC"\n',
     '        -e "GLM53_ROLE_TOKEN_CACHE=$GLM53_ROLE_TOKEN_CACHE"\n'),
    ("        -v '/tmp/patch_prefix_cache_sparse_miss_metric.py:/opt/glm53/patch_prefix_cache_sparse_miss_metric.py:ro' \\\n",
     "        -v '/tmp/patch_role_token_cache.py:/opt/glm53/patch_role_token_cache.py:ro' \\\n"
     "        -v '/tmp/role_token_cache.py:/opt/glm53/role_token_cache.py:ro' \\\n"),
    ('        -v "$PREFIX_CACHE_SPARSE_MISS_PATCH_HOST:/opt/glm53/patch_prefix_cache_sparse_miss_metric.py:ro" \\\n',
     '        -v "$ROLE_TOKEN_CACHE_PATCH_HOST:/opt/glm53/patch_role_token_cache.py:ro" \\\n'
     '        -v "$ROLE_TOKEN_CACHE_MODULE_HOST:/opt/glm53/role_token_cache.py:ro" \\\n'),
)


def render(source: str) -> str:
    expected = [1] * len(EDITS)
    expected[5] = 2
    installed = [source.count(anchor + extra) == count
                 for (anchor, extra), count in zip(EDITS, expected)]
    loop = "for _v in GLM53_KV_CAPACITY_LOG GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC "
    new_loop = loop + "GLM53_ROLE_TOKEN_CACHE "
    if all(installed) and source.count(new_loop) == 1:
        return source
    if any(installed) or "GLM53_ROLE_TOKEN_CACHE" in source:
        raise ValueError("partial token-cache launcher wiring")
    for (anchor, _), count in zip(EDITS, expected):
        if source.count(anchor) != count:
            raise ValueError(f"launcher anchor drift: {anchor[:70]!r}")
    if source.count(loop) != 1:
        raise ValueError("strict-bool loop drift")
    for anchor, extra in EDITS:
        source = source.replace(anchor, anchor + extra)
    return source.replace(loop, new_loop, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("launcher", type=Path)
    args = parser.parse_args()
    original = args.launcher.read_text()
    updated = render(original)
    if updated == original:
        return
    fd, temporary = tempfile.mkstemp(dir=args.launcher.parent)
    try:
        with os.fdopen(fd, "w") as output:
            output.write(updated)
        os.chmod(temporary, stat.S_IMODE(args.launcher.stat().st_mode))
        os.replace(temporary, args.launcher)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    main()
