#!/usr/bin/env python3
"""Fail-closed, atomic renderer overlay for token-exact segment caching."""
from __future__ import annotations

import ast
import os
import stat
import tempfile
from pathlib import Path

ENV = "GLM53_ROLE_TOKEN_CACHE"
TARGET = Path(os.environ.get(
    "GLM53_RENDERER_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/renderers/base.py",
))
INIT_OLD = "        self.tokenizer = tokenizer\n"
INIT_NEW = INIT_OLD + '''        # [glm53-role-token-cache] init
        import sys as _glm53_sys
        if "/opt/glm53" not in _glm53_sys.path:
            _glm53_sys.path.insert(0, "/opt/glm53")
        from role_token_cache import make_cache
        self._glm53_role_tokens = make_cache(tokenizer)
'''
ENCODE_OLD = '''        encoding = tokenizer(prompt["prompt"], **kwargs)
        return self._build_tokens_prompt(
'''
ENCODE_NEW = '''        # [glm53-role-token-cache] encode
        if self._glm53_role_tokens is not None:
            encoding = self._glm53_role_tokens.tokenize(
                prompt["prompt"], kwargs, prompt.get("cache_salt")
            )
        else:
            encoding = tokenizer(prompt["prompt"], **kwargs)
        return self._build_tokens_prompt(
'''


def install(path: Path, enabled: str) -> bool:
    if enabled not in ("0", "1"):
        raise ValueError(f"{ENV} must be exactly 0 or 1")
    if enabled == "0":
        return False
    source = path.read_text()
    pairs = ((INIT_OLD, INIT_NEW), (ENCODE_OLD, ENCODE_NEW))
    installed = [source.count(new) == 1 for _, new in pairs]
    if all(installed):
        if source.count("[glm53-role-token-cache]") != 2:
            raise ValueError("unexpected token-cache markers")
        return False
    if any(installed) or "[glm53-role-token-cache]" in source:
        raise ValueError("partial token-cache installation")
    for old, _ in pairs:
        if source.count(old) != 1:
            raise ValueError("renderer anchor drift")
    for old, new in pairs:
        source = source.replace(old, new, 1)
    ast.parse(source)
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as output:
            output.write(source)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return True


if __name__ == "__main__":
    changed = install(TARGET, os.environ.get(ENV, "0"))
    print(f"[glm53-role-token-cache] {'applied' if changed else 'unchanged'}", flush=True)
