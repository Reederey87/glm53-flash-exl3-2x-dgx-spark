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
        self._glm53_role_tokens = (
            None if self.model_config.is_encoder_decoder else make_cache(tokenizer)
        )
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
SALT_SYNC_OLD = "        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)\n"
SALT_ASYNC_OLD = "        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)\n"
SALT_PREFIX = '''        # [glm53-role-token-cache] request salt
        if self._glm53_role_tokens is not None:
            from role_token_cache import with_request_salt
            dict_prompts = with_request_salt(dict_prompts, prompt_extras)
'''
SALT_SYNC_NEW = SALT_PREFIX + SALT_SYNC_OLD
SALT_ASYNC_NEW = SALT_PREFIX + SALT_ASYNC_OLD
PAIRS = ((INIT_OLD, INIT_NEW, 1), (ENCODE_OLD, ENCODE_NEW, 1),
         (SALT_SYNC_OLD, SALT_SYNC_NEW, 2), (SALT_ASYNC_OLD, SALT_ASYNC_NEW, 2))


def install(path: Path, enabled: str) -> bool:
    if enabled not in ("0", "1"):
        raise ValueError(f"{ENV} must be exactly 0 or 1")
    if enabled == "0":
        return False
    source = path.read_text()
    installed = [source.count(new) == count for _, new, count in PAIRS]
    if all(installed):
        if source.count("[glm53-role-token-cache]") != 6:
            raise ValueError("unexpected token-cache markers")
        return False
    if any(installed) or "[glm53-role-token-cache]" in source:
        raise ValueError("partial token-cache installation")
    for old, _, count in PAIRS:
        if source.count(old) != count:
            raise ValueError("renderer anchor drift")
    for old, new, _ in PAIRS:
        source = source.replace(old, new)
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
