#!/usr/bin/env python3
"""Exercise the actual installed renderer entrypoints without model/GPU imports."""
from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

METHODS = {"render_cmpl", "render_cmpl_async", "render_chat", "render_chat_async",
           "_tokenize_prompt", "_build_tokens_prompt", "_apply_prompt_extras"}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def renderer_class(source):
    """Compile only actual CPU entrypoints; replace model/engine services."""
    tree = ast.parse(source)
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "BaseRenderer")
    methods = [copy.deepcopy(node) for node in base.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in METHODS]
    if {node.name for node in methods} != METHODS:
        raise ValueError("actual renderer entrypoint contract incomplete")
    cls = ast.ClassDef(name="Renderer", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
                         names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    namespace = {"asyncio": asyncio, "time": time, "DictPrompt": dict, "TokensPrompt": dict,
                 "extract_target_prompt": lambda config, prompt: prompt}
    exec(compile(ast.fix_missing_locations(module), "<actual-renderer-entrypoints>", "exec"), namespace)
    return namespace["Renderer"]


def exercise(source, tokenizer_factory, cache_module):
    renderer_type = renderer_class(source)
    checks = 0
    for name in ("render_cmpl", "render_cmpl_async", "render_chat", "render_chat_async"):
        tokenizer = tokenizer_factory()
        renderer = renderer_type()
        renderer.model_config = type("Model", (), {"is_encoder_decoder": False})()
        renderer.tokenizer = tokenizer
        renderer._glm53_role_tokens = cache_module.SegmentCache(tokenizer)
        assert renderer._glm53_role_tokens.pattern is not None
        params = type("Params", (), {"get_encode_kwargs": lambda self:
                      {"add_special_tokens": False, "truncation": True, "max_length": 1000000}})()
        renderer.get_tokenizer = lambda: tokenizer
        renderer._wants_offsets = lambda prompt, params: False
        renderer.render_prompts = lambda prompts: [dict(p) for p in prompts]
        renderer.render_messages = lambda conversation, params: (conversation, dict(conversation[0]))
        renderer.tokenize_prompts = lambda prompts, params: [renderer._tokenize_prompt(p, params) for p in prompts]
        renderer.process_for_engine = lambda prompt, arrival_time, **kwargs: prompt

        async def render_prompts_async(prompts):
            return renderer.render_prompts(prompts)

        async def render_messages_async(conversation, params):
            return renderer.render_messages(conversation, params)

        async def tokenize_prompts_async(prompts, params):
            loop = asyncio.get_running_loop()
            return await asyncio.gather(*(loop.run_in_executor(
                None, renderer._tokenize_prompt, p, params) for p in prompts))

        async def process_for_engine_async(prompt, arrival_time, **kwargs):
            return renderer.process_for_engine(prompt, arrival_time, **kwargs)

        renderer.render_prompts_async = render_prompts_async
        renderer.render_messages_async = render_messages_async
        renderer.tokenize_prompts_async = tokenize_prompts_async
        renderer.process_for_engine_async = process_for_engine_async
        raw = {"prompt": "<|system|>" + "Hello world\n" * 150 + "<|user|>turn<|assistant|>"}
        stock = tokenizer(raw["prompt"], **params.get_encode_kwargs())["input_ids"]
        for i, salt in enumerate(("tenant-A", "tenant-B", None, "tenant-A", "tenant-B", None)):
            extras = {"trace": "after-tokenization"}
            if salt is not None:
                extras["cache_salt"] = salt
            args = ([[raw]], None, params) if "chat" in name else ([raw], params)
            output = getattr(renderer, name)(*args, prompt_extras=extras)
            if name.endswith("_async"):
                output = asyncio.run(output)
            prompts = output[1] if "chat" in name else output
            assert prompts[0]["prompt_token_ids"] == stock
            assert prompts[0].get("cache_salt") == salt
            assert prompts[0]["trace"] == "after-tokenization"
            assert "cache_salt" not in raw and "trace" not in raw
            assert renderer._glm53_role_tokens.hits == max(0, i - 2), (name, salt)
            checks += 1
        assert {key[0] for key in renderer._glm53_role_tokens.entries} == {"tenant-A", "tenant-B", None}
    return checks


def main():
    parser = argparse.ArgumentParser()
    for arg in ("renderer", "tokenizer", "module", "installer", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    from transformers import PreTrainedTokenizerFast
    cache = load("role_token_cache", args.module)
    sys.modules["role_token_cache"] = cache
    installer = load("role_token_installer", args.installer)
    with tempfile.TemporaryDirectory() as folder:
        renderer = Path(folder) / "base.py"
        renderer.write_bytes(args.renderer.read_bytes())
        installer.install(renderer, "1")
        checks = exercise(renderer.read_text(), lambda: PreTrainedTokenizerFast(
                          tokenizer_file=str(args.tokenizer)), cache)
    report = {"passed": True, "entrypoints": 4, "salt_boundary_checks": checks}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
