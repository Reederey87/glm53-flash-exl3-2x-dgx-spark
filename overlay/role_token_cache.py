"""Bounded token-exact reuse of independently encoded added-token segments.

No prefix guessing: every seam is a plain added token extracted before BPE.
Only the qualified GLM fast-tokenizer pipeline is admitted. Other tokenizers,
offset requests, implicit special tokens and unsupported kwargs use stock.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from collections import OrderedDict

ENV = "GLM53_ROLE_TOKEN_CACHE"
MARK = "[glm53-role-token-cache]"
MAX_BYTES = 32 * 1024 * 1024
MAX_ENTRIES = 128
MIN_CHARS = 1024


def with_request_salt(prompts, extras):
    """Copy only the salt into raw decoder prompts, before any tokenization."""
    if not extras or "cache_salt" not in extras:
        return prompts
    return [{**prompt, "cache_salt": extras["cache_salt"]} for prompt in prompts]


def make_cache(tokenizer):
    raw = os.environ.get(ENV, "0")
    if raw not in ("0", "1"):
        raise RuntimeError(f"{ENV} must be exactly 0 or 1")
    if raw == "0" or tokenizer is None:
        return None
    cache = SegmentCache(tokenizer)
    print(f"{MARK} qualified={int(cache.pattern is not None)} budget={MAX_BYTES}", flush=True)
    return cache


class SegmentCache:
    def __init__(self, tokenizer, *, max_bytes=MAX_BYTES, max_entries=MAX_ENTRIES):
        self.tokenizer = tokenizer
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self._announced_hit = False
        self.lock = threading.RLock()
        self.pattern = self._qualify(tokenizer)

    def tokenize(self, text, kwargs, salt=None):
        """Serialize cache and stock calls on this renderer's HF tokenizer."""
        with self.lock:
            ids = self.encode(text, kwargs, salt)
            if ids is not None:
                return {"input_ids": ids}
            return self.tokenizer(text, **kwargs)

    @staticmethod
    def _qualify(tokenizer):
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if not getattr(tokenizer, "is_fast", False) or backend is None:
            return None
        try:
            config = json.loads(backend.to_str())
            model = config["model"]
            if (config["normalizer"] is not None or model["type"] != "BPE"
                    or model.get("dropout") is not None):
                return None
            # This exact family of stateless per-split operations is qualified.
            pre = config["pre_tokenizer"]
            parts = pre.get("pretokenizers", []) if pre["type"] == "Sequence" else [pre]
            if not parts or any(
                p["type"] not in ("Split", "ByteLevel")
                or (p["type"] == "ByteLevel" and p.get("add_prefix_space") is not False)
                for p in parts
            ):
                return None
            post = config["post_processor"]
            if post is not None and post["type"] != "ByteLevel":
                return None
            added = config["added_tokens"]
            if not added or any(
                not t["content"] or any(t[f] for f in
                    ("single_word", "lstrip", "rstrip", "normalized"))
                for t in added
            ):
                return None
            # Same leftmost-longest selection as tokenizers' added vocabulary.
            return re.compile("|".join(re.escape(t["content"]) for t in
                              sorted(added, key=lambda t: -len(t["content"]))))
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    def encode(self, text, kwargs, salt=None):
        """Return ids, or None to take the unmodified full-encoding path."""
        if (self.pattern is None or not isinstance(text, str)
                or len(text) < MIN_CHARS
                or kwargs.get("add_special_tokens") is not False
                or set(kwargs) - {"add_special_tokens", "truncation", "max_length"}
                or kwargs.get("truncation", False) not in (False, True)
                or getattr(self.tokenizer, "split_special_tokens", False)
                or getattr(self.tokenizer.backend_tokenizer, "encode_special_tokens", False)
                or (salt is not None and not isinstance(salt, str))):
            return None
        limit = kwargs.get("max_length")
        if limit is not None and (type(limit) is not int or limit < 1):
            return None
        # Hold the lock across encoding: HF fast tokenizers temporarily mutate
        # backend truncation options. Shared requests must not race those options.
        with self.lock:
            matches = list(self.pattern.finditer(text))
            if not matches:
                return None
            ids = []
            pending = []
            start = 0
            for match in matches:
                chunk = text[start:match.start()]
                if chunk:
                    key = (salt, chunk)
                    cached = self.entries.get(key)
                    if cached is None:
                        self.misses += 1
                        tokens = tuple(self.tokenizer(
                            chunk, add_special_tokens=False, truncation=False
                        )["input_ids"])
                        if len(chunk) >= MIN_CHARS:
                            pending.append((key, tokens))
                    else:
                        self.hits += 1
                        self.entries.move_to_end(key)
                        tokens = cached[0]
                    ids.extend(tokens)
                # The added token itself is an atomic separator, never cached.
                ids.extend(self.tokenizer(
                    match.group(), add_special_tokens=False, truncation=False
                )["input_ids"])
                start = match.end()
            if start < len(text):
                ids.extend(self.tokenizer(
                    text[start:], add_special_tokens=False, truncation=False
                )["input_ids"])
            # Overflow must use stock truncation-side/postprocessing semantics.
            # Never retain fragments from an over-limit request.
            if kwargs.get("truncation") and limit is not None and len(ids) > limit:
                return None
            for key, tokens in pending:
                size = (sys.getsizeof(key) + sys.getsizeof(key[1])
                        + sys.getsizeof(key[0]) + sys.getsizeof(tokens)
                        + sum(sys.getsizeof(t) for t in tokens) + 256)
                if size > self.max_bytes:
                    continue
                previous = self.entries.pop(key, None)
                if previous:
                    self.bytes -= previous[1]
                self.entries[key] = (tokens, size)
                self.bytes += size
                while self.bytes > self.max_bytes or len(self.entries) > self.max_entries:
                    _, (_, removed) = self.entries.popitem(last=False)
                    self.bytes -= removed
            if self.hits and not self._announced_hit:
                self._announced_hit = True
                print(f"{MARK} segment reuse active", flush=True)
            return ids
