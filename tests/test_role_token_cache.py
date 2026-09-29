from __future__ import annotations

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, processors, trainers

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    spec = importlib.util.spec_from_file_location(relative.replace("/", "_"), ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cache_module = load("overlay/role_token_cache.py")
patch = load("overlay/patch_role_token_cache.py")
wiring = load("scripts/wire_role_token_cache.py")
api_probe = load("scripts/probe_role_token_api.py")


class Adapter:
    is_fast = True
    split_special_tokens = False

    def __init__(self):
        self.backend_tokenizer = Tokenizer(models.BPE())
        self.backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.backend_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokens = [AddedToken(t, normalized=False, special=True) for t in
                  ("<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<tool_call>")]
        self.backend_tokenizer.train_from_iterator(
            ["Hello world\n" * 40, "中文 café 👩‍💻 é العربية ```python a = 1```"],
            trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                special_tokens=tokens),
        )

    def __call__(self, text, **kwargs):
        result = self.backend_tokenizer.encode(text, add_special_tokens=kwargs.get("add_special_tokens", True))
        ids = result.ids
        if kwargs.get("truncation") and kwargs.get("max_length"):
            ids = ids[:kwargs["max_length"]]
        out = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            out["offset_mapping"] = result.offsets
        return out


@pytest.fixture
def tok():
    return Adapter()


def prompt(tail="new turn"):
    return "<|system|>" + "Hello world\n" * 150 + "<|user|>" + tail + "<|assistant|>"


KWARGS = {"add_special_tokens": False, "truncation": True, "max_length": 100000}


@pytest.mark.parametrize("tail", [
    "中文 café 👩‍💻 é العربية", " \n \t", "<|user", "<|user|><|assistant|>",
    "<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value>",
    "```python\nprint('hello')\n```", "</think><|observation|>done",
])
def test_forked_segments_are_token_exact_and_reused(tok, tail):
    cache = cache_module.SegmentCache(tok)
    assert cache.pattern is not None
    assert cache.encode(prompt(), KWARGS) == tok(prompt(), **KWARGS)["input_ids"]
    text = prompt(tail)
    assert cache.encode(text, KWARGS) == tok(text, **KWARGS)["input_ids"]
    assert cache.hits > 0


def test_returned_ids_do_not_mutate_stored_tokens(tok):
    cache = cache_module.SegmentCache(tok)
    ids = cache.encode(prompt(), KWARGS)
    ids[:] = [999]
    assert cache.encode(prompt(), KWARGS) == tok(prompt(), **KWARGS)["input_ids"]


def test_salts_isolate_reuse(tok):
    cache = cache_module.SegmentCache(tok)
    cache.encode(prompt(), KWARGS, "A")
    cache.encode(prompt(), KWARGS, "B")
    assert cache.hits == 0
    cache.encode(prompt("fork"), KWARGS, "A")
    assert cache.hits == 1


@pytest.mark.parametrize("kwargs", [
    {"add_special_tokens": True}, {**KWARGS, "return_offsets_mapping": True},
    {**KWARGS, "padding": True}, {**KWARGS, "max_length": 0},
])
def test_unsupported_options_fall_back(tok, kwargs):
    cache = cache_module.SegmentCache(tok)
    assert cache.encode(prompt(), kwargs) is None
    assert cache.tokenize(prompt(), kwargs) == tok(prompt(), **kwargs)
    assert not cache.entries


def test_overflow_falls_back_and_does_not_store(tok):
    cache = cache_module.SegmentCache(tok)
    kwargs = {**KWARGS, "max_length": 4}
    assert cache.encode(prompt(), kwargs) is None
    assert not cache.entries
    assert cache.tokenize(prompt(), kwargs) == tok(prompt(), **kwargs)


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(normalizer={"type": "Lowercase"}),
    lambda d: d["model"].update(dropout=0.1),
    lambda d: d["added_tokens"][0].update(rstrip=True),
    lambda d: d["added_tokens"][0].update(single_word=True),
    lambda d: d["pre_tokenizer"].update(add_prefix_space=True),
    lambda d: d.update(post_processor={"type": "TemplateProcessing"}),
])
def test_unqualified_pipelines_are_not_cached(tok, mutation):
    config = json.loads(tok.backend_tokenizer.to_str())
    mutation(config)
    tok.backend_tokenizer = type("Backend", (), {"to_str": lambda self: json.dumps(config)})()
    assert cache_module.SegmentCache(tok).pattern is None


def test_budget_and_entry_cap_evict_without_changing_results(tok):
    cache = cache_module.SegmentCache(tok, max_bytes=30000, max_entries=2)
    for i in range(8):
        text = "<|system|>" + ("Hello world " + str(i)) * 100 + "<|user|>new"
        assert cache.encode(text, KWARGS) == tok(text, **KWARGS)["input_ids"]
        assert cache.bytes <= cache.max_bytes
        assert len(cache.entries) <= 2


def test_concurrent_requests_keep_exact_ids(tok):
    cache = cache_module.SegmentCache(tok)
    cache.encode(prompt(), KWARGS)
    texts = [prompt(f"fork {i} 中文") for i in range(24)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(lambda text: cache.tokenize(text, KWARGS), texts))
    assert outputs == [tok(text, **KWARGS) for text in texts]
    assert cache.hits == 24


@pytest.mark.parametrize("value", ["", " 1", "1 ", "01", "true"])
def test_bad_flags_are_rejected_before_source_read(tmp_path, value):
    with pytest.raises(ValueError):
        patch.install(tmp_path / "missing", value)


def test_installer_off_is_byte_inert_and_on_is_atomic_idempotent(tmp_path):
    target = tmp_path / "base.py"
    target.write_text("class Renderer:\n    def init(self):\n" + patch.INIT_OLD
                      + "    def encode(self):\n" + patch.ENCODE_OLD + "            [], {}\n        )\n")
    target.chmod(0o640)
    original = target.read_bytes()
    assert not patch.install(target, "0")
    assert target.read_bytes() == original
    assert patch.install(target, "1")
    installed = target.read_bytes()
    assert not patch.install(target, "1")
    assert target.read_bytes() == installed
    assert target.stat().st_mode & 0o777 == 0o640


def test_partial_or_drifted_installer_never_writes(tmp_path):
    target = tmp_path / "base.py"
    for text in ("class X: pass", patch.INIT_NEW + patch.ENCODE_OLD):
        target.write_text(text)
        with pytest.raises(ValueError):
            patch.install(target, "1")
        assert target.read_text() == text


def test_launcher_surgery_matches_repo_and_preserves_unrelated_changes():
    candidate = (ROOT / "start.sh").read_text()
    original = candidate
    for anchor, extra in wiring.EDITS:
        original = original.replace(anchor + extra, anchor)
    original = original.replace(
        "for _v in GLM53_KV_CAPACITY_LOG GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC GLM53_ROLE_TOKEN_CACHE ",
        "for _v in GLM53_KV_CAPACITY_LOG GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC ",
    )
    assert wiring.render(original) == candidate
    assert wiring.render(candidate) == candidate
    drifted = original + "\n# unrelated runtime arm\n"
    assert wiring.render(drifted) == candidate + "\n# unrelated runtime arm\n"
    with pytest.raises(ValueError):
        wiring.render(original.replace(wiring.EDITS[0][0], ""))


def api_receipt():
    import hashlib
    return {name: {"text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "ids_sha256": "a" * 64, "count": 30, "seconds": 1.0}
            for name, text in api_probe.cases()}


def test_api_comparison_requires_complete_finite_exact_evidence():
    baseline = api_receipt()
    assert api_probe.compare(baseline, api_receipt())["exact_cases"] == 30
    candidate = api_receipt()
    candidate.pop(next(iter(candidate)))
    with pytest.raises(ValueError):
        api_probe.compare(baseline, candidate)
    for field, value in (("seconds", float("nan")), ("ids_sha256", "b" * 64),
                         ("count", True), ("text_sha256", "drift")):
        candidate = api_receipt()
        candidate[next(iter(candidate))][field] = value
        with pytest.raises(ValueError):
            api_probe.compare(baseline, candidate)
