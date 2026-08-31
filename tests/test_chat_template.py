#!/usr/bin/env python3
"""Regression checks for GLM-5.3 chat-template reasoning controls."""

import unittest
from pathlib import Path

from jinja2 import Environment


TEMPLATE = Path(__file__).parents[1] / "files" / "chat_template.jinja"


def render_generation_prompt(**kwargs: object) -> str:
    template = Environment(extensions=["jinja2.ext.loopcontrols"]).from_string(
        TEMPLATE.read_text()
    )
    return template.render(
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        add_generation_prompt=True,
        **kwargs,
    )


class ChatTemplateTests(unittest.TestCase):
    def test_thinking_defaults_on(self) -> None:
        rendered = render_generation_prompt()
        self.assertIn("<|system|>Reasoning Effort: Max", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think>"), rendered)

    def test_thinking_can_be_disabled(self) -> None:
        # LOCAL (W16): the effort line is emitted unconditionally so the
        # thinking toggle never invalidates the prefix cache at token ~2.
        rendered = render_generation_prompt(enable_thinking=False)
        self.assertIn("<|system|>Reasoning Effort: Max", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think></think>"), rendered)

    def test_thinking_alias_matches_parser_behavior(self) -> None:
        rendered = render_generation_prompt(thinking=False)
        self.assertIn("<|system|>Reasoning Effort: Max", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think></think>"), rendered)

    def test_thinking_off_is_strict_extension_of_on(self) -> None:
        # Prefix stability: the off-shape must equal the on-shape plus the
        # closed think block, so the two share the whole cached prefix.
        on = render_generation_prompt(enable_thinking=True)
        off = render_generation_prompt(enable_thinking=False)
        self.assertTrue(on.endswith("<|assistant|><think>"), on)
        self.assertEqual(off, on + "</think>")

    def test_explicit_thinking_preserves_reasoning_effort(self) -> None:
        rendered = render_generation_prompt(
            enable_thinking=True,
            reasoning_effort="low",
        )
        self.assertIn("<|system|>Reasoning Effort: Low", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think>"), rendered)


if __name__ == "__main__":
    unittest.main()
