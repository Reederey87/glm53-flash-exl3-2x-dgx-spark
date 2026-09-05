#!/usr/bin/env python3
"""Golden regression checks for the production GLM-5.3 chat template."""

import json
import unittest
from pathlib import Path

from jinja2 import Environment


TEMPLATE = Path(__file__).parents[1] / "files" / "chat_template.jinja"


def compile_template(source: str | None = None):
    environment = Environment(extensions=["jinja2.ext.loopcontrols"])
    environment.filters["tojson"] = lambda value, ensure_ascii=False: json.dumps(
        value, ensure_ascii=ensure_ascii
    )
    return environment.from_string(source if source is not None else TEMPLATE.read_text())


def render(
    messages: list[dict[str, object]],
    *,
    tools: list[dict[str, object]] | None = None,
    add_generation_prompt: bool = False,
    template=None,
    **kwargs: object,
) -> str:
    template = template or compile_template()
    return template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def render_generation_prompt(**kwargs: object) -> str:
    return render(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
        **kwargs,
    )


def tool_call(call_id: str, name: object = "lookup") -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": {"query": call_id}},
    }


def tool_result(call_id: str | None, content: object) -> dict[str, object]:
    result: dict[str, object] = {"role": "tool", "content": content}
    if call_id is not None:
        result["tool_call_id"] = call_id
    return result


def list_output(call_id: str | None, output: object) -> dict[str, object]:
    result: dict[str, object] = {"output": output}
    if call_id is not None:
        result["tool_call_id"] = call_id
    return result


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


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

    def test_valid_shuffled_tool_results_are_sorted_by_call_order(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("call-0"), tool_call("call-1")],
            },
            tool_result("call-1", "second"),
            tool_result("call-0", "first"),
        ]
        rendered = render(messages)
        self.assertIn(
            "<|observation|><tool_response>first</tool_response>"
            "<tool_response>second</tool_response>",
            rendered,
        )

    def test_missing_id_falls_back_to_input_order(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("call-0"), tool_call("call-1")],
            },
            tool_result(None, "missing"),
            tool_result("call-0", "known"),
        ]
        rendered = render(messages)
        self.assertIn(
            "<|observation|><tool_response>missing</tool_response>"
            "<tool_response>known</tool_response>",
            rendered,
        )

    def test_duplicate_result_id_falls_back_to_input_order(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("call-0"), tool_call("call-1")],
            },
            tool_result("call-0", "first"),
            tool_result("call-0", "duplicate"),
        ]
        rendered = render(messages)
        self.assertIn(
            "<|observation|><tool_response>first</tool_response>"
            "<tool_response>duplicate</tool_response>",
            rendered,
        )

    def test_unknown_result_id_falls_back_to_input_order(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("call-0")],
            },
            tool_result("unknown", "unknown-result"),
            tool_result("call-0", "known-result"),
        ]
        rendered = render(messages)
        self.assertIn(
            "<|observation|><tool_response>unknown-result</tool_response>"
            "<tool_response>known-result</tool_response>",
            rendered,
        )

    def test_duplicate_tool_call_id_falls_back_to_input_order(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("call-0"), tool_call("call-0")],
            },
            tool_result("call-0", "result"),
        ]
        rendered = render(messages)
        self.assertIn(
            "<|observation|><tool_response>result</tool_response>",
            rendered,
        )

    def test_list_outputs_are_sorted_by_call_order(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("call-0"), tool_call("call-1")],
            },
            {
                "role": "tool",
                "content": [
                    list_output("call-1", "second"),
                    list_output("call-0", "first"),
                ],
            },
        ]
        rendered = render(messages)
        self.assertIn(
            "<|observation|><tool_response>first</tool_response>"
            "<tool_response>second</tool_response>",
            rendered,
        )

    def test_tool_reference_outputs_preserve_tool_definition(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("call-0")],
            },
            {
                "role": "tool",
                "tool_call_id": "call-0",
                "content": [{"type": "tool_reference", "name": "lookup"}],
            },
        ]
        rendered = render(messages, tools=TOOLS)
        self.assertIn("<tool_response><tools>\n", rendered)
        self.assertIn('"name": "lookup"', rendered)
        self.assertIn("</tools></tool_response>", rendered)

    def test_null_content_is_suppressed(self) -> None:
        rendered = render(
            [
                {"role": "system", "content": None},
                {"role": "user", "content": None},
                {"role": "assistant", "content": None},
            ]
        )
        self.assertNotIn("None", rendered)
        self.assertEqual(
            rendered,
            "[gMASK]<sop><|system|>Reasoning Effort: Max"
            "<|system|><|user|><|assistant|><think></think>",
        )

    def test_non_string_tool_name_is_coerced(self) -> None:
        rendered = render(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call("call-0", name=7)],
                }
            ]
        )
        self.assertIn("<tool_call>7<arg_key>query</arg_key>", rendered)

    def test_multimedia_placeholders_are_preserved(self) -> None:
        rendered = render(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect:"},
                        {"type": "image_url", "image_url": {"url": "image"}},
                        {"type": "video_url", "video_url": {"url": "video"}},
                        {"type": "input_audio", "input_audio": {"data": "audio"}},
                    ],
                }
            ]
        )
        self.assertIn(
            "inspect:<|begin_of_image|><|image|><|end_of_image|>"
            "<|begin_of_video|><|video|><|end_of_video|>"
            "<|begin_of_audio|><|end_of_audio|>",
            rendered,
        )

    def test_fallback_handles_1_10_100_and_500_tool_results(self) -> None:
        template = compile_template()
        for count in (1, 10, 100, 500):
            with self.subTest(count=count):
                messages = [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            tool_call(f"call-{index}") for index in range(count)
                        ],
                    },
                    tool_result(None, "missing-id"),
                    *[
                        tool_result(f"call-{index}", f"result-{index}")
                        for index in range(1, count)
                    ],
                ]
                rendered = render(messages, template=template)
                self.assertIn(
                    "<|observation|><tool_response>missing-id</tool_response>",
                    rendered,
                )
                if count > 1:
                    self.assertIn(
                        f"<tool_response>result-{count - 1}</tool_response>",
                        rendered,
                    )


if __name__ == "__main__":
    unittest.main()
