# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Frozen CPU entrypoint contract from the deployed vLLM renderer.
# Source SHA256: a358921359b241df2e5c08b547897b874a0eb5c73bc9051293bab800511aeedb
from __future__ import annotations

class BaseRenderer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def _build_tokens_prompt(
        token_ids: Sequence[int],
        prompt: "TextPrompt",
        *,
        offset_mapping: Sequence[tuple[int, int]] | None = None,
    ) -> "TokensPrompt":
        """Build a TokensPrompt from already-extracted token ids.

        ``offset_mapping`` is the per-token ``(start, end)`` sequence from
        a BatchEncoding; pass it only when offsets were requested, and it
        is attached as ``prompt_token_offsets``.
        """
        if offset_mapping is not None:
            return TokensPrompt(
                prompt_token_ids=list(token_ids),
                prompt_token_offsets=[(int(s), int(e)) for s, e in offset_mapping],
                **prompt,
            )
        return TokensPrompt(prompt_token_ids=list(token_ids), **prompt)


    def _tokenize_prompt(
        self,
        prompt: TextPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt:
        tokenizer = self.get_tokenizer()
        want_offsets = self._wants_offsets(prompt, params)
        kwargs = params.get_encode_kwargs()
        if want_offsets:
            kwargs = {**kwargs, "return_offsets_mapping": True}
        encoding = tokenizer(prompt["prompt"], **kwargs)
        return self._build_tokens_prompt(
            encoding["input_ids"],
            prompt,
            offset_mapping=encoding["offset_mapping"] if want_offsets else None,
        )


    def _apply_prompt_extras(
        self,
        prompts: Sequence[TokPrompt],
        prompt_extras: dict[str, Any] | None,
    ):
        if not prompt_extras:
            return

        for prompt in prompts:
            target_prompt = extract_target_prompt(self.model_config, prompt)
            target_prompt.update(prompt_extras)  # type: ignore[arg-type]


    def render_cmpl(
        self,
        prompts: Sequence[DictPrompt | bytes],
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_cmpl_tok_params

        dict_prompts = self.render_prompts(prompts)
        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        return [
            self.process_for_engine(prompt, arrival_time, skip_mm_cache=skip_mm_cache)
            for prompt in tok_prompts
        ]


    async def render_cmpl_async(
        self,
        prompts: Sequence[DictPrompt | bytes],
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_cmpl_tok_params

        dict_prompts = await self.render_prompts_async(prompts)
        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        return await asyncio.gather(
            *(
                self.process_for_engine_async(
                    p, arrival_time, skip_mm_cache=skip_mm_cache
                )
                for p in tok_prompts
            )
        )


    def render_chat(
        self,
        conversations: Sequence[list["ChatCompletionMessageParam"]],
        chat_params: ChatParams,
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_chat_tok_params

        rendered = [
            self.render_messages(conversation, chat_params)
            for conversation in conversations
        ]

        out_conversations = list[list["ConversationMessage"]]()
        dict_prompts = list[DictPrompt]()
        for conv, prompt in rendered:
            out_conversations.append(conv)
            dict_prompts.append(prompt)

        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        eng_prompts = [
            self.process_for_engine(prompt, arrival_time, skip_mm_cache=skip_mm_cache)
            for prompt in tok_prompts
        ]

        return out_conversations, eng_prompts


    async def render_chat_async(
        self,
        conversations: Sequence[list["ChatCompletionMessageParam"]],
        chat_params: ChatParams,
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_chat_tok_params

        rendered = [
            self.render_messages_async(conversation, chat_params)
            for conversation in conversations
        ]

        out_conversations = list[list["ConversationMessage"]]()
        dict_prompts = list[DictPrompt]()
        for conv, prompt in await asyncio.gather(*rendered):
            out_conversations.append(conv)
            dict_prompts.append(prompt)

        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        eng_prompts = await asyncio.gather(
            *(
                self.process_for_engine_async(
                    p, arrival_time, skip_mm_cache=skip_mm_cache
                )
                for p in tok_prompts
            )
        )

        return out_conversations, eng_prompts