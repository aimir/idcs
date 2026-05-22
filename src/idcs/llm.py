"""Thin Anthropic SDK wrapper.

Defaults to claude-opus-4-7 with prompt caching on the system prompt and
adaptive thinking off. G and D will both call through this — keeping it tiny
and uniform means changing the model or caching strategy is one edit.

Note: prompt caching has a model-dependent minimum prefix length (~4K tokens
on Opus 4.7). Short system prompts get the cache_control marker but the API
silently skips caching them. That is fine — once prompts grow during
optimization the cache kicks in for free.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-opus-4-7"


class LLM:
    """One-shot completion wrapper with cached system prompts."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.model = model

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 16000,
        thinking: bool = False,
        cache_system: bool = True,
    ) -> str:
        """Return the concatenated text content of one completion.

        Streams when max_tokens exceeds the safe non-streaming threshold,
        so callers can bump max_tokens without worrying about SDK timeouts.
        """
        system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
        if cache_system:
            system_blocks[0]["cache_control"] = {"type": "ephemeral"}

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": [{"role": "user", "content": user}],
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        if max_tokens > 16000:
            with self.client.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
        else:
            message = self.client.messages.create(**kwargs)

        return "".join(block.text for block in message.content if block.type == "text")
