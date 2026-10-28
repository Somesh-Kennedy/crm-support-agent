"""LLM clients behind one small interface.

AnthropicLLM calls Claude through the official SDK. ScriptedLLM replays a fixed
policy, which lets the agent loop, guardrails and eval harness be tested
deterministically, with no network and no cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class Block:
    type: str                      # "text" | "tool_use" (thinking etc. are kept only in raw_content)
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict | None = None


@dataclass
class LLMResponse:
    blocks: list[Block]
    stop_reason: str               # end_turn | tool_use | max_tokens | refusal | pause_turn
    raw_content: Any = None        # replayed verbatim into the next request (thinking blocks must survive)
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    def complete(self, system: str, messages: list, tools: list) -> LLMResponse: ...


class AnthropicLLM:
    """Claude via the Anthropic SDK. Reads ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment."""

    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 16000, client: Any = None):
        if client is None:
            import anthropic  # imported lazily so the rest of the project works without the package
            client = anthropic.Anthropic()
        self.client, self.model, self.max_tokens = client, model, max_tokens

    def complete(self, system: str, messages: list, tools: list) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            thinking={"type": "adaptive"}, tools=tools, messages=messages,
        )
        blocks = []
        for block in response.content:
            if block.type == "text":
                blocks.append(Block("text", text=block.text))
            elif block.type == "tool_use":
                blocks.append(Block("tool_use", id=block.id, name=block.name, input=block.input))
        return LLMResponse(blocks, response.stop_reason, response.content,
                           response.usage.input_tokens, response.usage.output_tokens)


@dataclass
class ScriptedLLM:
    """Replays decisions from a policy function: policy(messages) -> LLMResponse."""
    policy: Callable[[list], LLMResponse]
    calls: int = field(default=0)

    def complete(self, system: str, messages: list, tools: list) -> LLMResponse:
        self.calls += 1
        return self.policy(messages)


def tool_call(call_id: str, name: str, **args) -> LLMResponse:
    block = Block("tool_use", id=call_id, name=name, input=args)
    raw = [{"type": "tool_use", "id": call_id, "name": name, "input": args}]
    return LLMResponse([block], "tool_use", raw)


def say(text: str) -> LLMResponse:
    return LLMResponse([Block("text", text=text)], "end_turn", [{"type": "text", "text": text}])
