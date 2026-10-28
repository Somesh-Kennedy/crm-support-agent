"""LLM backends behind one small interface.

The agent keeps its conversation in a provider-neutral format:

    {"role": "user", "content": "text"}
    {"role": "assistant", "blocks": [Block, ...], "raw": <provider object, replayed verbatim if needed>}
    {"role": "tool_results", "results": [{"id": ..., "content": "...", "is_error": bool}, ...]}

Each backend converts that to its own wire format:
  HuggingFaceLLM  Hugging Face Inference Providers API (OpenAI-style chat + tools). The default.
  AnthropicLLM    Claude via the Anthropic SDK (optional).
  ScriptedLLM     replays a fixed policy, so the loop, guardrails and evals are testable offline.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

DEFAULT_HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


class LLMError(Exception):
    """The model backend failed (network, auth, rate limit, bad response)."""


@dataclass
class Block:
    type: str                      # "text" | "tool_use"
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: Any = None              # normally a dict; the raw string if the model produced invalid JSON


@dataclass
class LLMResponse:
    blocks: list[Block]
    stop_reason: str               # end_turn | tool_use | max_tokens | refusal
    raw_content: Any = None
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    def complete(self, system: str, messages: list, tools: list) -> LLMResponse: ...


# ---------------------------------------------------------------------------------------------
# Hugging Face Inference Providers
# ---------------------------------------------------------------------------------------------

def to_openai_tools(tools: list) -> list:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools]


def to_openai_messages(system: str, messages: list) -> list:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            text = "".join(b.text or "" for b in m["blocks"] if b.type == "text")
            calls = [{"id": b.id, "type": "function",
                      "function": {"name": b.name,
                                   "arguments": b.input if isinstance(b.input, str) else json.dumps(b.input or {})}}
                     for b in m["blocks"] if b.type == "tool_use"]
            message: dict = {"role": "assistant", "content": text or None}
            if calls:
                message["tool_calls"] = calls
            out.append(message)
        else:  # tool_results -> one "tool" message per result
            out.extend({"role": "tool", "tool_call_id": r["id"], "content": r["content"]} for r in m["results"])
    return out


class HuggingFaceLLM:
    """Chat completion with tool calling through the Hugging Face Inference Providers API.

    Credentials: pass token=, or set HF_TOKEN, or log in with `hf auth login`.
    """

    def __init__(self, model: str = DEFAULT_HF_MODEL, provider: str = "auto", token: str | None = None,
                 max_tokens: int = 1024, max_retries: int = 3, client: Any = None,
                 sleep: Callable[[float], None] = time.sleep):
        if client is None:
            from huggingface_hub import InferenceClient  # imported lazily so offline tests need no network setup
            client = InferenceClient(model=model, provider=provider, token=token)
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.max_retries, self._sleep = max_retries, sleep

    def _create(self, **kwargs):
        delay = 2.0
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.chat_completion(**kwargs)
            except Exception as error:  # huggingface_hub raises several error types; classify by HTTP status
                status = getattr(getattr(error, "response", None), "status_code", None)
                retryable = status in (408, 429, 500, 502, 503, 504) or "timed out" in str(error).lower()
                if not retryable or attempt == self.max_retries:
                    hint = {401: " (check your Hugging Face token)", 402: " (out of inference credits)",
                            403: " (token lacks permission for this model)", 404: " (model not served by any provider)"}.get(status, "")
                    raise LLMError(f"Hugging Face API error{f' {status}' if status else ''}: {str(error)[:200]}{hint}") from error
                self._sleep(delay)
                delay *= 2

    def complete(self, system: str, messages: list, tools: list) -> LLMResponse:
        response = self._create(messages=to_openai_messages(system, messages), tools=to_openai_tools(tools),
                                max_tokens=self.max_tokens, temperature=0.0)
        if not getattr(response, "choices", None):
            raise LLMError("Hugging Face API returned no choices")
        choice = response.choices[0]
        message = choice.message
        blocks = []
        text = (message.content or "").strip()
        if text:
            blocks.append(Block("text", text=text))
        for index, call in enumerate(message.tool_calls or []):
            raw_args = call.function.arguments
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
            except ValueError:
                arguments = raw_args  # not valid JSON: the executor reports it back to the model as an error
            blocks.append(Block("tool_use", id=call.id or f"call_{index}", name=call.function.name, input=arguments))
        if any(b.type == "tool_use" for b in blocks):
            stop = "tool_use"
        else:
            stop = "max_tokens" if choice.finish_reason == "length" else "end_turn"
        usage = getattr(response, "usage", None)
        return LLMResponse(blocks, stop, message,
                           getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0)


# ---------------------------------------------------------------------------------------------
# Anthropic (optional)
# ---------------------------------------------------------------------------------------------

def to_anthropic_messages(messages: list) -> list:
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            out.append({"role": "assistant", "content": m["raw"]})  # includes thinking blocks, which must be echoed back
        else:
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"], "is_error": r["is_error"]}
                for r in m["results"]]})
    return out


class AnthropicLLM:
    """Claude via the Anthropic SDK. Reads ANTHROPIC_API_KEY (or an `ant auth login` profile)."""

    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 16000, client: Any = None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.max_tokens = client, model, max_tokens

    def complete(self, system: str, messages: list, tools: list) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            thinking={"type": "adaptive"}, tools=tools, messages=to_anthropic_messages(messages),
        )
        blocks = []
        for block in response.content:
            if block.type == "text":
                blocks.append(Block("text", text=block.text))
            elif block.type == "tool_use":
                blocks.append(Block("tool_use", id=block.id, name=block.name, input=block.input))
        return LLMResponse(blocks, response.stop_reason, response.content,
                           response.usage.input_tokens, response.usage.output_tokens)


# ---------------------------------------------------------------------------------------------
# Scripted (tests and harness validation)
# ---------------------------------------------------------------------------------------------

@dataclass
class ScriptedLLM:
    """Replays decisions from a policy function: policy(messages) -> LLMResponse."""
    policy: Callable[[list], LLMResponse]
    calls: int = field(default=0)

    def complete(self, system: str, messages: list, tools: list) -> LLMResponse:
        self.calls += 1
        return self.policy(messages)


def tool_call(call_id: str, name: str, **args) -> LLMResponse:
    return LLMResponse([Block("tool_use", id=call_id, name=name, input=args)], "tool_use", None)


def say(text: str) -> LLMResponse:
    return LLMResponse([Block("text", text=text)], "end_turn", None)
