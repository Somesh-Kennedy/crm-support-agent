from types import SimpleNamespace

from agent.llm import AnthropicLLM
from agent.tools import TOOLS


class FakeMessages:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def fake_client(content, stop_reason="tool_use"):
    usage = SimpleNamespace(input_tokens=11, output_tokens=7)
    response = SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)
    messages = FakeMessages(response)
    return SimpleNamespace(messages=messages), messages


def test_request_uses_the_documented_parameters():
    client, messages = fake_client([SimpleNamespace(type="text", text="hi")], "end_turn")
    AnthropicLLM(client=client).complete("sys", [{"role": "user", "content": "x"}], TOOLS)
    sent = messages.kwargs
    assert sent["model"] == "claude-opus-5-5"
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["system"] == "sys" and sent["tools"] == TOOLS
    assert "tool_choice" not in sent  # forced tool use is rejected by current models; we leave it on auto
    assert sent["max_tokens"] >= 1024


def test_model_is_configurable():
    client, messages = fake_client([SimpleNamespace(type="text", text="hi")], "end_turn")
    AnthropicLLM(model="claude-sonnet-5-5", client=client).complete("s", [], TOOLS)
    assert messages.kwargs["model"] == "claude-sonnet-5-5"


def test_tool_use_and_text_blocks_are_parsed_and_raw_content_is_kept():
    thinking = SimpleNamespace(type="thinking", thinking="", signature="sig")
    content = [thinking, SimpleNamespace(type="text", text="Looking up"),
               SimpleNamespace(type="tool_use", id="tu_1", name="list_cases", input={"status": "New"})]
    client, _ = fake_client(content)
    reply = AnthropicLLM(client=client).complete("s", [], TOOLS)
    assert [b.type for b in reply.blocks] == ["text", "tool_use"]
    assert reply.blocks[1].id == "tu_1" and reply.blocks[1].input == {"status": "New"}
    assert reply.raw_content is content  # thinking blocks must be sent back unchanged
    assert (reply.input_tokens, reply.output_tokens, reply.stop_reason) == (11, 7, "tool_use")
