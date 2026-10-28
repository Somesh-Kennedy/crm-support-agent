import json
from types import SimpleNamespace

import pytest

from agent.llm import Block, HuggingFaceLLM, LLMError, to_openai_messages, to_openai_tools
from agent.tools import TOOLS


def completion(content=None, tool_calls=None, finish="stop", usage=(10, 5)):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]))


def call(name, arguments, call_id="c1"):
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=args))


class FakeClient:
    def __init__(self, *outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class HttpError(Exception):
    def __init__(self, status, text="boom"):
        super().__init__(text)
        self.response = SimpleNamespace(status_code=status)


def llm(*outcomes, **kw):
    client = FakeClient(*outcomes)
    return HuggingFaceLLM(client=client, sleep=lambda s: None, **kw), client


# ---- request shape --------------------------------------------------------------------

def test_tools_are_sent_in_openai_function_format():
    tools = to_openai_tools(TOOLS)
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == TOOLS[0]["name"]
    assert tools[0]["function"]["parameters"] == TOOLS[0]["input_schema"]


def test_request_is_deterministic_and_bounded():
    model, client = llm(completion("hi"))
    model.complete("sys", [{"role": "user", "content": "x"}], TOOLS)
    sent = client.requests[0]
    assert sent["temperature"] == 0.0 and sent["max_tokens"] == 2048
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert "tool_choice" not in sent


def test_conversation_converts_to_openai_messages():
    neutral = [
        {"role": "user", "content": "close it"},
        {"role": "assistant", "blocks": [Block("text", text="Checking"),
                                         Block("tool_use", id="a", name="get_case", input={"case_id": "5"}),
                                         Block("tool_use", id="b", name="list_cases", input={})], "raw": None},
        {"role": "tool_results", "results": [{"id": "a", "content": "{}", "is_error": False},
                                             {"id": "b", "content": "{}", "is_error": False}]},
    ]
    out = to_openai_messages("sys", neutral)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "tool"]
    assistant = out[2]
    assert assistant["content"] == "Checking"
    assert [c["id"] for c in assistant["tool_calls"]] == ["a", "b"]
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"case_id": "5"}
    assert [m["tool_call_id"] for m in out[3:]] == ["a", "b"]


def test_an_assistant_turn_with_only_tool_calls_has_null_content():
    out = to_openai_messages("s", [{"role": "assistant", "blocks": [Block("tool_use", id="a", name="x", input={})], "raw": None}])
    assert out[1]["content"] is None


# ---- response parsing -------------------------------------------------------------------

def test_plain_answer():
    model, _ = llm(completion("All done.", finish="stop"))
    reply = model.complete("s", [], TOOLS)
    assert [b.type for b in reply.blocks] == ["text"] and reply.stop_reason == "end_turn"
    assert (reply.input_tokens, reply.output_tokens) == (10, 5)


def test_tool_calls_are_parsed_with_json_arguments():
    model, _ = llm(completion(None, [call("list_cases", {"status": "New"}, "x1"), call("get_case", {"case_id": "5"}, "x2")]))
    reply = model.complete("s", [], TOOLS)
    assert reply.stop_reason == "tool_use"
    assert [(b.id, b.name, b.input) for b in reply.blocks] == [
        ("x1", "list_cases", {"status": "New"}), ("x2", "get_case", {"case_id": "5"})]


def test_invalid_json_arguments_are_passed_on_as_a_string_not_a_crash():
    model, _ = llm(completion(None, [call("get_case", "{not json")]))
    block = model.complete("s", [], TOOLS).blocks[0]
    assert block.input == "{not json"


def test_missing_tool_call_id_gets_a_stable_fallback():
    model, _ = llm(completion(None, [call("list_cases", {}, call_id=None)]))
    assert model.complete("s", [], TOOLS).blocks[0].id == "call_0"


def test_length_finish_reason_is_reported_as_max_tokens():
    model, _ = llm(completion("partial answ", finish="length"))
    assert model.complete("s", [], TOOLS).stop_reason == "max_tokens"


def test_empty_choices_is_an_error():
    model, _ = llm(SimpleNamespace(choices=[], usage=None))
    with pytest.raises(LLMError):
        model.complete("s", [], TOOLS)


# ---- retries and errors --------------------------------------------------------------------

def test_rate_limits_are_retried_then_succeed():
    sleeps = []
    client = FakeClient(HttpError(429), HttpError(503), completion("ok"))
    model = HuggingFaceLLM(client=client, sleep=sleeps.append)
    assert model.complete("s", [], TOOLS).blocks[0].text == "ok"
    assert len(client.requests) == 3 and sleeps == [2.0, 4.0]


def test_retries_give_up_with_a_clear_error():
    model, client = llm(*[HttpError(429) for _ in range(6)])
    with pytest.raises(LLMError, match="429"):
        model.complete("s", [], TOOLS)
    assert len(client.requests) == 6


def test_the_retry_after_header_is_honoured():
    slow = HttpError(429)
    slow.response.headers = {"retry-after": "7"}
    sleeps = []
    client = FakeClient(slow, completion("ok"))
    HuggingFaceLLM(client=client, sleep=sleeps.append).complete("s", [], TOOLS)
    assert sleeps == [7.0]


@pytest.mark.parametrize("status, hint", [(401, "token"), (402, "credits"), (404, "not served")])
def test_non_retryable_errors_fail_fast_with_a_helpful_hint(status, hint):
    model, client = llm(HttpError(status))
    with pytest.raises(LLMError, match=hint):
        model.complete("s", [], TOOLS)
    assert len(client.requests) == 1


def test_a_provider_side_parse_failure_is_retried_with_a_little_randomness():
    glitch = HttpError(400, "Parsing failed. The model generated output that could not be parsed. failed_generation")
    client = FakeClient(glitch, completion("ok"))
    model = HuggingFaceLLM(client=client, sleep=lambda s: None)
    assert model.complete("s", [], TOOLS).blocks[0].text == "ok"
    assert [r["temperature"] for r in client.requests] == [0.0, 0.3]


def test_persistent_parse_failures_eventually_surface_as_an_error():
    glitch = lambda: HttpError(400, "tool_use_failed")  # noqa: E731
    model, client = llm(*[glitch() for _ in range(6)])
    with pytest.raises(LLMError):
        model.complete("s", [], TOOLS)
    assert len(client.requests) <= 6


def test_default_output_budget_leaves_room_for_reasoning_models():
    assert HuggingFaceLLM(client=FakeClient()).max_tokens >= 2048


def test_an_empty_turn_is_resampled_with_a_little_randomness():
    empty = completion("", None, finish="length")
    model, client = llm(empty, empty, completion("Here you go."))
    reply = model.complete("s", [], TOOLS)
    assert reply.blocks[0].text == "Here you go."
    assert [r["temperature"] for r in client.requests] == [0.0, 0.3, 0.6]


def test_a_turn_that_is_empty_every_time_is_returned_not_looped_forever():
    empty = completion("", None, finish="length")
    model, client = llm(empty, empty, empty)
    reply = model.complete("s", [], TOOLS)
    assert reply.blocks == [] and reply.stop_reason == "max_tokens" and len(client.requests) == 3


def test_a_normal_answer_is_not_resampled():
    model, client = llm(completion("fine"))
    model.complete("s", [], TOOLS)
    assert len(client.requests) == 1


def test_the_real_client_is_always_created_with_a_timeout(monkeypatch):
    import huggingface_hub
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(huggingface_hub, "InferenceClient", Recorder)
    HuggingFaceLLM(model="m", provider="groq", token="t")
    assert seen["timeout"] == 90.0 and seen["provider"] == "groq" and seen["model"] == "m"


def test_a_timed_out_request_is_retried():
    class ReadTimeout(Exception):
        pass

    client = FakeClient(ReadTimeout("The read operation timed out"), completion("ok"))
    model = HuggingFaceLLM(client=client, sleep=lambda s: None)
    assert model.complete("s", [], TOOLS).blocks[0].text == "ok"
    assert len(client.requests) == 2
