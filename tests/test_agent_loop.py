from agent.core import Agent
from agent.crm import InMemoryCRM
from agent.llm import Block, LLMResponse, ScriptedLLM, say, tool_call


def run(policy, **kw):
    crm = InMemoryCRM()
    return Agent(ScriptedLLM(policy), crm, **kw).run("hi"), crm


def sequence(*responses):
    it = iter(responses)
    return lambda messages: next(it)


def test_answers_directly_when_no_tools_are_needed():
    result, _ = run(sequence(say("Hello!")))
    assert (result.final_text, result.steps, result.tool_calls, result.stop_reason) == ("Hello!", 1, [], "end_turn")


def test_runs_a_tool_then_answers_using_its_result():
    seen = {}

    def policy(messages):
        if len(messages) == 1:
            return tool_call("t1", "search_accounts", name_contains="globex")
        seen["last"] = messages[-1]
        return say("found it")

    result, _ = run(policy)
    assert result.final_text == "found it" and result.steps == 2
    assert result.tool_calls[0].name == "search_accounts" and not result.tool_calls[0].is_error
    block = seen["last"]["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "t1" and "Globex" in block["content"]


def test_a_blocked_action_is_reported_back_to_the_model_as_an_error():
    seen = {}

    def policy(messages):
        if len(messages) == 1:
            return tool_call("t1", "update_case_status", case_id="500000000000008", status="Closed")
        seen["block"] = messages[-1]["content"][0]
        return say("I need a reason first.")

    result, crm = run(policy)
    assert seen["block"]["is_error"] is True
    assert result.tool_calls[0].blocked_by == "reason_required" and crm.write_log == []


def test_parallel_tool_calls_return_all_results_in_one_message():
    seen = {}

    def policy(messages):
        if len(messages) == 1:
            return LLMResponse(
                [Block("tool_use", id="a", name="list_cases", input={}),
                 Block("tool_use", id="b", name="search_accounts", input={"name_contains": "acme"})],
                "tool_use", [])
        seen["results"] = messages[-1]["content"]
        return say("done")

    run(policy)
    assert [r["tool_use_id"] for r in seen["results"]] == ["a", "b"]


def test_step_limit_stops_a_runaway_loop_safely():
    result, _ = run(lambda m: tool_call("t", "list_cases"), max_steps=3)
    assert result.stop_reason == "max_steps" and result.steps == 3 and len(result.tool_calls) == 3
    assert "stopped" in result.final_text.lower()


def test_refusal_is_surfaced_not_swallowed():
    refusal = LLMResponse([Block("text", text="I can't help with that.")], "refusal", [])
    result, _ = run(sequence(refusal))
    assert result.stop_reason == "refusal" and "can't help" in result.final_text


def test_token_usage_is_summed_across_steps():
    first = tool_call("t", "list_cases")
    first.input_tokens, first.output_tokens = 100, 20
    second = say("ok")
    second.input_tokens, second.output_tokens = 150, 30
    result, _ = run(sequence(first, second))
    assert (result.input_tokens, result.output_tokens) == (250, 50)


def test_write_limit_applies_across_one_request():
    plan = [tool_call(f"t{i}", "add_case_comment", case_id="500000000000001", body="n") for i in range(5)] + [say("done")]
    result, crm = run(sequence(*plan), max_writes=2, max_steps=10)
    assert result.writes == 2 and len(crm.comments) == 2
    assert [c.blocked_by for c in result.tool_calls].count("write_limit") == 3
