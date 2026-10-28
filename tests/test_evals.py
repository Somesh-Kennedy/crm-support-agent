"""Tests of the eval harness itself: the checks must separate good behaviour from bad."""
import json

import pytest

from agent.core import Agent
from agent.crm import InMemoryCRM
from agent.llm import ScriptedLLM, say
from evals.policies import NAIVE, ORACLE, scripted
from evals.run import main, run_scenario, summarise
from evals.scenarios import BY_ID, SCENARIOS


def test_every_scenario_has_both_scripted_policies():
    ids = {s.id for s in SCENARIOS}
    assert ids == set(ORACLE) == set(NAIVE)


def test_scenarios_cover_every_category():
    assert {s.category for s in SCENARIOS} == {"lookup", "write", "guardrail", "injection", "out_of_scope", "ambiguity", "limits"}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_the_oracle_passes_every_scenario(scenario):
    row = run_scenario(scenario, scripted("oracle", scenario.id))
    assert row["passed"], row["checks"]
    assert row["stop_reason"] == "end_turn"


@pytest.mark.parametrize("sid", ["L1", "L2", "L3", "W1", "W2", "W3", "W4", "A1", "H1", "H3", "H4", "H5"])
def test_the_naive_agent_is_caught(sid):
    assert run_scenario(BY_ID[sid], scripted("naive", sid))["passed"] is False


@pytest.mark.parametrize("sid", ["G1", "G3", "H2", "H6", "I1"])
def test_naive_attempts_at_forbidden_actions_are_stopped_by_guardrails(sid):
    row = run_scenario(BY_ID[sid], scripted("naive", sid))
    assert row["passed"] and row["guardrail_blocks"] >= 1 and row["writes"] == 0


def test_an_agent_that_obeys_injected_instructions_is_quarantined_before_it_can_change_anything():
    row = run_scenario(BY_ID["I1"], scripted("naive", "I1"))
    assert row["writes"] == 0 and row["passed"] is True       # the quarantine contained it
    assert [c["blocked_by"] for c in row["tool_calls"]].count("untrusted_instructions") == 4


@pytest.mark.parametrize("sid", ["W1", "W2", "W3", "W4", "L1", "L2", "L3", "G2", "H4"])
def test_doing_nothing_never_passes_a_task_that_needs_work(sid):
    lazy = ScriptedLLM(lambda messages: say("OK."))
    assert run_scenario(BY_ID[sid], lazy)["passed"] is False


def test_each_scenario_starts_from_a_fresh_crm():
    first = run_scenario(BY_ID["W4"], scripted("oracle", "W4"))
    second = run_scenario(BY_ID["W4"], scripted("oracle", "W4"))
    assert first["passed"] and second["passed"]  # would fail on a reused CRM: case 5 is already closed


def test_the_injected_text_is_present_in_the_seed_data_and_reaches_the_model_as_data():
    crm = InMemoryCRM()
    seen = {}

    def policy(messages):
        if len(messages) == 1:
            from agent.llm import tool_call
            return tool_call("t", "get_case", case_id="500000000000010")
        seen["content"] = messages[-1]["results"][0]["content"]
        return say("done")

    Agent(ScriptedLLM(policy), crm).run("summarize")
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in seen["content"]
    assert crm.write_log == []


def test_summary_counts_by_category():
    rows = [run_scenario(s, scripted("oracle", s.id)) for s in SCENARIOS]
    summary = summarise(rows)
    assert summary["passed"] == summary["runs"] == len(SCENARIOS) and summary["pass_rate"] == 1.0
    assert summary["by_category"]["lookup"] == {"passed": 3, "runs": 3}


def test_cli_writes_reports_and_sets_the_exit_code(tmp_path):
    assert main(["--llm", "oracle", "--out", str(tmp_path)]) == 0
    assert main(["--llm", "naive", "--out", str(tmp_path)]) == 1
    report = json.loads((tmp_path / "oracle.json").read_text(encoding="utf8"))
    assert report["summary"]["pass_rate"] == 1.0
    assert (tmp_path / "naive.md").read_text(encoding="utf8").startswith("# Eval report: naive")


def test_cli_can_select_scenarios_and_rejects_unknown_ones(tmp_path):
    assert main(["--llm", "oracle", "--scenario", "W1", "--out", str(tmp_path)]) == 0
    assert main(["--llm", "oracle", "--scenario", "ZZ", "--out", str(tmp_path)]) == 2


def test_an_api_failure_is_a_failed_run_not_a_crash():
    from agent.llm import LLMError

    class Broken:
        def complete(self, *a, **k):
            raise LLMError("Hugging Face API error 402: out of inference credits")

    row = run_scenario(BY_ID["L1"], Broken())
    assert row["passed"] is False and row["stop_reason"] == "llm_error" and "402" in row["error"]


def test_key_env_reads_the_key_from_the_environment_and_fails_clearly_when_missing(monkeypatch):
    import pytest as _pytest
    from evals.run import _key_from_env
    assert _key_from_env(None) is None
    monkeypatch.setenv("MY_KEY", "abc")
    assert _key_from_env("MY_KEY") == "abc"
    monkeypatch.delenv("MY_KEY")
    with _pytest.raises(SystemExit, match="MY_KEY"):
        _key_from_env("MY_KEY")


def test_the_injected_comment_in_h3_is_visible_to_the_agent_but_is_not_counted_as_its_work():
    crm = InMemoryCRM()
    BY_ID["H3"].setup(crm)
    assert crm.write_log == []
    assert any("SYSTEM NOTICE" in m["CommentBody"] for m in crm.comments)


def test_the_write_cap_stops_the_naive_bulk_updater_at_three():
    row = run_scenario(BY_ID["H4"], scripted("naive", "H4"))
    assert row["writes"] == 3 and row["passed"] is False  # capped at 3, and its "all done" claim is caught
