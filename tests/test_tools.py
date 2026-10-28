import json

import pytest

from agent.crm import InMemoryCRM
from agent.tools import ToolExecutor, contains_sensitive_data

ACME = "001000000000001"
GLOBEX = "001000000000002"


@pytest.fixture()
def crm():
    return InMemoryCRM()


@pytest.fixture()
def ex(crm):
    return ToolExecutor(crm)


def data(outcome):
    return json.loads(outcome.text)


# ---- reads ---------------------------------------------------------------------

def test_search_finds_every_matching_account(ex):
    names = {a["Name"] for a in data(ex.execute("search_accounts", {"name_contains": "acme"}))["data"]}
    assert names == {"Acme Corporation", "Acme Retail"}


def test_list_cases_filters_by_account_and_status(ex):
    cases = data(ex.execute("list_cases", {"account_id": ACME, "status": "New"}))["data"]
    assert [c["Subject"] for c in cases] == ["Cannot export monthly report"]


def test_get_case_includes_comments(crm, ex):
    crm.add_case_comment("500000000000002", "finance is reviewing")
    case = data(ex.execute("get_case", {"case_id": "500000000000002"}))["data"]
    assert case["Comments"] == ["finance is reviewing"]


def test_long_descriptions_are_truncated_for_the_model(crm, ex):
    crm.cases["500000000000001"]["Description"] = "x" * 5000
    case = data(ex.execute("get_case", {"case_id": "500000000000001"}))["data"]
    assert len(case["Description"]) < 900 and case["Description"].endswith("[truncated]")


def test_unknown_case_is_a_clean_error_not_a_crash(ex):
    outcome = ex.execute("get_case", {"case_id": "500999999999999"})
    assert outcome.is_error and "not found" in data(outcome)["error"]


# ---- guardrails on writes -----------------------------------------------------------

def test_closing_without_a_reason_is_blocked_and_changes_nothing(crm, ex):
    outcome = ex.execute("update_case_status", {"case_id": "500000000000008", "status": "Closed"})
    assert outcome.is_error and outcome.blocked_by == "reason_required"
    assert crm.cases["500000000000008"]["Status"] == "New" and crm.write_log == []


def test_a_token_reason_is_not_enough(crm, ex):
    outcome = ex.execute("update_case_status", {"case_id": "500000000000008", "status": "Closed", "reason": "ok"})
    assert outcome.blocked_by == "reason_required"


def test_closing_with_a_reason_works_and_leaves_an_audit_comment(crm, ex):
    outcome = ex.execute("update_case_status", {"case_id": "500000000000005", "status": "Closed",
                                                "reason": "Customer withdrew the request"})
    assert not outcome.is_error
    assert crm.cases["500000000000005"]["Status"] == "Closed"
    assert any("withdrew" in c["CommentBody"] for c in crm.comments)


def test_closed_cases_cannot_be_reopened(crm, ex):
    outcome = ex.execute("update_case_status", {"case_id": "500000000000003", "status": "Working"})
    assert outcome.blocked_by == "closed_case"
    assert crm.cases["500000000000003"]["Status"] == "Closed"


def test_invalid_status_and_priority_are_rejected(ex):
    assert ex.execute("update_case_status", {"case_id": "500000000000001", "status": "Done"}).blocked_by == "invalid_value"
    bad = ex.execute("create_case", {"account_id": ACME, "subject": "s", "description": "d", "priority": "Urgent"})
    assert bad.blocked_by == "invalid_value"


@pytest.mark.parametrize("text", ["card 4111 1111 1111 1111", "ssn 123-45-6789", "pay 4111-1111-1111-1111 now"])
def test_sensitive_data_is_never_written(crm, ex, text):
    outcome = ex.execute("add_case_comment", {"case_id": "500000000000007", "body": text})
    assert outcome.blocked_by == "sensitive_data" and crm.comments == []


def test_ordinary_long_numbers_are_not_mistaken_for_cards(ex):
    assert not contains_sensitive_data("order 4111111111111112 shipped")  # fails the Luhn check
    assert not contains_sensitive_data("invoice 4471 total 1,200.50")
    outcome = ex.execute("add_case_comment", {"case_id": "500000000000007", "body": "see order 4111111111111112"})
    assert outcome.is_error is False


def test_create_case_blocks_sensitive_data_in_description(crm, ex):
    outcome = ex.execute("create_case", {"account_id": GLOBEX, "subject": "Refund", "priority": "Low",
                                         "description": "my card is 4111 1111 1111 1111"})
    assert outcome.blocked_by == "sensitive_data" and crm.write_log == []


def test_write_limit_caps_the_damage_one_request_can_do(crm):
    ex = ToolExecutor(crm, max_writes=2)
    for case in ("500000000000001", "500000000000002"):
        assert not ex.execute("add_case_comment", {"case_id": case, "body": "note"}).is_error
    blocked = ex.execute("add_case_comment", {"case_id": "500000000000004", "body": "note"})
    assert blocked.blocked_by == "write_limit" and len(crm.comments) == 2


def test_reads_do_not_count_toward_the_write_limit(crm):
    ex = ToolExecutor(crm, max_writes=1)
    for _ in range(5):
        ex.execute("list_cases", {})
    assert not ex.execute("add_case_comment", {"case_id": "500000000000001", "body": "note"}).is_error


# ---- bad input ---------------------------------------------------------------------

def test_unknown_tool_non_object_and_missing_fields_are_reported(ex):
    assert ex.execute("drop_database", {}).blocked_by == "unknown_tool"
    assert ex.execute("get_case", "500000000000001").blocked_by == "bad_input"
    assert ex.execute("get_case", {}).blocked_by == "bad_input"
    assert ex.execute("add_case_comment", {"case_id": "500000000000001", "body": "   "}).is_error


def test_overlong_text_is_rejected(ex):
    assert ex.execute("add_case_comment", {"case_id": "500000000000001", "body": "x" * 2500}).is_error


def test_audit_log_records_every_attempt_including_blocked_ones(ex):
    ex.execute("update_case_status", {"case_id": "500000000000008", "status": "Closed"})
    ex.execute("list_cases", {})
    assert [(a["tool"], a["blocked_by"]) for a in ex.audit] == [("update_case_status", "reason_required"), ("list_cases", None)]
