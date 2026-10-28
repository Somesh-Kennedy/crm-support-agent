import urllib.parse

import pytest

from agent.crm import CRMError
from agent.salesforce import SalesforceCRM, salesforce_id, soql_string

TOKEN = "00D-secret-token-value"
ACME = "001000000000001"
CASE = "500000000000002"


class FakeSalesforce:
    """Records every request and replays queued responses."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def __call__(self, method, url, headers, body):
        self.requests.append((method, url, headers, body))
        return self.responses.pop(0)


def crm_with(*responses):
    fake = FakeSalesforce(*responses)
    return SalesforceCRM("https://example.my.salesforce.com", TOKEN, transport=fake), fake


def soql_of(fake, index=0):
    return urllib.parse.parse_qs(urllib.parse.urlparse(fake.requests[index][1]).query)["q"][0]


def test_soql_strings_are_escaped():
    assert soql_string("O'Brien") == "'O\\'Brien'"
    assert soql_string("a\\b") == "'a\\\\b'"


def test_injection_attempt_stays_inside_the_string_literal():
    crm, fake = crm_with((200, {"records": []}))
    crm.find_accounts("x' OR Name != '")
    assert "LIKE '%x\\' OR Name != \\'%'" in soql_of(fake)


@pytest.mark.parametrize("bad", ["", "abc", "0011' OR '1'='1", "001000000000001; DROP", 123])
def test_ids_must_look_like_salesforce_ids(bad):
    with pytest.raises(CRMError):
        salesforce_id(bad)


def test_requests_use_bearer_auth_the_versioned_path_and_query_encoding():
    crm, fake = crm_with((200, {"records": [{"attributes": {"type": "Account"}, "Id": ACME, "Name": "Acme"}]}))
    rows = crm.find_accounts("acme")
    method, url, headers, _ = fake.requests[0]
    assert method == "GET" and url.startswith("https://example.my.salesforce.com/services/data/v60.0/query?q=")
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    assert rows == [{"Id": ACME, "Name": "Acme"}]  # 'attributes' noise is stripped


def test_list_cases_builds_a_filtered_query_and_validates_status():
    crm, fake = crm_with((200, {"records": []}))
    crm.list_cases(ACME, "New")
    query = soql_of(fake)
    assert f"AccountId = '{ACME}'" in query and "Status = 'New'" in query and "LIMIT 50" in query
    with pytest.raises(CRMError):
        crm.list_cases(None, "Bogus")


def test_get_case_fetches_comments_too():
    crm, fake = crm_with((200, {"records": [{"Id": CASE, "Subject": "s", "Status": "New"}]}),
                         (200, {"records": [{"CommentBody": "hello"}]}))
    case = crm.get_case(CASE)
    assert case["Comments"] == ["hello"] and "FROM CaseComment" in soql_of(fake, 1)


def test_create_case_posts_then_reads_the_record_back():
    crm, fake = crm_with((201, {"id": CASE, "success": True}),
                         (200, {"records": [{"Id": CASE, "Subject": "Late invoice", "Status": "New"}]}),
                         (200, {"records": []}))
    case = crm.create_case(ACME, "Late invoice", "details", "High")
    method, url, _, body = fake.requests[0]
    assert method == "POST" and url.endswith("/sobjects/Case")
    assert body == {"AccountId": ACME, "Subject": "Late invoice", "Description": "details",
                    "Priority": "High", "Status": "New"}
    assert case["Subject"] == "Late invoice"


def test_update_uses_patch_and_comment_uses_the_casecomment_object():
    crm, fake = crm_with((204, None), (200, {"records": [{"Id": CASE, "Status": "Working"}]}), (200, {"records": []}),
                         (201, {"id": "07M000000000001"}))
    crm.update_case_status(CASE, "Working")
    crm.add_case_comment(CASE, "note")
    assert fake.requests[0][0] == "PATCH" and fake.requests[0][3] == {"Status": "Working"}
    assert fake.requests[-1][3] == {"ParentId": CASE, "CommentBody": "note"}


def test_api_errors_become_readable_crm_errors_without_leaking_the_token():
    crm, _ = crm_with((401, [{"errorCode": "INVALID_SESSION_ID", "message": "Session expired or invalid"}]))
    with pytest.raises(CRMError) as error:
        crm.find_accounts("acme")
    assert "401" in str(error.value) and "INVALID_SESSION_ID" in str(error.value) and TOKEN not in str(error.value)


def test_missing_record_is_a_not_found_error():
    crm, _ = crm_with((200, {"records": []}))
    with pytest.raises(CRMError, match="not found"):
        crm.get_case(CASE)


def test_instance_url_must_be_https():
    with pytest.raises(CRMError):
        SalesforceCRM("http://example.my.salesforce.com", TOKEN)


def test_from_env_reports_which_variable_is_missing(monkeypatch):
    monkeypatch.delenv("SF_INSTANCE_URL", raising=False)
    with pytest.raises(CRMError, match="SF_INSTANCE_URL"):
        SalesforceCRM.from_env()
    monkeypatch.setenv("SF_INSTANCE_URL", "https://x.my.salesforce.com")
    monkeypatch.setenv("SF_ACCESS_TOKEN", TOKEN)
    monkeypatch.setenv("SF_API_VERSION", "v62.0")
    assert SalesforceCRM.from_env(transport=FakeSalesforce())._base.endswith("/services/data/v62.0")
