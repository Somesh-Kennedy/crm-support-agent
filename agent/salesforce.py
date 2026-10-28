"""Salesforce REST API backend for the CRM interface.

Talks to a real Salesforce org (for example a free Developer Edition org) using
an access token and instance URL. The HTTP layer is injectable, so the request
shapes, SOQL escaping and error handling are unit tested without a network.

Credentials come from the environment and are never logged or put in errors:
    SF_INSTANCE_URL   e.g. https://yourorg.my.salesforce.com
    SF_ACCESS_TOKEN   e.g. from `sf org display --target-org <alias> --json`
    SF_API_VERSION    optional, default v60.0
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

from .crm import CASE_PRIORITIES, CASE_STATUSES, CRMError

# (method, url, headers, json_body) -> (status_code, parsed_json_or_None)
Transport = Callable[[str, str, dict, dict | None], tuple[int, object]]

_ID = re.compile(r"^[a-zA-Z0-9]{15,18}$")


def soql_string(value: str) -> str:
    """Quote a value for use inside a SOQL string literal (prevents SOQL injection)."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def salesforce_id(value: str) -> str:
    if not isinstance(value, str) or not _ID.match(value):
        raise CRMError(f"'{value}' is not a valid Salesforce record id")
    return value


def _picklist(value: str, allowed: tuple[str, ...], what: str) -> str:
    if value not in allowed:
        raise CRMError(f"{what} must be one of {', '.join(allowed)}")
    return value


def urllib_transport(method: str, url: str, headers: dict, body: dict | None) -> tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            return response.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, None


class SalesforceCRM:
    def __init__(self, instance_url: str, access_token: str, api_version: str = "v60.0",
                 transport: Transport = urllib_transport):
        if not instance_url.startswith("https://"):
            raise CRMError("instance_url must start with https://")
        self._base = f"{instance_url.rstrip('/')}/services/data/{api_version}"
        self._headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        self._transport = transport

    @classmethod
    def from_env(cls, transport: Transport = urllib_transport) -> "SalesforceCRM":
        try:
            return cls(os.environ["SF_INSTANCE_URL"], os.environ["SF_ACCESS_TOKEN"],
                       os.environ.get("SF_API_VERSION", "v60.0"), transport)
        except KeyError as missing:
            raise CRMError(f"Set the {missing.args[0]} environment variable") from None

    # -- HTTP helpers ----------------------------------------------------------
    def _call(self, method: str, path: str, body: dict | None = None) -> object:
        status, payload = self._transport(method, f"{self._base}{path}", self._headers, body)
        if status >= 400:
            detail = ""
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                detail = f": {payload[0].get('errorCode', '')} {payload[0].get('message', '')}".rstrip()
            raise CRMError(f"Salesforce API returned HTTP {status}{detail}")
        return payload

    def _query(self, soql: str) -> list[dict]:
        payload = self._call("GET", "/query?" + urllib.parse.urlencode({"q": soql}))
        records = payload.get("records", []) if isinstance(payload, dict) else []
        return [{k: v for k, v in r.items() if k != "attributes"} for r in records]

    # -- CRM interface -----------------------------------------------------------
    def find_accounts(self, name_contains: str) -> list[dict]:
        pattern = soql_string("%" + name_contains.replace("%", "").replace("_", "") + "%")
        return self._query(f"SELECT Id, Name, Industry, Type FROM Account WHERE Name LIKE {pattern} LIMIT 20")

    def get_account(self, account_id: str) -> dict:
        rows = self._query(f"SELECT Id, Name, Industry, Type FROM Account WHERE Id = {soql_string(salesforce_id(account_id))}")
        if not rows:
            raise CRMError(f"Account {account_id} not found")
        return rows[0]

    def list_cases(self, account_id: str | None = None, status: str | None = None) -> list[dict]:
        clauses = []
        if account_id:
            clauses.append(f"AccountId = {soql_string(salesforce_id(account_id))}")
        if status:
            clauses.append(f"Status = {soql_string(_picklist(status, CASE_STATUSES, 'status'))}")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._query("SELECT Id, CaseNumber, Subject, Status, Priority, AccountId, CreatedDate "
                           f"FROM Case{where} ORDER BY CreatedDate DESC LIMIT 50")

    def get_case(self, case_id: str) -> dict:
        cid = soql_string(salesforce_id(case_id))
        rows = self._query("SELECT Id, CaseNumber, Subject, Description, Status, Priority, AccountId, CreatedDate "
                           f"FROM Case WHERE Id = {cid}")
        if not rows:
            raise CRMError(f"Case {case_id} not found")
        case = rows[0]
        case["Comments"] = [r["CommentBody"] for r in self._query(
            f"SELECT CommentBody FROM CaseComment WHERE ParentId = {cid} ORDER BY CreatedDate")]
        return case

    def create_case(self, account_id: str, subject: str, description: str, priority: str) -> dict:
        result = self._call("POST", "/sobjects/Case", {
            "AccountId": salesforce_id(account_id), "Subject": subject, "Description": description,
            "Priority": _picklist(priority, CASE_PRIORITIES, "priority"), "Status": "New",
        })
        return self.get_case(result["id"])

    def update_case_status(self, case_id: str, status: str) -> dict:
        self._call("PATCH", f"/sobjects/Case/{salesforce_id(case_id)}",
                   {"Status": _picklist(status, CASE_STATUSES, "status")})
        return self.get_case(case_id)

    def add_case_comment(self, case_id: str, body: str) -> dict:
        self._call("POST", "/sobjects/CaseComment", {"ParentId": salesforce_id(case_id), "CommentBody": body})
        return {"ParentId": case_id, "CommentBody": body}
