"""Errors the Minimax nodes raise beyond what `NodeError` builds itself.

A missing key and a failed HTTP call come from `NodeError.from_missing_secret` and
`NodeError.from_request_exception`. Minimax also reports failures inside an HTTP 200 body, as
`{"base_resp": {"status_code": 1004, "status_msg": "..."}}` with 0 meaning success. This module reads
those, and covers a task that ends without a result.

It is imported by bare name because the engine loads each node file with the library folder on `sys.path`.
"""

from __future__ import annotations

from typing import Any

import requests
from griptape_nodes.exe_types.core_types import NodeError, NodeErrorLink

__all__ = [
    "API_KEY_ENV_VAR",
    "API_KEY_URL",
    "CREDENTIAL_REJECTED_CODES",
    "PROVIDER",
    "base_resp_error",
    "request_error",
    "request_json",
    "result_error",
    "task_failed_error",
]

API_KEY_ENV_VAR = "MINIMAX_API_KEY"
API_KEY_URL = "https://platform.minimax.io/console/keys"
PROVIDER = "Minimax"

# 1004 is "login fail" and 2049 is "invalid api key".
CREDENTIAL_REJECTED_CODES = (1004, 2049)
HTTP_UNAUTHORIZED = 401


def base_resp_error(body: dict[str, Any], *, fields: dict[str, Any] | None = None) -> NodeError | None:
    """The error for a body whose `base_resp.status_code` is not 0, or None when it succeeded."""
    base_resp = body.get("base_resp")
    if not isinstance(base_resp, dict) or base_resp.get("status_code") in (None, 0):
        return None
    code = base_resp["status_code"]
    reason = str(base_resp.get("status_msg") or "no reason given").strip()
    error_fields = {"error_code": code, **(fields or {})}
    if code in CREDENTIAL_REJECTED_CODES:
        return NodeError(
            f"{PROVIDER} refused the API key: {reason.rstrip('.')}. "
            f"Update {API_KEY_ENV_VAR} in Settings → API Keys & Secrets, then run the node again.",
            fields=error_fields,
            response=body,
            links=[NodeErrorLink.from_secret(API_KEY_ENV_VAR, "Update the API key")],
        )
    return NodeError(reason, fields=error_fields, response=body)


def request_error(exc: requests.RequestException, *, fields: dict[str, Any] | None = None) -> NodeError:
    """The error for a failed call. Only an HTTP 401 reads as a refused key.

    Minimax reports a rejected key in a 200 body (see `base_resp_error`), so an HTTP 403 is not assumed to
    be an auth failure and shows Minimax's own message.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return NodeError.from_request_exception(
        exc, provider=PROVIDER, secret_name=API_KEY_ENV_VAR, key_refused=status == HTTP_UNAUTHORIZED, fields=fields
    )


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call a Minimax endpoint and return its JSON body.

    Raises:
        NodeError: For an HTTP or connection failure, an unreadable body, or a `base_resp` failure.
    """
    try:
        response = requests.request(method, url, headers=headers, params=params, json=json, timeout=timeout)
        response.raise_for_status()
        body = response.json()
    except requests.RequestException as e:
        raise request_error(e, fields=fields) from e
    if not isinstance(body, dict):
        raise NodeError(f"{PROVIDER} sent a response the node could not read.", fields=fields, response={"body": body})
    if error := base_resp_error(body, fields={"status_code": response.status_code, **(fields or {})}):
        raise error
    return body


def result_error(
    what: str, *, fields: dict[str, Any] | None = None, response: dict[str, Any] | None = None
) -> NodeError:
    """The error for a successful call whose result is missing something the node needs."""
    return NodeError(f"{PROVIDER} finished but its result has no {what}.", fields=fields, response=response)


def task_failed_error(body: dict[str, Any], *, task_id: str) -> NodeError:
    """The error for a task Minimax accepted and then reported as failed."""
    base_resp = body.get("base_resp") if isinstance(body.get("base_resp"), dict) else {}
    reason = base_resp.get("status_msg") or body.get("error_message") or "no reason given"
    fields: dict[str, Any] = {"task_id": task_id, "status": "Fail"}
    if base_resp.get("status_code") not in (None, 0):
        fields["error_code"] = base_resp["status_code"]
    return NodeError(f"{PROVIDER} could not finish the video: {str(reason).rstrip('.')}.", fields=fields, response=body)
