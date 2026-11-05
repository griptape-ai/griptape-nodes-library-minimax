"""Polling and file retrieval shared by the Minimax video nodes.

Imported by bare name (`from minimax_video import ...`), like `minimax_errors`, because the engine loads each
node file on its own with the library folder on `sys.path`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import requests
from griptape_nodes.exe_types.core_types import NodeError
from minimax_errors import (
    CREDENTIAL_REJECTED_CODES,
    base_resp_error,
    request_error,
    request_json,
    result_error,
    task_failed_error,
)

__all__ = ["poll_for_completion", "retrieve_video_file"]

QUERY_URL = "https://api.minimax.io/v1/query/video_generation"
RETRIEVE_URL = "https://api.minimax.io/v1/files/retrieve"
DEFAULT_TIMEOUT = 60


def poll_for_completion(
    task_id: str,
    headers: dict[str, str],
    *,
    log: Callable[[str], None],
    interval: float,
    max_attempts: int,
) -> str:
    """Poll the API until the task succeeds and return its file_id.

    Raises:
        NodeError: If the task fails, Minimax rejects the key, a result is missing, or polling gives up.
    """
    log(f"Starting to poll for task completion: {task_id}")

    fields = {"task_id": task_id}
    last_error: NodeError | None = None
    last_body: dict[str, Any] | None = None

    for attempt in range(max_attempts):
        # Recommended polling interval to avoid unnecessary server load
        time.sleep(interval)
        last_error = None

        try:
            # Query task status with task_id as query parameter
            response = requests.get(QUERY_URL, headers=headers, params={"task_id": task_id}, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            status_data = response.json()
        except requests.RequestException as e:
            log(f"Polling request failed (attempt {attempt + 1}): {e}")
            last_error = request_error(e, fields=fields)
            continue

        if not isinstance(status_data, dict):
            status_data = {}
        last_body = status_data

        status = status_data.get("status", "unknown")
        log(f"Polling attempt {attempt + 1}: Status = {status}")

        if status == "Fail":
            # Log full response for debugging
            log(f"Full API error response: {json.dumps(status_data, indent=2)}")
            raise task_failed_error(status_data, task_id=task_id)

        # Minimax reports other failures, such as a rejected key, inside an HTTP 200 body.
        if error := base_resp_error(status_data, fields=fields):
            log(f"Polling attempt {attempt + 1}: {error}")
            if status_data["base_resp"]["status_code"] in CREDENTIAL_REJECTED_CODES:
                raise error
            last_error = error
            continue

        if status == "Success":
            log("Task completed successfully")
            file_id = status_data.get("file_id")
            if not file_id:
                raise result_error("file ID", fields=fields, response=status_data)
            return file_id
        # Task still in progress (Processing, Pending, etc.)

    # The last poll failed, so report why instead of calling it a timeout.
    if last_error is not None:
        raise last_error

    # If we get here, we've exceeded max attempts
    raise NodeError(
        f"The Minimax video did not finish within {max_attempts * interval // 60:.0f} minutes. "
        "It may still be running in Minimax. Run the node again, or check the task ID in your Minimax account.",
        fields=fields,
        response=last_body,
    )


def retrieve_video_file(
    file_id: str,
    headers: dict[str, str],
    *,
    log: Callable[[str], None],
    outputs: dict[str, Any],
) -> str:
    """Return the download URL for a finished video, and store the response in `outputs["provider_response"]`."""
    log(f"Retrieving video file for file_id: {file_id}")

    fields = {"file_id": file_id}
    response_data = request_json(
        "GET", RETRIEVE_URL, headers=headers, params={"file_id": file_id}, timeout=DEFAULT_TIMEOUT, fields=fields
    )
    outputs["provider_response"] = response_data

    # Extract download URL
    file_info = response_data.get("file")
    download_url = file_info.get("download_url") if isinstance(file_info, dict) else None
    if not download_url:
        raise result_error("download URL", fields=fields, response=response_data)

    log(f"Retrieved download URL: {download_url[:100]}...")
    return download_url
