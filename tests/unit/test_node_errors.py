"""Tests that the Minimax nodes report failures as `NodeError`.

Requests go through the real `requests` stack: only the adapter that would open the socket is
replaced, so `raise_for_status`, `.json()` and the exception types are the ones a live failure produces.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import requests
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from minimax_first_last_frame_to_video import MinimaxFirstLastFrameToVideo
from minimax_image_to_image import MinimaxImageToImage
from minimax_image_to_video import MinimaxImageToVideo
from minimax_music_generation import MinimaxMusicGeneration
from minimax_text_to_image import MinimaxTextToImage
from minimax_text_to_video import MinimaxTextToVideo

LIBRARY_DIR = Path(__file__).resolve().parents[2] / "minimax"
SETTINGS_LINK = "#settings-secrets?filter=MINIMAX_API_KEY"
IMAGE = ImageUrlArtifact("https://example.com/a.png")

REFUSED_BODY = {"base_resp": {"status_code": 1004, "status_msg": "login fail: Please carry the API secret key"}}
SUBMITTED_BODY = {"task_id": "task-1", "base_resp": {"status_code": 0, "status_msg": "success"}}
PROCESSING_BODY = {"task_id": "task-1", "status": "Processing", "base_resp": {"status_code": 0}}
FAILED_BODY = {
    "task_id": "task-1",
    "status": "Fail",
    "base_resp": {"status_code": 1026, "status_msg": "The prompt did not pass the content check"},
}

# Node class, inputs that pass validation.
NODES: dict[str, tuple[type, dict[str, Any]]] = {
    "text_to_image": (MinimaxTextToImage, {"prompt": "a cat"}),
    "image_to_image": (MinimaxImageToImage, {"prompt": "a cat", "reference_image": IMAGE}),
    "text_to_video": (MinimaxTextToVideo, {"prompt": "a cat"}),
    "image_to_video": (MinimaxImageToVideo, {"first_frame_image": IMAGE}),
    "first_last_frame": (MinimaxFirstLastFrameToVideo, {"first_frame_image": IMAGE, "last_frame_image": IMAGE}),
    "music": (MinimaxMusicGeneration, {"prompt": "an upbeat song", "lyrics": "la la la la la la"}),
}
VIDEO_NODES = ["text_to_video", "image_to_video", "first_last_frame"]

Route = Callable[[requests.PreparedRequest], tuple[int, Any]]


def _make(name: str) -> Any:
    cls, inputs = NODES[name]
    node = cls(name="node")
    for parameter, value in inputs.items():
        node.set_parameter_value(parameter, value)
    return node


@pytest.fixture
def secret() -> Iterator[None]:
    with patch.object(GriptapeNodes, "SecretsManager") as manager:
        manager.return_value.get_secret.return_value = "test-key"
        yield


@pytest.fixture
def no_secret() -> Iterator[None]:
    with patch.object(GriptapeNodes, "SecretsManager") as manager:
        manager.return_value.get_secret.return_value = None
        yield


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Callable[[Route], list[requests.PreparedRequest]]:
    """Install a route for every Minimax call, and return the list of requests it saw."""
    seen: list[requests.PreparedRequest] = []

    def install(route: Route) -> list[requests.PreparedRequest]:
        def send(_adapter: Any, request: requests.PreparedRequest, **_kwargs: Any) -> requests.Response:
            seen.append(request)
            status, body = route(request)
            response = requests.Response()
            response.status_code = status
            response._content = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            response.url = request.url or ""
            response.request = request
            return response

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
        monkeypatch.setattr("time.sleep", lambda _seconds: None)
        return seen

    return install


def _post_then(poll: tuple[int, Any]) -> Route:
    return lambda request: (200, SUBMITTED_BODY) if request.method == "POST" else poll


@pytest.mark.parametrize("name", list(NODES))
def test_missing_key_is_the_only_validation_error(name: str, no_secret: None) -> None:
    node = _make(name)
    errors = node.validate_before_node_run()
    assert errors is not None
    assert len(errors) == 1
    error = errors[0]
    assert isinstance(error, NodeError)
    assert "MINIMAX_API_KEY is not set" in str(error)
    assert error.links[0].url == SETTINGS_LINK


@pytest.mark.parametrize("name", list(NODES))
def test_missing_key_raises_at_run_time(name: str, no_secret: None) -> None:
    node = _make(name)
    with pytest.raises(NodeError, match="MINIMAX_API_KEY is not set"):
        node._process()


@pytest.mark.parametrize("name", list(NODES))
def test_refused_key_in_a_200_body_links_to_settings(name: str, secret: None, server: Any) -> None:
    server(lambda _request: (200, REFUSED_BODY))
    node = _make(name)
    with pytest.raises(NodeError) as excinfo:
        node._process()
    error = excinfo.value
    assert "Minimax refused the API key: login fail" in str(error)
    assert error.fields["error_code"] == 1004
    assert error.response == REFUSED_BODY
    assert error.links[0].url == SETTINGS_LINK


@pytest.mark.parametrize("name", list(NODES))
def test_http_401_links_to_settings(name: str, secret: None, server: Any) -> None:
    server(lambda _request: (401, {"error": {"message": "Unauthorized"}}))
    node = _make(name)
    with pytest.raises(NodeError, match="Minimax refused the API key") as excinfo:
        node._process()
    assert excinfo.value.fields["status_code"] == 401
    assert excinfo.value.links[0].url == SETTINGS_LINK


def test_provider_error_message_is_minimax_s_own(secret: None, server: Any) -> None:
    body = {"base_resp": {"status_code": 1026, "status_msg": "The prompt did not pass the content check"}}
    server(lambda _request: (200, body))
    with pytest.raises(NodeError) as excinfo:
        _make("text_to_image")._process()
    assert str(excinfo.value) == "The prompt did not pass the content check"
    assert excinfo.value.fields == {"error_code": 1026, "status_code": 200}
    assert excinfo.value.response == body


def test_gateway_failure_with_html_body_keeps_the_status(secret: None, server: Any) -> None:
    server(lambda _request: (502, "<html>Bad Gateway</html>"))
    with pytest.raises(NodeError) as excinfo:
        _make("text_to_image")._process()
    assert excinfo.value.fields["status_code"] == 502


@pytest.mark.parametrize("name", VIDEO_NODES)
def test_failed_task_carries_the_task_id_and_reason(name: str, secret: None, server: Any) -> None:
    server(_post_then((200, FAILED_BODY)))
    with pytest.raises(NodeError) as excinfo:
        _make(name)._process()
    assert str(excinfo.value) == "Minimax could not finish the video: The prompt did not pass the content check."
    assert excinfo.value.fields == {"task_id": "task-1", "status": "Fail", "error_code": 1026}
    assert excinfo.value.response == FAILED_BODY


@pytest.mark.parametrize("name", VIDEO_NODES)
def test_poll_that_keeps_returning_500_reports_the_failure(name: str, secret: None, server: Any) -> None:
    server(_post_then((500, {"message": "Internal error"})))
    with pytest.raises(NodeError) as excinfo:
        _make(name)._process()
    assert excinfo.value.fields["status_code"] == 500
    assert excinfo.value.fields["task_id"] == "task-1"


@pytest.mark.parametrize("name", VIDEO_NODES)
def test_poll_with_a_rejected_key_stops_at_once(name: str, secret: None, server: Any) -> None:
    seen = server(_post_then((200, REFUSED_BODY)))
    with pytest.raises(NodeError, match="Minimax refused the API key"):
        _make(name)._process()
    assert [request.method for request in seen] == ["POST", "GET"]


@pytest.mark.parametrize("name", VIDEO_NODES)
def test_task_that_never_finishes_times_out(name: str, secret: None, server: Any) -> None:
    server(_post_then((200, PROCESSING_BODY)))
    with pytest.raises(NodeError, match="did not finish within 10 minutes") as excinfo:
        _make(name)._process()
    assert excinfo.value.fields == {"task_id": "task-1"}


@pytest.mark.parametrize("name", VIDEO_NODES)
def test_submit_without_a_task_id_raises(name: str, secret: None, server: Any) -> None:
    server(lambda _request: (200, {"base_resp": {"status_code": 0}}))
    with pytest.raises(NodeError, match="result has no task ID"):
        _make(name)._process()


@pytest.mark.parametrize("name", VIDEO_NODES)
def test_success_without_a_download_url_raises(name: str, secret: None, server: Any) -> None:
    def route(request: requests.PreparedRequest) -> tuple[int, Any]:
        if request.method == "POST":
            return 200, SUBMITTED_BODY
        if "query" in (request.url or ""):
            return 200, {"status": "Success", "file_id": "file-9", "base_resp": {"status_code": 0}}
        return 200, {"file": {}, "base_resp": {"status_code": 0}}

    server(route)
    with pytest.raises(NodeError, match="result has no download URL") as excinfo:
        _make(name)._process()
    assert excinfo.value.fields == {"file_id": "file-9"}


@pytest.mark.parametrize("name", ["text_to_image", "image_to_image"])
def test_image_response_without_images_raises(name: str, secret: None, server: Any) -> None:
    server(lambda _request: (200, {"data": {}, "base_resp": {"status_code": 0}}))
    with pytest.raises(NodeError, match="result has no images"):
        _make(name)._process()


def test_music_that_is_not_finished_raises(secret: None, server: Any) -> None:
    server(lambda _request: (200, {"data": {"status": 1}, "base_resp": {"status_code": 0}}))
    with pytest.raises(NodeError, match="did not return finished audio") as excinfo:
        _make("music")._process()
    assert excinfo.value.fields == {"status": "1"}


def test_input_problems_do_not_name_the_node(secret: None) -> None:
    node = MinimaxTextToImage(name="Pretty Name")
    errors = node.validate_before_node_run()
    assert errors is not None
    assert all(isinstance(error, NodeError) for error in errors)
    assert [str(error) for error in errors] == ["'Prompt' is empty. Enter a prompt and run the node again."]


@pytest.mark.parametrize("path", sorted(LIBRARY_DIR.glob("minimax_*.py")), ids=lambda path: path.name)
def test_node_files_load_standalone(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine loads each file with no parent package and the library folder on `sys.path`."""
    monkeypatch.syspath_prepend(str(LIBRARY_DIR))
    monkeypatch.setitem(sys.modules, "minimax", None)
    spec = importlib.util.spec_from_file_location(f"standalone_{path.stem}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
