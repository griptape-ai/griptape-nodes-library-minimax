from __future__ import annotations

import json as _json
import logging
from contextlib import suppress
from copy import deepcopy
from typing import Any

from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, DataNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from minimax_errors import API_KEY_URL, request_json, result_error
from minimax_video import poll_for_completion, retrieve_video_file

logger = logging.getLogger(__name__)

__all__ = ["MinimaxTextToVideo"]

# Define constants
PROMPT_TRUNCATE_LENGTH = 100
DEFAULT_TIMEOUT = 60
POLLING_INTERVAL = 10  # seconds (recommended by Minimax)
MAX_POLLING_ATTEMPTS = 60  # 10 minutes max (60 * 10s)

# Model options for Minimax text-to-video
MODEL_OPTIONS = ["MiniMax-Hailuo-02", "T2V-01-Director", "T2V-01"]

# Resolution options based on model
RESOLUTION_OPTIONS_HAILUO = ["512P", "768P", "1080P"]

RESOLUTION_OPTIONS_OTHER = ["720P"]

# Duration options
DURATION_OPTIONS = [6, 10]


class MinimaxTextToVideo(DataNode):
    """Generate videos using Minimax text-to-video API.

    This node uses the Minimax API to generate videos from text prompts.
    The process involves submitting a task, polling for completion, and retrieving results.

    Inputs:
        - prompt (str): Text description of the video to generate (up to 2000 characters)
        - model (str): Model to use for generation (MiniMax-Hailuo-02, T2V-01-Director, T2V-01)
        - duration (int): Video length in seconds (6 or 10, model dependent)
        - resolution (str): Video resolution (512P/768P/1080P for Hailuo-02, 720P for others)
        - prompt_optimizer (bool): Automatically optimize prompt (default: True)
        - fast_pretreatment (bool): Reduce optimization time for Hailuo-02 (default: False)

    Outputs:
        - video_url (VideoUrlArtifact): Generated video as URL artifact
        - task_id (str): Task ID from the API
        - provider_response (dict): Full API response
    """

    SERVICE_NAME = "Minimax"
    API_KEY_NAME = "MINIMAX_API_KEY"
    API_BASE_URL = "https://api.minimax.io/v1/video_generation"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "Video Generation"
        self.description = "Generate videos using Minimax text-to-video API"
        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="minimax_video.mp4")
        self._output_file.add_parameter()

        # Core prompt parameter
        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str"],
                type="str",
                tooltip="Text description of the video to generate (up to 2000 characters). For MiniMax-Hailuo-02 and T2V-01-Director, use [command] syntax for camera movement: [Truck left/right], [Pan left/right], [Push in], [Pull out], [Pedestal up/down], [Tilt up/down], [Zoom in/out], [Shake], [Tracking shot], [Static shot]. Combine movements with commas inside brackets.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True,
                    "placeholder_text": "Describe the video you want to generate...",
                    "display_name": "Prompt",
                },
            )
        )

        # Model selection
        self.add_parameter(
            Parameter(
                name="model",
                input_types=["str"],
                type="str",
                default_value="MiniMax-Hailuo-02",
                tooltip="Model to use for video generation",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=MODEL_OPTIONS)},
                ui_options={"display_name": "Model"},
            )
        )

        # Duration selection
        duration_param = Parameter(
            name="duration",
            input_types=["int"],
            type="int",
            default_value=6,
            tooltip="Video duration in seconds (6 or 10). 10s only available for MiniMax-Hailuo-02 at 512P/768P",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=[6, 10])},
            ui_options={"display_name": "Duration (seconds)"},
        )
        self.add_parameter(duration_param)

        # Resolution selection
        resolution_param = Parameter(
            name="resolution",
            input_types=["str"],
            type="str",
            default_value="768P",
            tooltip="Video resolution. Options depend on model: MiniMax-Hailuo-02 supports 512P/768P/1080P, others support 720P",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=RESOLUTION_OPTIONS_HAILUO)},
            ui_options={"display_name": "Resolution"},
        )
        self.add_parameter(resolution_param)

        # Prompt optimizer
        self.add_parameter(
            Parameter(
                name="prompt_optimizer",
                input_types=["bool"],
                type="bool",
                default_value=True,
                tooltip="Automatically optimize the prompt for better results",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Prompt Optimizer"},
            )
        )

        # Fast pretreatment (only for MiniMax-Hailuo-02)
        self.add_parameter(
            Parameter(
                name="fast_pretreatment",
                input_types=["bool"],
                type="bool",
                default_value=False,
                tooltip="Reduce optimization time (only applies to MiniMax-Hailuo-02 with prompt_optimizer enabled)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "display_name": "Fast Pretreatment",
                    "hide": True,  # Hidden by default, shown for Hailuo-02
                },
            )
        )

        # OUTPUTS
        self.add_parameter(
            Parameter(
                name="video_url",
                output_type="VideoUrlArtifact",
                type="VideoUrlArtifact",
                tooltip="Generated video as URL artifact",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"is_full_width": True, "pulse_on_run": True},
            )
        )

        self.add_parameter(
            Parameter(
                name="task_id",
                output_type="str",
                tooltip="Task ID from the API",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"hide_property": True},
            )
        )

        self.add_parameter(
            Parameter(
                name="provider_response",
                output_type="dict",
                type="dict",
                tooltip="Full response from Minimax API",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"hide_property": True},
            )
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Handle parameter value changes and enforce model/duration constraints."""
        if parameter.name == "model":
            if value == "MiniMax-Hailuo-02":
                # Show fast_pretreatment for Hailuo-02
                self.show_parameter_by_name("fast_pretreatment")
                # Update resolution options based on current duration
                self._update_resolution_options_for_hailuo()
                # Reset resolution to default 768P
                self.set_parameter_value("resolution", "768P")
            else:
                # Hide fast_pretreatment for other models
                self.hide_parameter_by_name("fast_pretreatment")
                # Other models: only 6s duration and 720P resolution
                self.set_parameter_value("duration", 6)
                # Update resolution options to 720P only
                resolution_param = self.get_parameter_by_name("resolution")
                if resolution_param:
                    for child in resolution_param.children:
                        if hasattr(child, "choices"):
                            child.choices = RESOLUTION_OPTIONS_OTHER
                            break
                    self.set_parameter_value("resolution", "720P")

        if parameter.name == "duration":
            model = self.get_parameter_value("model")
            if model == "MiniMax-Hailuo-02":
                # Update resolution options based on duration
                self._update_resolution_options_for_hailuo()
            elif value == 10:
                # Other models don't support 10s, reset to 6s
                self.set_parameter_value("duration", 6)

        return super().after_value_set(parameter, value)

    def _update_resolution_options_for_hailuo(self) -> None:
        """Update resolution options for MiniMax-Hailuo-02 based on duration."""
        duration = self.get_parameter_value("duration")
        resolution_param = self.get_parameter_by_name("resolution")
        current_resolution = self.get_parameter_value("resolution")

        if resolution_param:
            for child in resolution_param.children:
                if hasattr(child, "choices"):
                    if duration == 10:
                        # 10s: only 512P and 768P available
                        child.choices = ["512P", "768P"]
                        # If current resolution is 1080P, change to 768P
                        if current_resolution == "1080P":
                            self.set_parameter_value("resolution", "768P")
                    else:
                        # 6s: all resolutions available (512P, 768P, 1080P)
                        child.choices = RESOLUTION_OPTIONS_HAILUO
                    break

    def _log(self, message: str) -> None:
        """Safe logging with exception suppression."""
        with suppress(Exception):
            logger.info(message)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate parameters before running the node."""
        # A missing key comes first and alone, so the user sees the Settings link without input noise.
        if not GriptapeNodes.SecretsManager().get_secret(self.API_KEY_NAME):
            return [NodeError.from_missing_secret(self.API_KEY_NAME, key_url=API_KEY_URL)]

        exceptions: list[Exception] = []

        # Validate prompt is provided
        prompt = self.get_parameter_value("prompt")
        if not prompt or not prompt.strip():
            exceptions.append(NodeError("'Prompt' is empty. Enter a prompt and run the node again."))
        elif len(prompt) > 2000:
            exceptions.append(NodeError(f"'Prompt' must be 2000 characters or less. It has {len(prompt)}."))

        # Validate duration and resolution compatibility
        model = self.get_parameter_value("model")
        duration = self.get_parameter_value("duration")
        resolution = self.get_parameter_value("resolution")

        if duration == 10:
            if model != "MiniMax-Hailuo-02":
                exceptions.append(
                    NodeError(
                        "A 10 second 'Duration' is only supported by MiniMax-Hailuo-02. Choose 6 seconds or change 'Model'."
                    )
                )
            elif resolution == "1080P":
                exceptions.append(
                    NodeError(
                        "A 10 second 'Duration' is not supported at 1080P. Choose 6 seconds or a lower 'Resolution'."
                    )
                )

        if model == "MiniMax-Hailuo-02":
            if resolution not in RESOLUTION_OPTIONS_HAILUO:
                exceptions.append(
                    NodeError(
                        f"'Resolution' {resolution} is not supported by MiniMax-Hailuo-02. Choose 512P, 768P or 1080P."
                    )
                )
        else:
            if resolution != "720P":
                exceptions.append(NodeError(f"{model} only supports 720P. Set 'Resolution' to 720P or change 'Model'."))

        return exceptions if exceptions else None

    def process(self) -> AsyncResult[None]:
        """Process the video generation request asynchronously."""
        yield lambda: self._process()

    def _process(self) -> None:
        """Main processing method for video generation."""
        try:
            # Get parameters
            params = self._get_parameters()

            # Validate API key
            api_key = self._validate_api_key()

            # Prepare headers
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            # Submit task and get task ID
            task_response = self._submit_task(params, headers)

            task_id = task_response.get("task_id")
            if not task_id:
                raise result_error("task ID", response=task_response)

            self.parameter_output_values["task_id"] = task_id
            self._log(f"Task submitted successfully: {task_id}")

            # Poll for completion and get file_id
            file_id = poll_for_completion(
                task_id, headers, log=self._log, interval=POLLING_INTERVAL, max_attempts=MAX_POLLING_ATTEMPTS
            )

            self._log(f"Task succeeded, File ID: {file_id}")

            # Retrieve video file from file_id
            video_url = retrieve_video_file(file_id, headers, log=self._log, outputs=self.parameter_output_values)

            # Save the video
            self._save_video_from_url(video_url)

        except Exception as e:
            self._log(f"Error in video generation: {e}")
            self._set_safe_defaults()
            raise

    def _get_parameters(self) -> dict[str, Any]:
        """Get and validate all parameters."""
        params = {
            "prompt": self.get_parameter_value("prompt") or "",
            "model": self.get_parameter_value("model") or "MiniMax-Hailuo-02",
            "duration": self.get_parameter_value("duration") or 6,
            "resolution": self.get_parameter_value("resolution") or "768P",
            "prompt_optimizer": self.get_parameter_value("prompt_optimizer")
            if self.get_parameter_value("prompt_optimizer") is not None
            else True,
            "fast_pretreatment": self.get_parameter_value("fast_pretreatment") or False,
        }

        return params

    def _validate_api_key(self) -> str:
        """Validate and return the API key."""
        api_key = GriptapeNodes.SecretsManager().get_secret(self.API_KEY_NAME)
        if not api_key:
            self._set_safe_defaults()
            raise NodeError.from_missing_secret(self.API_KEY_NAME, key_url=API_KEY_URL)
        return api_key

    def _submit_task(self, params: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        """Submit the video generation task to Minimax API."""
        payload = self._build_payload(params)

        self._log("Submitting video generation task to Minimax API")
        self._log_request(payload)

        response_data = request_json(
            "POST",
            self.API_BASE_URL,
            headers=headers,
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        self._log("Task submission successful")
        return response_data

    def _build_payload(self, params: dict[str, Any]) -> dict[str, Any]:
        """Build the API request payload."""
        payload = {
            "model": params["model"],
            "prompt": params["prompt"],
            "duration": params["duration"],
            "resolution": params["resolution"],
        }

        # Add optional parameters
        if params["prompt_optimizer"] is not None:
            payload["prompt_optimizer"] = params["prompt_optimizer"]

        # Only add fast_pretreatment for MiniMax-Hailuo-02 when prompt_optimizer is enabled
        if params["model"] == "MiniMax-Hailuo-02" and params["prompt_optimizer"] and params["fast_pretreatment"]:
            payload["fast_pretreatment"] = True

        return payload

    def _log_request(self, payload: dict[str, Any]) -> None:
        """Log the request payload with sensitive data sanitized."""
        with suppress(Exception):
            sanitized_payload = deepcopy(payload)
            # Truncate long prompts for logging
            if "prompt" in sanitized_payload and len(sanitized_payload["prompt"]) > PROMPT_TRUNCATE_LENGTH:
                sanitized_payload["prompt"] = sanitized_payload["prompt"][:PROMPT_TRUNCATE_LENGTH] + "..."

            self._log(f"Request payload: {_json.dumps(sanitized_payload, indent=2)}")

    def _save_video_from_url(self, video_url: str) -> None:
        """Save video from URL to project storage."""
        self._log("Processing generated video URL")

        # Download video bytes
        video_bytes = File(video_url).read_bytes()

        # Save to project storage
        saved = self._output_file.build_file().write_bytes(video_bytes)

        # Create VideoUrlArtifact
        self.parameter_output_values["video_url"] = VideoUrlArtifact(value=saved.location, name=saved.location)
        self._log(f"Saved video to project storage as {saved.location}")

    def _set_safe_defaults(self) -> None:
        """Set safe default values for all outputs."""
        self.parameter_output_values["video_url"] = None
        self.parameter_output_values["task_id"] = ""
        self.parameter_output_values["provider_response"] = None
