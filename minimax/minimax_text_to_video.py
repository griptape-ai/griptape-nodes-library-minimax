from __future__ import annotations

import json as _json
import logging
import time
from contextlib import suppress
from copy import deepcopy
from typing import Any

import requests
from griptape.artifacts import VideoUrlArtifact

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, DataNode
from griptape_nodes.traits.options import Options

logger = logging.getLogger(__name__)

__all__ = ["MinimaxTextToVideo"]

# Define constants
PROMPT_TRUNCATE_LENGTH = 100
DEFAULT_TIMEOUT = 60
POLLING_INTERVAL = 10  # seconds (recommended by Minimax)
MAX_POLLING_ATTEMPTS = 60  # 10 minutes max (60 * 10s)

# Model options for Minimax text-to-video
MODEL_OPTIONS = [
    "MiniMax-Hailuo-02",
    "T2V-01-Director",
    "T2V-01"
]

# Resolution options based on model
RESOLUTION_OPTIONS_HAILUO = [
    "512P",
    "768P",
    "1080P"
]

RESOLUTION_OPTIONS_OTHER = [
    "720P"
]

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
                    "hide": True  # Hidden by default, shown for Hailuo-02
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
                        if hasattr(child, 'choices'):
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
                if hasattr(child, 'choices'):
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
        exceptions = []
        
        # Validate prompt is provided
        prompt = self.get_parameter_value("prompt")
        if not prompt or not prompt.strip():
            exceptions.append(ValueError(f"{self.name}: Prompt is required"))
        elif len(prompt) > 2000:
            exceptions.append(ValueError(f"{self.name}: Prompt must be 2000 characters or less (current: {len(prompt)} characters)"))
        
        # Validate duration and resolution compatibility
        model = self.get_parameter_value("model")
        duration = self.get_parameter_value("duration")
        resolution = self.get_parameter_value("resolution")
        
        if duration == 10:
            if model != "MiniMax-Hailuo-02":
                exceptions.append(ValueError(f"{self.name}: 10s duration only supported by MiniMax-Hailuo-02 model"))
            elif resolution == "1080P":
                exceptions.append(ValueError(f"{self.name}: 10s duration not supported with 1080P resolution"))
        
        if model == "MiniMax-Hailuo-02":
            if resolution not in RESOLUTION_OPTIONS_HAILUO:
                exceptions.append(ValueError(f"{self.name}: Resolution {resolution} not supported for MiniMax-Hailuo-02"))
        else:
            if resolution != "720P":
                exceptions.append(ValueError(f"{self.name}: Only 720P resolution supported for {model}"))
        
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
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Submit task and get task ID
            task_response = self._submit_task(params, headers)
            if not task_response:
                return
                
            task_id = task_response.get("task_id")
            if not task_id:
                self._log("No task_id in response")
                self._set_safe_defaults()
                return
                
            self.parameter_output_values["task_id"] = task_id
            self._log(f"Task submitted successfully: {task_id}")
            
            # Poll for completion and get file_id
            file_id = self._poll_for_completion(task_id, headers)
            if not file_id:
                return
            
            self._log(f"Task succeeded, File ID: {file_id}")
            
            # Retrieve video file from file_id
            video_url = self._retrieve_video_file(file_id, headers)
            if not video_url:
                return
                
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
            "prompt_optimizer": self.get_parameter_value("prompt_optimizer") if self.get_parameter_value("prompt_optimizer") is not None else True,
            "fast_pretreatment": self.get_parameter_value("fast_pretreatment") or False,
        }
        
        return params

    def _validate_api_key(self) -> str:
        """Validate and return the API key."""
        from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
        
        api_key = GriptapeNodes.SecretsManager().get_secret(self.API_KEY_NAME)
        if not api_key:
            self._set_safe_defaults()
            msg = f"{self.name} is missing {self.API_KEY_NAME}. Ensure it's set in the environment/config."
            raise ValueError(msg)
        return api_key

    def _submit_task(self, params: dict[str, Any], headers: dict[str, str]) -> dict[str, Any] | None:
        """Submit the video generation task to Minimax API."""
        payload = self._build_payload(params)
        
        self._log("Submitting video generation task to Minimax API")
        self._log_request(payload)
        
        try:
            response = requests.post(
                self.API_BASE_URL,
                json=payload,
                headers=headers,
                timeout=DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            
            response_data = response.json()
            self._log("Task submission successful")
            return response_data
            
        except requests.RequestException as e:
            self._log(f"Task submission failed: {e}")
            msg = f"{self.name} task submission failed: {e}"
            raise RuntimeError(msg) from e
        except Exception as e:
            self._log(f"Request failed: {e}")
            msg = f"{self.name} request failed: {e}"
            raise RuntimeError(msg) from e

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

    def _poll_for_completion(self, task_id: str, headers: dict[str, str]) -> str | None:
        """Poll the API for task completion and return file_id."""
        self._log(f"Starting to poll for task completion: {task_id}")
        
        query_url = "https://api.minimax.io/v1/query/video_generation"
        
        for attempt in range(MAX_POLLING_ATTEMPTS):
            # Recommended polling interval to avoid unnecessary server load
            time.sleep(POLLING_INTERVAL)
            
            try:
                # Query task status with task_id as query parameter
                response = requests.get(
                    query_url, 
                    headers=headers, 
                    params={"task_id": task_id},
                    timeout=DEFAULT_TIMEOUT
                )
                response.raise_for_status()
                
                status_data = response.json()
                status = status_data.get("status", "unknown")
                
                self._log(f"Polling attempt {attempt + 1}: Status = {status}")
                
                if status == "Success":
                    self._log("Task completed successfully")
                    file_id = status_data.get("file_id")
                    if not file_id:
                        raise RuntimeError("Success response missing file_id")
                    return file_id
                elif status == "Fail":
                    error_msg = status_data.get("error_message", "Unknown error")
                    self._log(f"Task failed: {error_msg}")
                    raise RuntimeError(f"Video generation failed: {error_msg}")
                else:
                    # Task still in progress (Processing, Pending, etc.)
                    continue
                    
            except requests.RequestException as e:
                self._log(f"Polling request failed (attempt {attempt + 1}): {e}")
                if attempt < MAX_POLLING_ATTEMPTS - 1:
                    continue
                else:
                    raise RuntimeError(f"Polling failed after {MAX_POLLING_ATTEMPTS} attempts: {e}")
        
        # If we get here, we've exceeded max attempts
        raise RuntimeError(f"Task did not complete within {MAX_POLLING_ATTEMPTS * POLLING_INTERVAL} seconds")

    def _retrieve_video_file(self, file_id: str, headers: dict[str, str]) -> str | None:
        """Retrieve video download URL from file_id."""
        try:
            self._log(f"Retrieving video file for file_id: {file_id}")
            
            retrieve_url = "https://api.minimax.io/v1/files/retrieve"
            response = requests.get(
                retrieve_url,
                headers=headers,
                params={"file_id": file_id},
                timeout=DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            
            response_data = response.json()
            self.parameter_output_values["provider_response"] = response_data
            
            # Extract download URL
            download_url = response_data.get("file", {}).get("download_url")
            if not download_url:
                self._log("No download_url found in file retrieval response")
                return None
                
            self._log(f"Retrieved download URL: {download_url[:100]}...")
            return download_url
            
        except requests.RequestException as e:
            self._log(f"File retrieval failed: {e}")
            raise RuntimeError(f"Failed to retrieve video file: {e}")

    def _save_video_from_url(self, video_url: str) -> None:
        """Save video from URL to static storage."""
        try:
            self._log("Processing generated video URL")
            
            # Download video bytes
            video_bytes = self._download_bytes_from_url(video_url)
            if video_bytes:
                # Generate filename with timestamp
                filename = f"minimax_video_{int(time.time())}.mp4"
                
                # Save to static storage
                from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
                
                static_files_manager = GriptapeNodes.StaticFilesManager()
                saved_url = static_files_manager.save_static_file(video_bytes, filename)
                
                # Create VideoUrlArtifact
                self.parameter_output_values["video_url"] = VideoUrlArtifact(
                    value=saved_url, 
                    name=filename
                )
                self._log(f"Saved video to static storage as {filename}")
            else:
                # Fallback to original URL if download fails
                self.parameter_output_values["video_url"] = VideoUrlArtifact(value=video_url)
                self._log("Using original video URL (download failed)")
                
        except Exception as e:
            self._log(f"Failed to save video from URL: {e}")
            # Fallback to original URL
            self.parameter_output_values["video_url"] = VideoUrlArtifact(value=video_url)

    def _set_safe_defaults(self) -> None:
        """Set safe default values for all outputs."""
        self.parameter_output_values["video_url"] = None
        self.parameter_output_values["task_id"] = ""
        self.parameter_output_values["provider_response"] = None

    @staticmethod
    def _download_bytes_from_url(url: str) -> bytes | None:
        """Download video bytes from URL."""
        try:
            resp = requests.get(url, timeout=120)  # Longer timeout for videos
            resp.raise_for_status()
            return resp.content
        except Exception:
            return None
