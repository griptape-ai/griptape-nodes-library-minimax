from __future__ import annotations

import base64
import json as _json
import logging
import time
from contextlib import suppress
from copy import deepcopy
from io import BytesIO
from typing import Any

import requests
from griptape.artifacts import ImageArtifact, ImageUrlArtifact, VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, DataNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from minimax_errors import (
    API_KEY_ENV_VAR,
    API_KEY_URL,
    CREDENTIAL_REJECTED_CODES,
    PROVIDER,
    base_resp_error,
    request_json,
    result_error,
    task_failed_error,
)
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = ["MinimaxFirstLastFrameToVideo"]

# Define constants
PROMPT_TRUNCATE_LENGTH = 100
DEFAULT_TIMEOUT = 60
POLLING_INTERVAL = 10  # seconds (recommended by Minimax)
MAX_POLLING_ATTEMPTS = 60  # 10 minutes max (60 * 10s)

# Model options for Minimax first-last-frame-to-video
MODEL_OPTIONS = ["MiniMax-Hailuo-02"]

# Resolution options
RESOLUTION_OPTIONS = ["768P", "1080P"]

# Duration options
DURATION_OPTIONS = [6, 10]


class MinimaxFirstLastFrameToVideo(DataNode):
    """Generate videos using Minimax first-frame-to-last-frame API.

    This node uses the Minimax API to generate videos between two key frames.
    The process involves submitting a task, polling for completion, and retrieving results.

    Inputs:
        - first_frame_image (ImageArtifact | ImageUrlArtifact): Starting frame image
        - last_frame_image (ImageArtifact | ImageUrlArtifact): Ending frame image
        - prompt (str): Text description of the video motion (up to 2000 characters)
        - model (str): Model to use for generation (MiniMax-Hailuo-02)
        - duration (int): Video length in seconds (6 or 10)
        - resolution (str): Video resolution (512P/768P/1080P)
        - prompt_optimizer (bool): Automatically optimize prompt (default: False)
        - fast_pretreatment (bool): Reduce optimization time (default: False)

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
        self.description = "Generate videos using Minimax first-frame-to-last-frame API"
        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="minimax_f2l.mp4")
        self._output_file.add_parameter()

        # First frame image parameter
        self.add_parameter(
            Parameter(
                name="first_frame_image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="Starting frame image for the video. Supports JPG, JPEG, PNG, WebP (< 20MB, short edge > 300px, aspect ratio 2:5 to 5:2).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "display_name": "First Frame Image",
                    "clickable_file_browser": True,
                },
            )
        )

        # Last frame image parameter
        self.add_parameter(
            Parameter(
                name="last_frame_image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="Ending frame image for the video. Supports JPG, JPEG, PNG, WebP (< 20MB, short edge > 300px, aspect ratio 2:5 to 5:2).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "display_name": "Last Frame Image",
                    "clickable_file_browser": True,
                },
            )
        )

        # Core prompt parameter
        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str"],
                type="str",
                tooltip="Text description of the video motion (up to 2000 characters). Use [command] syntax for camera movement: [Truck left/right], [Pan left/right], [Push in], [Pull out], [Pedestal up/down], [Tilt up/down], [Zoom in/out], [Shake], [Tracking shot], [Static shot]. Combine movements with commas inside brackets.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True,
                    "placeholder_text": "Describe the video motion between frames...",
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
                tooltip="Model to use for video generation (currently only MiniMax-Hailuo-02 supports first-last frame)",
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
            tooltip="Video duration in seconds (6 or 10). 10s only available at 512P/768P",
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
            tooltip="Video resolution (512P/768P/1080P)",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=RESOLUTION_OPTIONS)},
            ui_options={"display_name": "Resolution"},
        )
        self.add_parameter(resolution_param)

        # Prompt optimizer
        self.add_parameter(
            Parameter(
                name="prompt_optimizer",
                input_types=["bool"],
                type="bool",
                default_value=False,
                tooltip="Automatically optimize the prompt for better results",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Prompt Optimizer"},
            )
        )

        # Fast pretreatment
        self.add_parameter(
            Parameter(
                name="fast_pretreatment",
                input_types=["bool"],
                type="bool",
                default_value=False,
                tooltip="Reduce optimization time (only applies when prompt_optimizer is enabled)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Fast Pretreatment"},
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
        """Handle parameter value changes and enforce constraints."""
        if parameter.name == "resolution":
            # 1080P only supports 6s duration
            if value == "1080P":
                self.set_parameter_value("duration", 6)

        return super().after_value_set(parameter, value)

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

        # Validate both frame images are provided
        first_frame_image = self.get_parameter_value("first_frame_image")
        last_frame_image = self.get_parameter_value("last_frame_image")

        if not first_frame_image:
            exceptions.append(NodeError("Connect an image to 'First Frame Image'."))
        else:
            # Validate first frame image requirements
            image_validation_errors = self._validate_image(first_frame_image, "first_frame_image")
            if image_validation_errors:
                exceptions.extend(image_validation_errors)

        if not last_frame_image:
            exceptions.append(NodeError("Connect an image to 'Last Frame Image'."))
        else:
            # Validate last frame image requirements
            image_validation_errors = self._validate_image(last_frame_image, "last_frame_image")
            if image_validation_errors:
                exceptions.extend(image_validation_errors)

        # Validate prompt
        prompt = self.get_parameter_value("prompt")
        if prompt and len(prompt) > 2000:
            exceptions.append(NodeError(f"'Prompt' must be 2000 characters or less. It has {len(prompt)}."))

        # Validate duration and resolution compatibility
        duration = self.get_parameter_value("duration")
        resolution = self.get_parameter_value("resolution")

        if duration == 10 and resolution == "1080P":
            exceptions.append(
                NodeError("A 10 second 'Duration' is not supported at 1080P. Choose 6 seconds or a lower 'Resolution'.")
            )

        return exceptions if exceptions else None

    def _validate_image(self, image_artifact: ImageArtifact | ImageUrlArtifact, param_name: str) -> list[Exception]:
        """Validate image requirements: format, size, dimensions, aspect ratio."""
        exceptions = []
        label = param_name.replace("_", " ").title()

        try:
            # For ImageUrlArtifact, we can't easily validate without downloading
            # The API will validate it, so we'll skip detailed checks
            if isinstance(image_artifact, ImageUrlArtifact):
                return exceptions

            # For ImageArtifact, validate format, size, and dimensions
            if isinstance(image_artifact, ImageArtifact):
                # Get image bytes
                image_bytes = None
                if hasattr(image_artifact, "value") and hasattr(image_artifact.value, "read"):
                    image_artifact.value.seek(0)
                    image_bytes = image_artifact.value.read()
                    image_artifact.value.seek(0)  # Reset for later use
                elif hasattr(image_artifact, "data"):
                    if isinstance(image_artifact.data, bytes):
                        image_bytes = image_artifact.data
                    elif hasattr(image_artifact.data, "read"):
                        image_artifact.data.seek(0)
                        image_bytes = image_artifact.data.read()
                        image_artifact.data.seek(0)  # Reset for later use

                if not image_bytes:
                    return exceptions  # Can't validate without bytes

                # Validate size (< 20MB)
                size_mb = len(image_bytes) / (1024 * 1024)
                if size_mb >= 20:
                    exceptions.append(
                        NodeError(f"The image connected to '{label}' must be under 20MB. It is {size_mb:.1f}MB.")
                    )

                # Validate format and dimensions using PIL
                try:
                    img = Image.open(BytesIO(image_bytes))

                    # Validate format
                    if img.format not in ["JPEG", "PNG", "WEBP"]:
                        exceptions.append(
                            NodeError(f"The image connected to '{label}' must be JPG, PNG or WebP. It is {img.format}.")
                        )

                    # Validate dimensions
                    width, height = img.size
                    short_edge = min(width, height)

                    if short_edge <= 300:
                        exceptions.append(
                            NodeError(
                                f"The image connected to '{label}' needs a short edge over 300px. It is {short_edge}px."
                            )
                        )

                    # Validate aspect ratio (between 2:5 and 5:2)
                    aspect_ratio = width / height
                    min_ratio = 2 / 5  # 0.4
                    max_ratio = 5 / 2  # 2.5

                    if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                        exceptions.append(
                            NodeError(
                                f"The image connected to '{label}' must have an aspect ratio between 2:5 and 5:2. "
                                f"It is {width}x{height}."
                            )
                        )

                except ImportError:
                    self._log("PIL not available for image validation")
                except Exception as e:
                    self._log(f"Error validating {param_name}: {e}")

        except Exception as e:
            self._log(f"Error in {param_name} validation: {e}")

        return exceptions

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
            file_id = self._poll_for_completion(task_id, headers)

            self._log(f"Task succeeded, File ID: {file_id}")

            # Retrieve video file from file_id
            video_url = self._retrieve_video_file(file_id, headers)

            # Save the video
            self._save_video_from_url(video_url)

        except Exception as e:
            self._log(f"Error in video generation: {e}")
            self._set_safe_defaults()
            raise

    def _get_parameters(self) -> dict[str, Any]:
        """Get and validate all parameters."""
        params = {
            "first_frame_image": self.get_parameter_value("first_frame_image"),
            "last_frame_image": self.get_parameter_value("last_frame_image"),
            "prompt": self.get_parameter_value("prompt") or "",
            "model": self.get_parameter_value("model") or "MiniMax-Hailuo-02",
            "duration": self.get_parameter_value("duration") or 6,
            "resolution": self.get_parameter_value("resolution") or "768P",
            "prompt_optimizer": self.get_parameter_value("prompt_optimizer")
            if self.get_parameter_value("prompt_optimizer") is not None
            else False,
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
        """Submit the video generation request to Minimax API."""
        payload = self._build_payload(params)

        self._log("Submitting request to Minimax API")
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
            "duration": params["duration"],
            "resolution": params["resolution"],
        }

        # Add prompt if provided
        if params["prompt"]:
            payload["prompt"] = params["prompt"]

        # Add first_frame_image (required)
        if params["first_frame_image"]:
            first_frame_data = self._get_image_data(params["first_frame_image"])
            if first_frame_data:
                payload["first_frame_image"] = first_frame_data
                self._log(f"Including first_frame_image for model {params['model']} at {params['resolution']}")

        # Add last_frame_image (required)
        if params["last_frame_image"]:
            last_frame_data = self._get_image_data(params["last_frame_image"])
            if last_frame_data:
                payload["last_frame_image"] = last_frame_data
                self._log(f"Including last_frame_image for model {params['model']} at {params['resolution']}")

        # Add optional parameters
        if params["prompt_optimizer"] is not None:
            payload["prompt_optimizer"] = params["prompt_optimizer"]

        # Only add fast_pretreatment when prompt_optimizer is enabled
        if params["prompt_optimizer"] and params["fast_pretreatment"]:
            payload["fast_pretreatment"] = True

        return payload

    def _get_image_data(self, image_artifact: ImageArtifact | ImageUrlArtifact) -> str:
        """Convert ImageArtifact or ImageUrlArtifact to URL or base64 data URI."""
        # For ImageUrlArtifact, check if it's a public URL or localhost
        if isinstance(image_artifact, ImageUrlArtifact):
            url = image_artifact.value

            # If it's a localhost URL, download and convert to base64
            if url.startswith(("http://localhost", "http://127.0.0.1", "https://localhost", "https://127.0.0.1")):
                self._log(f"Converting localhost URL to base64: {url[:100]}...")
                try:
                    image_bytes = File(url).read_bytes()
                    mime_type = "image/jpeg"

                    # Detect and convert unsupported formats with PIL
                    try:
                        img = Image.open(BytesIO(image_bytes))

                        # Convert unsupported formats (MPO, etc.) to JPEG
                        if img.format not in ["JPEG", "PNG", "WEBP"]:
                            self._log(f"Converting {img.format} to JPEG for API compatibility")
                            # Convert to RGB if needed (for formats like MPO)
                            if img.mode not in ["RGB", "L"]:
                                img = img.convert("RGB")
                            # Save as JPEG to bytes
                            output = BytesIO()
                            img.save(output, format="JPEG", quality=95)
                            image_bytes = output.getvalue()
                            mime_type = "image/jpeg"
                        else:
                            format_to_mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
                            mime_type = format_to_mime.get(img.format, "image/jpeg")
                            self._log(f"Detected image format from downloaded image: {img.format} -> {mime_type}")
                    except Exception:
                        pass

                    base64_data = base64.b64encode(image_bytes).decode("utf-8")
                    return f"data:{mime_type};base64,{base64_data}"

                except Exception as e:
                    self._log(f"Failed to download localhost image: {e}")
                    raise ValueError(f"Failed to download image from localhost URL: {e}") from e

            # For public URLs, use directly
            self._log(f"Using public URL: {url[:100]}...")
            return url

        # If it's an ImageArtifact, convert to base64 data URI
        if isinstance(image_artifact, ImageArtifact):
            try:
                # Use the .base64 property if available (preferred method)
                if hasattr(image_artifact, "base64") and hasattr(image_artifact, "mime_type"):
                    base64_data = image_artifact.base64
                    mime_type = image_artifact.mime_type

                    # Check if base64 already has data URI prefix
                    if base64_data.startswith("data:"):
                        self._log("Using ImageArtifact.base64 (already has data URI)")
                        return base64_data

                    # Add data URI prefix
                    self._log(f"Using ImageArtifact.base64 with mime_type: {mime_type}")
                    return f"data:{mime_type};base64,{base64_data}"

                # Fallback: manually extract bytes and encode
                self._log("Falling back to manual base64 encoding")
                if hasattr(image_artifact, "value") and hasattr(image_artifact.value, "read"):
                    image_artifact.value.seek(0)
                    image_bytes = image_artifact.value.read()
                    image_artifact.value.seek(0)  # Reset for validation
                elif hasattr(image_artifact, "data"):
                    if isinstance(image_artifact.data, bytes):
                        image_bytes = image_artifact.data
                    elif hasattr(image_artifact.data, "read"):
                        image_artifact.data.seek(0)
                        image_bytes = image_artifact.data.read()
                        image_artifact.data.seek(0)  # Reset for validation
                    else:
                        raise ValueError("Unsupported ImageArtifact data format")
                else:
                    raise ValueError("Unsupported ImageArtifact format")

                # Detect image format and convert unsupported formats using PIL
                mime_type = "image/jpeg"  # Default
                try:
                    img = Image.open(BytesIO(image_bytes))

                    # Convert unsupported formats (MPO, etc.) to JPEG
                    if img.format not in ["JPEG", "PNG", "WEBP"]:
                        self._log(f"Converting {img.format} to JPEG for API compatibility")
                        # Convert to RGB if needed (for formats like MPO)
                        if img.mode not in ["RGB", "L"]:
                            img = img.convert("RGB")
                        # Save as JPEG to bytes
                        output = BytesIO()
                        img.save(output, format="JPEG", quality=95)
                        image_bytes = output.getvalue()
                        mime_type = "image/jpeg"
                    else:
                        format_to_mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
                        mime_type = format_to_mime.get(img.format, "image/jpeg")
                        self._log(f"Detected image format: {img.format} -> {mime_type}")
                except Exception as e:
                    self._log(f"Could not detect image format, using default jpeg: {e}")

                # Convert to base64 data URI with correct MIME type
                base64_data = base64.b64encode(image_bytes).decode("utf-8")
                return f"data:{mime_type};base64,{base64_data}"

            except Exception as e:
                self._log(f"Error converting ImageArtifact to base64: {e}")
                raise ValueError(f"Failed to process image artifact: {e}") from e

        raise ValueError("Unsupported artifact type for frame image")

    def _log_request(self, payload: dict[str, Any]) -> None:
        """Log the request payload with sensitive data sanitized."""
        with suppress(Exception):
            # Log payload sizes
            if "first_frame_image" in payload:
                first_frame_len = len(payload.get("first_frame_image", ""))
                self._log(f"first_frame_image data length: {first_frame_len} chars (~{first_frame_len / 1024:.1f}KB)")
            if "last_frame_image" in payload:
                last_frame_len = len(payload.get("last_frame_image", ""))
                self._log(f"last_frame_image data length: {last_frame_len} chars (~{last_frame_len / 1024:.1f}KB)")

            sanitized_payload = deepcopy(payload)
            # Truncate long prompts for logging
            if "prompt" in sanitized_payload and len(sanitized_payload["prompt"]) > PROMPT_TRUNCATE_LENGTH:
                sanitized_payload["prompt"] = sanitized_payload["prompt"][:PROMPT_TRUNCATE_LENGTH] + "..."
            # Truncate base64 image data
            if "first_frame_image" in sanitized_payload and sanitized_payload["first_frame_image"].startswith("data:"):
                parts = sanitized_payload["first_frame_image"].split(",", 1)
                header = parts[0] if parts else "data:image/"
                sanitized_payload["first_frame_image"] = f"{header},<base64 data redacted>"
            if "last_frame_image" in sanitized_payload and sanitized_payload["last_frame_image"].startswith("data:"):
                parts = sanitized_payload["last_frame_image"].split(",", 1)
                header = parts[0] if parts else "data:image/"
                sanitized_payload["last_frame_image"] = f"{header},<base64 data redacted>"

            self._log(f"Request payload: {_json.dumps(sanitized_payload, indent=2)}")

    def _poll_for_completion(self, task_id: str, headers: dict[str, str]) -> str:
        """Poll the API for task completion and return file_id."""
        self._log(f"Starting to poll for task completion: {task_id}")

        query_url = "https://api.minimax.io/v1/query/video_generation"
        fields = {"task_id": task_id}
        last_error: NodeError | None = None
        last_body: dict[str, Any] | None = None

        for attempt in range(MAX_POLLING_ATTEMPTS):
            # Recommended polling interval to avoid unnecessary server load
            time.sleep(POLLING_INTERVAL)
            last_error = None

            try:
                # Query task status with task_id as query parameter
                response = requests.get(
                    query_url, headers=headers, params={"task_id": task_id}, timeout=DEFAULT_TIMEOUT
                )
                response.raise_for_status()
                status_data = response.json()
            except requests.RequestException as e:
                self._log(f"Polling request failed (attempt {attempt + 1}): {e}")
                last_error = NodeError.from_request_exception(
                    e, provider=PROVIDER, secret_name=API_KEY_ENV_VAR, fields=fields
                )
                continue

            if not isinstance(status_data, dict):
                status_data = {}
            last_body = status_data

            status = status_data.get("status", "unknown")
            self._log(f"Polling attempt {attempt + 1}: Status = {status}")

            if status == "Fail":
                # Log full response for debugging
                self._log(f"Full API error response: {_json.dumps(status_data, indent=2)}")
                raise task_failed_error(status_data, task_id=task_id)

            # Minimax reports other failures, such as a rejected key, inside an HTTP 200 body.
            if error := base_resp_error(status_data, fields=fields):
                self._log(f"Polling attempt {attempt + 1}: {error}")
                if status_data["base_resp"]["status_code"] in CREDENTIAL_REJECTED_CODES:
                    raise error
                last_error = error
                continue

            if status == "Success":
                self._log("Task completed successfully")
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
            f"The Minimax video did not finish within {MAX_POLLING_ATTEMPTS * POLLING_INTERVAL // 60} minutes. "
            "It may still be running in Minimax. Run the node again, or check the task ID in your Minimax account.",
            fields=fields,
            response=last_body,
        )

    def _retrieve_video_file(self, file_id: str, headers: dict[str, str]) -> str:
        """Retrieve video download URL from file_id."""
        self._log(f"Retrieving video file for file_id: {file_id}")

        retrieve_url = "https://api.minimax.io/v1/files/retrieve"
        fields = {"file_id": file_id}
        response_data = request_json(
            "GET",
            retrieve_url,
            headers=headers,
            params={"file_id": file_id},
            timeout=DEFAULT_TIMEOUT,
            fields=fields,
        )
        self.parameter_output_values["provider_response"] = response_data

        # Extract download URL
        file_info = response_data.get("file")
        download_url = file_info.get("download_url") if isinstance(file_info, dict) else None
        if not download_url:
            raise result_error("download URL", fields=fields, response=response_data)

        self._log(f"Retrieved download URL: {download_url[:100]}...")
        return download_url

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
