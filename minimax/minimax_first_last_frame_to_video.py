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

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, DataNode
from griptape_nodes.traits.options import Options

logger = logging.getLogger(__name__)

__all__ = ["MinimaxFirstLastFrameToVideo"]

# Define constants
PROMPT_TRUNCATE_LENGTH = 100
DEFAULT_TIMEOUT = 60
POLLING_INTERVAL = 10  # seconds (recommended by Minimax)
MAX_POLLING_ATTEMPTS = 60  # 10 minutes max (60 * 10s)

# Model options for Minimax first-last-frame-to-video
MODEL_OPTIONS = [
    "MiniMax-Hailuo-02"
]

# Resolution options
RESOLUTION_OPTIONS = [
    "512P",
    "768P",
    "1080P"
]

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

    def _log(self, message: str) -> None:
        """Safe logging with exception suppression."""
        with suppress(Exception):
            logger.info(message)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate parameters before running the node."""
        exceptions = []
        
        # Validate both frame images are provided
        first_frame_image = self.get_parameter_value("first_frame_image")
        last_frame_image = self.get_parameter_value("last_frame_image")
        
        if not first_frame_image:
            exceptions.append(ValueError(f"{self.name}: First frame image is required"))
        else:
            # Validate first frame image requirements
            image_validation_errors = self._validate_image(first_frame_image, "first_frame_image")
            if image_validation_errors:
                exceptions.extend(image_validation_errors)
        
        if not last_frame_image:
            exceptions.append(ValueError(f"{self.name}: Last frame image is required"))
        else:
            # Validate last frame image requirements
            image_validation_errors = self._validate_image(last_frame_image, "last_frame_image")
            if image_validation_errors:
                exceptions.extend(image_validation_errors)
        
        # Validate prompt
        prompt = self.get_parameter_value("prompt")
        if prompt and len(prompt) > 2000:
            exceptions.append(ValueError(f"{self.name}: Prompt must be 2000 characters or less (current: {len(prompt)} characters)"))
        
        # Validate duration and resolution compatibility
        duration = self.get_parameter_value("duration")
        resolution = self.get_parameter_value("resolution")
        
        if duration == 10 and resolution == "1080P":
            exceptions.append(ValueError(f"{self.name}: 10s duration not supported with 1080P resolution"))
        
        return exceptions if exceptions else None
    
    def _validate_image(self, image_artifact: ImageArtifact | ImageUrlArtifact, param_name: str) -> list[Exception]:
        """Validate image requirements: format, size, dimensions, aspect ratio."""
        exceptions = []
        
        try:
            # For ImageUrlArtifact, we can't easily validate without downloading
            # The API will validate it, so we'll skip detailed checks
            if isinstance(image_artifact, ImageUrlArtifact):
                return exceptions
            
            # For ImageArtifact, validate format, size, and dimensions
            if isinstance(image_artifact, ImageArtifact):
                # Get image bytes
                image_bytes = None
                if hasattr(image_artifact, 'value') and hasattr(image_artifact.value, 'read'):
                    image_artifact.value.seek(0)
                    image_bytes = image_artifact.value.read()
                    image_artifact.value.seek(0)  # Reset for later use
                elif hasattr(image_artifact, 'data'):
                    if isinstance(image_artifact.data, bytes):
                        image_bytes = image_artifact.data
                    elif hasattr(image_artifact.data, 'read'):
                        image_artifact.data.seek(0)
                        image_bytes = image_artifact.data.read()
                        image_artifact.data.seek(0)  # Reset for later use
                
                if not image_bytes:
                    return exceptions  # Can't validate without bytes
                
                # Validate size (< 20MB)
                size_mb = len(image_bytes) / (1024 * 1024)
                if size_mb >= 20:
                    exceptions.append(ValueError(f"{self.name}: {param_name} size must be less than 20MB (current: {size_mb:.1f}MB)"))
                
                # Validate format and dimensions using PIL
                try:
                    from PIL import Image
                    from io import BytesIO
                    
                    img = Image.open(BytesIO(image_bytes))
                    
                    # Validate format
                    if img.format not in ['JPEG', 'PNG', 'WEBP']:
                        exceptions.append(ValueError(f"{self.name}: {param_name} format must be JPG, JPEG, PNG, or WebP (current: {img.format})"))
                    
                    # Validate dimensions
                    width, height = img.size
                    short_edge = min(width, height)
                    
                    if short_edge <= 300:
                        exceptions.append(ValueError(f"{self.name}: {param_name} short edge must be > 300px (current: {short_edge}px)"))
                    
                    # Validate aspect ratio (between 2:5 and 5:2)
                    aspect_ratio = width / height
                    min_ratio = 2 / 5  # 0.4
                    max_ratio = 5 / 2  # 2.5
                    
                    if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                        exceptions.append(ValueError(
                            f"{self.name}: {param_name} aspect ratio must be between 2:5 and 5:2 "
                            f"(current: {width}x{height} = {aspect_ratio:.2f})"
                        ))
                    
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
            "first_frame_image": self.get_parameter_value("first_frame_image"),
            "last_frame_image": self.get_parameter_value("last_frame_image"),
            "prompt": self.get_parameter_value("prompt") or "",
            "model": self.get_parameter_value("model") or "MiniMax-Hailuo-02",
            "duration": self.get_parameter_value("duration") or 6,
            "resolution": self.get_parameter_value("resolution") or "768P",
            "prompt_optimizer": self.get_parameter_value("prompt_optimizer") if self.get_parameter_value("prompt_optimizer") is not None else False,
            "fast_pretreatment": self.get_parameter_value("fast_pretreatment") or False,
        }
        
        return params

    def _validate_api_key(self) -> str:
        """Validate and return the API key."""
        api_key = self.get_config_value(service=self.SERVICE_NAME, value=self.API_KEY_NAME)
        if not api_key:
            self._set_safe_defaults()
            msg = f"{self.name} is missing {self.API_KEY_NAME}. Ensure it's set in the environment/config."
            raise ValueError(msg)
        return api_key

    def _submit_task(self, params: dict[str, Any], headers: dict[str, str]) -> dict[str, Any] | None:
        """Submit the video generation request to Minimax API."""
        payload = self._build_payload(params)
        
        self._log("Submitting request to Minimax API")
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
            self._log(f"Task submission response: {_json.dumps(response_data, indent=2)}")
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
            if url.startswith(('http://localhost', 'http://127.0.0.1', 'https://localhost', 'https://127.0.0.1')):
                self._log(f"Converting localhost URL to base64: {url[:100]}...")
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    image_bytes = response.content
                    
                    # Detect format from response
                    mime_type = response.headers.get('content-type', 'image/jpeg')
                    if not mime_type.startswith('image/'):
                        mime_type = 'image/jpeg'
                    
                    # Optionally detect with PIL for accuracy
                    try:
                        from PIL import Image
                        from io import BytesIO
                        
                        img = Image.open(BytesIO(image_bytes))
                        format_to_mime = {
                            'JPEG': 'image/jpeg',
                            'PNG': 'image/png',
                            'WEBP': 'image/webp'
                        }
                        detected_mime = format_to_mime.get(img.format)
                        if detected_mime:
                            mime_type = detected_mime
                            self._log(f"Detected image format from downloaded image: {img.format} -> {mime_type}")
                    except Exception:
                        pass
                    
                    base64_data = base64.b64encode(image_bytes).decode('utf-8')
                    return f"data:{mime_type};base64,{base64_data}"
                    
                except Exception as e:
                    self._log(f"Failed to download localhost image: {e}")
                    raise ValueError(f"Failed to download image from localhost URL: {e}")
            
            # For public URLs, use directly
            self._log(f"Using public URL: {url[:100]}...")
            return url
        
        # If it's an ImageArtifact, convert to base64 data URI
        if isinstance(image_artifact, ImageArtifact):
            try:
                # Use the .base64 property if available (preferred method)
                if hasattr(image_artifact, 'base64') and hasattr(image_artifact, 'mime_type'):
                    base64_data = image_artifact.base64
                    mime_type = image_artifact.mime_type
                    
                    # Check if base64 already has data URI prefix
                    if base64_data.startswith('data:'):
                        self._log(f"Using ImageArtifact.base64 (already has data URI)")
                        return base64_data
                    
                    # Add data URI prefix
                    self._log(f"Using ImageArtifact.base64 with mime_type: {mime_type}")
                    return f"data:{mime_type};base64,{base64_data}"
                
                # Fallback: manually extract bytes and encode
                self._log("Falling back to manual base64 encoding")
                if hasattr(image_artifact, 'value') and hasattr(image_artifact.value, 'read'):
                    image_artifact.value.seek(0)
                    image_bytes = image_artifact.value.read()
                    image_artifact.value.seek(0)  # Reset for validation
                elif hasattr(image_artifact, 'data'):
                    if isinstance(image_artifact.data, bytes):
                        image_bytes = image_artifact.data
                    elif hasattr(image_artifact.data, 'read'):
                        image_artifact.data.seek(0)
                        image_bytes = image_artifact.data.read()
                        image_artifact.data.seek(0)  # Reset for validation
                    else:
                        raise ValueError("Unsupported ImageArtifact data format")
                else:
                    raise ValueError("Unsupported ImageArtifact format")
                
                # Detect image format using PIL
                mime_type = "image/jpeg"  # Default
                try:
                    from PIL import Image
                    from io import BytesIO
                    
                    img = Image.open(BytesIO(image_bytes))
                    format_to_mime = {
                        'JPEG': 'image/jpeg',
                        'PNG': 'image/png',
                        'WEBP': 'image/webp'
                    }
                    mime_type = format_to_mime.get(img.format, 'image/jpeg')
                    self._log(f"Detected image format: {img.format} -> {mime_type}")
                except Exception as e:
                    self._log(f"Could not detect image format, using default jpeg: {e}")
                
                # Convert to base64 data URI with correct MIME type
                base64_data = base64.b64encode(image_bytes).decode('utf-8')
                return f"data:{mime_type};base64,{base64_data}"
                
            except Exception as e:
                self._log(f"Error converting ImageArtifact to base64: {e}")
                raise ValueError(f"Failed to process image artifact: {e}")
        
        raise ValueError("Unsupported artifact type for frame image")

    def _log_request(self, payload: dict[str, Any]) -> None:
        """Log the request payload with sensitive data sanitized."""
        with suppress(Exception):
            # Log payload sizes
            if "first_frame_image" in payload:
                first_frame_len = len(payload.get("first_frame_image", ""))
                self._log(f"first_frame_image data length: {first_frame_len} chars (~{first_frame_len/1024:.1f}KB)")
            if "last_frame_image" in payload:
                last_frame_len = len(payload.get("last_frame_image", ""))
                self._log(f"last_frame_image data length: {last_frame_len} chars (~{last_frame_len/1024:.1f}KB)")
            
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

    def _poll_for_completion(self, task_id: str, headers: dict[str, str]) -> str | None:
        """Poll the API for task completion and return file_id."""
        self._log(f"Starting to poll for task completion: {task_id}")
        
        query_url = "https://api.minimax.io/v1/query/video_generation"
        
        for attempt in range(MAX_POLLING_ATTEMPTS):
            time.sleep(POLLING_INTERVAL)
            
            try:
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
                    # Log full response for debugging
                    self._log(f"Full API error response: {_json.dumps(status_data, indent=2)}")
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
                filename = f"minimax_f2l_{int(time.time())}.mp4"
                
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

