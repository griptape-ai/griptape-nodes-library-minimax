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
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from PIL import Image

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, DataNode
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

logger = logging.getLogger(__name__)

__all__ = ["MinimaxImageToImage"]

# Define constants
PROMPT_TRUNCATE_LENGTH = 100
DEFAULT_TIMEOUT = 60

# Model options for Minimax image-to-image
MODEL_OPTIONS = ["image-01", "image-01-live"]

# Aspect ratio options
ASPECT_RATIO_OPTIONS = [
    "1:1 (1024x1024)",
    "16:9 (1280x720)",
    "4:3 (1152x864)",
    "3:2 (1248x832)",
    "2:3 (832x1248)",
    "3:4 (864x1152)",
    "9:16 (720x1280)",
    "21:9 (1344x576)",
    "Use height and width",
]

# Subject type options
SUBJECT_TYPE_OPTIONS = ["character"]


class MinimaxImageToImage(DataNode):
    """Generate images using Minimax image-to-image API.
    
    This node uses the Minimax API to generate images based on a reference image and text prompt.
    Perfect for portrait generation and character consistency.
    
    Inputs:
        - prompt (str): Text description of the image (up to 1500 characters)
        - reference_image (ImageArtifact | ImageUrlArtifact): Reference image for subject
        - model (str): Model to use for generation (image-01, image-01-live)
        - subject_type (str): Subject type (currently only "character" supported)
        - aspect_ratio (str): Image dimensions or "Use height and width"
        - height (int): Custom height when using custom dimensions (512-2048, step 8)
        - width (int): Custom width when using custom dimensions (512-2048, step 8)
        - seed (int): Random seed for reproducibility (-1 for random)
        - num_images (int): Number of images to generate (1-9)
        - prompt_optimizer (bool): Automatically optimize prompt
        
    Outputs:
        - image (ImageUrlArtifact): Generated image (single result)
        - images (list[ImageUrlArtifact]): All generated images (when num_images > 1)
        - provider_response (dict): Full API response
    """
    
    SERVICE_NAME = "Minimax"
    API_KEY_NAME = "MINIMAX_API_KEY"
    API_BASE_URL = "https://api.minimax.io/v1/image_generation"
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "Image Generation"
        self.description = "Generate images using Minimax image-to-image API"
        
        # Core prompt parameter
        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str"],
                type="str",
                tooltip="Text description of the image to generate (up to 1500 characters)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True,
                    "placeholder_text": "Describe the image you want to generate...",
                    "display_name": "Prompt",
                },
            )
        )
        
        # Reference image parameter
        self.add_parameter(
            Parameter(
                name="reference_image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="Reference image for subject. For best results, use a single front-facing portrait photo. Supports JPG, JPEG, PNG (< 10MB).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "display_name": "Reference Image",
                    "clickable_file_browser": True,
                },
            )
        )
        
        # Model selection
        self.add_parameter(
            Parameter(
                name="model",
                input_types=["str"],
                type="str",
                default_value="image-01",
                tooltip="Model to use for image generation",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=MODEL_OPTIONS)},
                ui_options={"display_name": "Model"},
            )
        )
        
        # Subject type (hidden by default, only "character" supported currently)
        self.add_parameter(
            Parameter(
                name="subject_type",
                input_types=["str"],
                type="str",
                default_value="character",
                tooltip="Subject type for image-to-image generation (currently only 'character' is supported)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=SUBJECT_TYPE_OPTIONS)},
                ui_options={
                    "display_name": "Subject Type",
                    "hide": True,
                },
            )
        )
        
        # Aspect ratio selection
        self.add_parameter(
            Parameter(
                name="aspect_ratio",
                input_types=["str"],
                type="str",
                default_value="1:1 (1024x1024)",
                tooltip="Aspect ratio of the generated image. Select 'Use height and width' for custom dimensions.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=ASPECT_RATIO_OPTIONS)},
                ui_options={"display_name": "Aspect Ratio"},
            )
        )
        
        # Height parameter (hidden by default)
        self.add_parameter(
            Parameter(
                name="height",
                input_types=["int"],
                type="int",
                default_value=1024,
                tooltip="Height of generated image in pixels (512-2048, must be multiple of 8). Only effective when model is image-01.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Slider(min_val=512, max_val=2048)},
                ui_options={
                    "display_name": "Height",
                    "hide": True,
                },
            )
        )
        
        # Width parameter (hidden by default)
        self.add_parameter(
            Parameter(
                name="width",
                input_types=["int"],
                type="int",
                default_value=1024,
                tooltip="Width of generated image in pixels (512-2048, must be multiple of 8). Only effective when model is image-01.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Slider(min_val=512, max_val=2048)},
                ui_options={
                    "display_name": "Width",
                    "hide": True,
                },
            )
        )
        
        # Seed parameter
        self.add_parameter(
            Parameter(
                name="seed",
                input_types=["int"],
                type="int",
                default_value=-1,
                tooltip="Random seed for reproducibility. Use -1 for random, or specify a number to reproduce results.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Seed"},
            )
        )
        
        # Number of images parameter (hidden)
        self.add_parameter(
            Parameter(
                name="num_images",
                input_types=["int"],
                type="int",
                default_value=1,
                tooltip="Number of images to generate (1-9)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Slider(min_val=1, max_val=9)},
                ui_options={
                    "display_name": "Number of Images",
                    "hide": True,
                },
            )
        )
        
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
        
        # OUTPUTS
        self.add_parameter(
            Parameter(
                name="image",
                output_type="ImageUrlArtifact",
                type="ImageUrlArtifact",
                tooltip="Generated image",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"is_full_width": True, "pulse_on_run": True},
            )
        )
        
        # Multiple images output (hidden by default)
        self.add_parameter(
            Parameter(
                name="images",
                output_type="list[ImageUrlArtifact]",
                type="list",
                tooltip="All generated images (when num_images > 1)",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={
                    "is_full_width": True,
                    "hide": True,
                },
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
        """Handle parameter value changes for dynamic UI updates."""
        if parameter.name == "aspect_ratio":
            if value == "Use height and width":
                self.show_parameter_by_name("height")
                self.show_parameter_by_name("width")
            else:
                self.hide_parameter_by_name("height")
                self.hide_parameter_by_name("width")
        
        if parameter.name == "num_images":
            if value > 1:
                self.show_parameter_by_name("images")
            else:
                self.hide_parameter_by_name("images")
        
        return super().after_value_set(parameter, value)

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
        elif len(prompt) > 1500:
            exceptions.append(ValueError(f"{self.name}: Prompt must be 1500 characters or less (current: {len(prompt)} characters)"))
        
        # Validate reference image is provided
        reference_image = self.get_parameter_value("reference_image")
        if not reference_image:
            exceptions.append(ValueError(f"{self.name}: Reference image is required"))
        else:
            # Validate image format and size
            validation_errors = self._validate_reference_image(reference_image)
            if validation_errors:
                exceptions.extend(validation_errors)
        
        # Validate height and width if using custom dimensions
        aspect_ratio = self.get_parameter_value("aspect_ratio")
        if aspect_ratio == "Use height and width":
            height = self.get_parameter_value("height")
            width = self.get_parameter_value("width")
            
            if height < 512 or height > 2048:
                exceptions.append(ValueError(f"{self.name}: Height must be between 512 and 2048 (current: {height})"))
            elif height % 8 != 0:
                exceptions.append(ValueError(f"{self.name}: Height must be a multiple of 8 (current: {height})"))
            
            if width < 512 or width > 2048:
                exceptions.append(ValueError(f"{self.name}: Width must be between 512 and 2048 (current: {width})"))
            elif width % 8 != 0:
                exceptions.append(ValueError(f"{self.name}: Width must be a multiple of 8 (current: {width})"))
            
            # Only image-01 supports custom dimensions
            model = self.get_parameter_value("model")
            if model != "image-01":
                exceptions.append(ValueError(f"{self.name}: Custom height and width only supported by image-01 model (current model: {model})"))
        
        return exceptions if exceptions else None

    def _validate_reference_image(self, image_artifact: ImageArtifact | ImageUrlArtifact) -> list[Exception]:
        """Validate reference image format and size."""
        exceptions = []
        
        try:
            # Get image bytes
            if isinstance(image_artifact, ImageArtifact):
                if hasattr(image_artifact, 'base64') and image_artifact.base64:
                    image_bytes = base64.b64decode(image_artifact.base64.split(',')[1] if ',' in image_artifact.base64 else image_artifact.base64)
                elif hasattr(image_artifact, 'value') and hasattr(image_artifact.value, 'read'):
                    image_artifact.value.seek(0)
                    image_bytes = image_artifact.value.read()
                    image_artifact.value.seek(0)
                else:
                    return exceptions  # Skip validation if we can't get bytes
            elif isinstance(image_artifact, ImageUrlArtifact):
                # For URL artifacts, we'll skip byte-level validation
                # Just check if it's a localhost URL (which we'll convert to base64)
                if hasattr(image_artifact, 'value') and isinstance(image_artifact.value, str):
                    if 'localhost' in image_artifact.value or '127.0.0.1' in image_artifact.value:
                        # Will be converted to base64, validation will happen then
                        return exceptions
                return exceptions  # Skip validation for public URLs
            else:
                return exceptions
            
            # Check file size (< 10MB)
            if len(image_bytes) > 10 * 1024 * 1024:
                exceptions.append(ValueError(f"{self.name}: Reference image must be less than 10MB (current: {len(image_bytes) / (1024 * 1024):.1f}MB)"))
            
            # Check format (JPG, JPEG, PNG)
            try:
                img = Image.open(BytesIO(image_bytes))
                if img.format not in ['JPEG', 'PNG']:
                    exceptions.append(ValueError(f"{self.name}: Reference image must be JPG, JPEG, or PNG (current: {img.format})"))
            except Exception as e:
                exceptions.append(ValueError(f"{self.name}: Could not validate reference image format: {e}"))
        
        except Exception as e:
            self._log(f"Error validating reference image: {e}")
            # Don't fail validation if we can't check, let the API handle it
        
        return exceptions

    def process(self) -> AsyncResult[None]:
        """Process the image generation request asynchronously."""
        yield lambda: self._process()

    def _process(self) -> None:
        """Main processing method for image generation."""
        # Set safe defaults first
        self._set_safe_defaults()
        
        # Get parameters
        params = self._get_parameters()
        
        # Validate API key
        api_key = self._validate_api_key()
        
        # Prepare headers
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        # Get reference image data
        reference_image_data = self._get_image_data(params["reference_image"])
        if not reference_image_data:
            raise ValueError("Failed to process reference image")
        
        # Submit request
        try:
            response_data = self._submit_request(params, headers, reference_image_data)
            if not response_data:
                return
            
            self._handle_response(response_data, params)
            
        except Exception as e:
            self._log(f"Error in image generation: {e}")
            self._set_safe_defaults()
            raise

    def _get_parameters(self) -> dict[str, Any]:
        """Get and validate all parameters."""
        params = {
            "prompt": self.get_parameter_value("prompt") or "",
            "reference_image": self.get_parameter_value("reference_image"),
            "model": self.get_parameter_value("model") or "image-01",
            "subject_type": self.get_parameter_value("subject_type") or "character",
            "aspect_ratio": self.get_parameter_value("aspect_ratio") or "1:1 (1024x1024)",
            "height": self.get_parameter_value("height") or 1024,
            "width": self.get_parameter_value("width") or 1024,
            "seed": self.get_parameter_value("seed") if self.get_parameter_value("seed") is not None else -1,
            "num_images": self.get_parameter_value("num_images") or 1,
            "prompt_optimizer": self.get_parameter_value("prompt_optimizer") or False,
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

    def _get_image_data(self, image_artifact: ImageArtifact | ImageUrlArtifact) -> str:
        """
        Get image data as URL or base64 data URL.
        Handles localhost URLs by converting to base64.
        """
        try:
            # Handle ImageUrlArtifact
            if isinstance(image_artifact, ImageUrlArtifact):
                url = image_artifact.value
                
                # Check if it's a localhost URL
                if 'localhost' in url or '127.0.0.1' in url:
                    self._log("Detected localhost URL, converting to base64")
                    # Download and convert to base64
                    try:
                        response = requests.get(url, timeout=30)
                        response.raise_for_status()
                        image_bytes = response.content
                        
                        # Detect mime type
                        img = Image.open(BytesIO(image_bytes))
                        mime_type = f"image/{img.format.lower()}"
                        
                        # Convert to base64 data URL
                        b64_data = base64.b64encode(image_bytes).decode('utf-8')
                        return f"data:{mime_type};base64,{b64_data}"
                    except Exception as e:
                        self._log(f"Failed to convert localhost URL to base64: {e}")
                        raise ValueError(f"Failed to process localhost image URL: {e}")
                else:
                    # Public URL, return as-is
                    return url
            
            # Handle ImageArtifact
            elif isinstance(image_artifact, ImageArtifact):
                # Try using built-in base64 property first
                if hasattr(image_artifact, 'base64') and image_artifact.base64:
                    b64 = image_artifact.base64
                    mime_type = getattr(image_artifact, 'mime_type', 'image/jpeg')
                    
                    # Check if it's already a data URL
                    if b64.startswith('data:'):
                        return b64
                    else:
                        return f"data:{mime_type};base64,{b64}"
                
                # Fallback: manually extract and encode
                elif hasattr(image_artifact, 'value') and hasattr(image_artifact.value, 'read'):
                    image_artifact.value.seek(0)
                    image_bytes = image_artifact.value.read()
                    image_artifact.value.seek(0)
                    
                    # Detect mime type
                    img = Image.open(BytesIO(image_bytes))
                    mime_type = f"image/{img.format.lower()}"
                    
                    # Convert to base64
                    b64_data = base64.b64encode(image_bytes).decode('utf-8')
                    return f"data:{mime_type};base64,{b64_data}"
            
            raise ValueError(f"Unsupported image artifact type: {type(image_artifact)}")
            
        except Exception as e:
            self._log(f"Error processing image data: {e}")
            raise

    def _submit_request(self, params: dict[str, Any], headers: dict[str, str], reference_image_data: str) -> dict[str, Any] | None:
        """Submit the image generation request to Minimax API."""
        payload = self._build_payload(params, reference_image_data)
        
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
            self._log("Received response from API")
            
            # Check base_resp for errors
            base_resp = response_data.get("base_resp", {})
            status_code = base_resp.get("status_code")
            
            if status_code != 0:
                status_msg = base_resp.get("status_msg", "Unknown error")
                self._log(f"API error: {status_code} - {status_msg}")
                raise RuntimeError(f"Image generation failed: {status_msg} (code: {status_code})")
            
            return response_data
            
        except requests.RequestException as e:
            self._log(f"Request failed: {e}")
            msg = f"{self.name} request failed: {e}"
            raise RuntimeError(msg) from e

    def _build_payload(self, params: dict[str, Any], reference_image_data: str) -> dict[str, Any]:
        """Build the API request payload."""
        payload = {
            "prompt": params["prompt"],
            "model": params["model"],
            "subject_reference": [
                {
                    "type": params["subject_type"],
                    "image_file": reference_image_data,
                }
            ],
            "n": params["num_images"],
            "response_format": "url",
        }
        
        # Handle aspect ratio or custom dimensions
        if params["aspect_ratio"] == "Use height and width":
            # Round to nearest multiple of 8
            height = ((params["height"] + 4) // 8) * 8
            width = ((params["width"] + 4) // 8) * 8
            payload["height"] = height
            payload["width"] = width
        else:
            # Extract just the ratio part (e.g., "1:1" from "1:1 (1024x1024)")
            aspect_ratio = params["aspect_ratio"]
            if " " in aspect_ratio:
                aspect_ratio = aspect_ratio.split(" ")[0]
            payload["aspect_ratio"] = aspect_ratio
        
        # Add seed if not -1
        if params["seed"] != -1:
            payload["seed"] = params["seed"]
        
        # Add prompt_optimizer if enabled
        if params["prompt_optimizer"]:
            payload["prompt_optimizer"] = True
        
        return payload

    def _log_request(self, payload: dict[str, Any]) -> None:
        """Log the request payload with sensitive data sanitized."""
        with suppress(Exception):
            sanitized_payload = deepcopy(payload)
            if "prompt" in sanitized_payload and len(sanitized_payload["prompt"]) > PROMPT_TRUNCATE_LENGTH:
                sanitized_payload["prompt"] = sanitized_payload["prompt"][:PROMPT_TRUNCATE_LENGTH] + "..."
            if "subject_reference" in sanitized_payload and "image_file" in sanitized_payload["subject_reference"]:
                image_data = sanitized_payload["subject_reference"]["image_file"]
                if len(image_data) > PROMPT_TRUNCATE_LENGTH:
                    sanitized_payload["subject_reference"]["image_file"] = image_data[:PROMPT_TRUNCATE_LENGTH] + "..."
            
            self._log(f"Request payload: {_json.dumps(sanitized_payload, indent=2)}")

    def _handle_response(self, response_data: dict[str, Any], params: dict[str, Any]) -> None:
        """Handle the API response and save images."""
        self.parameter_output_values["provider_response"] = response_data
        
        # Extract image URLs
        data = response_data.get("data", {})
        image_urls = data.get("image_urls", [])
        
        if not image_urls:
            self._log("No image URLs in response")
            return
        
        self._log(f"Received {len(image_urls)} image URL(s)")
        
        # Save all images
        saved_artifacts = []
        for idx, image_url in enumerate(image_urls):
            artifact = self._save_image_from_url(image_url, idx)
            if artifact:
                saved_artifacts.append(artifact)
        
        # Set outputs
        if saved_artifacts:
            self.parameter_output_values["image"] = saved_artifacts[0]
            self.parameter_output_values["images"] = saved_artifacts

    def _save_image_from_url(self, image_url: str, index: int = 0) -> ImageUrlArtifact | None:
        """Save image from URL to static storage."""
        try:
            self._log(f"Downloading image {index + 1} from URL")
            
            # Download image bytes
            image_bytes = self._download_bytes_from_url(image_url)
            if image_bytes:
                # Generate filename with timestamp and index
                filename = f"minimax_image_{int(time.time())}_{index}.png"
                
                # Save to static storage
                from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
                
                static_files_manager = GriptapeNodes.StaticFilesManager()
                saved_url = static_files_manager.save_static_file(image_bytes, filename)
                
                self._log(f"Saved image {index + 1} to static storage as {filename}")
                return ImageUrlArtifact(value=saved_url, name=filename)
            else:
                # Fallback to original URL if download fails
                self._log(f"Using original image URL for image {index + 1} (download failed)")
                return ImageUrlArtifact(value=image_url)
                
        except Exception as e:
            self._log(f"Failed to save image {index + 1} from URL: {e}")
            # Fallback to original URL
            return ImageUrlArtifact(value=image_url)

    def _set_safe_defaults(self) -> None:
        """Set safe default values for all outputs."""
        self.parameter_output_values["image"] = None
        self.parameter_output_values["images"] = []
        self.parameter_output_values["provider_response"] = None

    @staticmethod
    def _download_bytes_from_url(url: str) -> bytes | None:
        """Download image bytes from URL."""
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.content
        except Exception:
            return None

