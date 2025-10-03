from __future__ import annotations

import json as _json
import logging
import time
from contextlib import suppress
from copy import deepcopy
from typing import Any

import requests
from griptape.artifacts import ImageUrlArtifact

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, DataNode
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

logger = logging.getLogger(__name__)

__all__ = ["MinimaxTextToImage"]

# Define constants
PROMPT_TRUNCATE_LENGTH = 100
DEFAULT_TIMEOUT = 60

# Model options for Minimax text-to-image
MODEL_OPTIONS = [
    "image-01"
]

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
    "Use height and width"
]


class MinimaxTextToImage(DataNode):
    """Generate images using Minimax text-to-image API.
    
    This node uses the Minimax API to generate images from text prompts.
    Supports various models, aspect ratios, custom dimensions, and multiple image generation.
    
    Inputs:
        - prompt (str): Text description of the image to generate
        - model (str): Model to use for generation
        - aspect_ratio (str): Aspect ratio of generated image or "Use height and width"
        - seed (int): Random seed for reproducible results
        - num_images (int): Number of images to generate (1-9, hidden parameter)
        - prompt_optimizer (bool): Enable prompt optimization for better quality
        - height (int): Custom height in pixels (512-2048, step of 8, hidden by default)
        - width (int): Custom width in pixels (512-2048, step of 8, hidden by default)
        
    Outputs:
        - image_url (ImageUrlArtifact): Primary generated image as URL artifact
        - images (list[ImageUrlArtifact]): List of all generated images (shown when num_images > 1)
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Full API response
    """
    
    SERVICE_NAME = "Minimax"
    API_KEY_NAME = "MINIMAX_API_KEY"
    API_BASE_URL = "https://api.minimax.io/v1/image_generation"
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "Image Generation"
        self.description = "Generate images using Minimax text-to-image API"
        
        # Core prompt parameter
        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str"],
                type="str",
                tooltip="Text description of the image to generate",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True,
                    "placeholder_text": "Describe the image you want to generate...",
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
                default_value="image-01",
                tooltip="Model to use for image generation",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=MODEL_OPTIONS)},
                ui_options={"display_name": "Model"},
            )
        )
        
        # Aspect ratio selection
        self.add_parameter(
            Parameter(
                name="aspect_ratio",
                input_types=["str"],
                type="str",
                default_value="1:1 (1024x1024)",
                tooltip="Aspect ratio of the generated image",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=ASPECT_RATIO_OPTIONS)},
                ui_options={"display_name": "Aspect Ratio"},
            )
        )
        
        
        # Seed parameter for reproducibility
        self.add_parameter(
            Parameter(
                name="seed",
                input_types=["int"],
                type="int",
                default_value=-1,
                tooltip="Random seed for reproducible results (-1 for random)",
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
                    "hide": False
                },
            )
        )
        
        # Prompt optimizer parameter
        self.add_parameter(
            Parameter(
                name="prompt_optimizer",
                input_types=["bool"],
                type="bool",
                default_value=False,
                tooltip="Enable prompt optimization to improve generation quality",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Prompt Optimizer"},
            )
        )
        
        # Height parameter (for custom dimensions)
        height_param = Parameter(
            name="height",
            input_types=["int"],
            type="int",
            default_value=1024,
            tooltip="Image height in pixels (512-2048, step of 8)",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={
                "display_name": "Height",
                "hide": True,  # Hidden by default, shown when "Use height and width" is selected
                "step": 8
            },  
        )
        height_param.add_trait(Slider(min_val=512, max_val=2048))
        self.add_parameter(height_param)
        
        # Width parameter (for custom dimensions)
        width_param = Parameter(
            name="width",
            input_types=["int"],
            type="int",
            default_value=1024,
            tooltip="Image width in pixels (512-2048, step of 8)",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={
                "display_name": "Width",
                "hide": True,  # Hidden by default, shown when "Use height and width" is selected
                "step": 8
            },
        )
        width_param.add_trait(Slider(min_val=512, max_val=2048))
        self.add_parameter(width_param)
        
        # OUTPUTS
        self.add_parameter(
            Parameter(
                name="image_url",
                output_type="ImageUrlArtifact",
                type="ImageUrlArtifact",
                tooltip="Generated image as URL artifact",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"is_full_width": True, "pulse_on_run": True},
            )
        )
        
        self.add_parameter(
            Parameter(
                name="generation_id",
                output_type="str",
                tooltip="Generation ID from the API",
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
        
        # Images list output (for multiple images when num_images > 1)
        self.add_parameter(
            Parameter(
                name="images",
                output_type="list[ImageUrlArtifact]",
                type="list[ImageUrlArtifact]",
                tooltip="List of generated images (when multiple images are requested)",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={
                    "is_full_width": True, 
                    "pulse_on_run": True,
                    "hide": True  # Hidden by default, shown when num_images > 1
                },
            )
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Handle parameter value changes."""
        if parameter.name == "aspect_ratio":
            if value == "Use height and width":
                # Show height and width parameters
                self.show_parameter_by_name("height")
                self.show_parameter_by_name("width")
            else:
                # Hide height and width parameters
                self.hide_parameter_by_name("height")
                self.hide_parameter_by_name("width")
        
        if parameter.name == "num_images":
            if value > 1:
                # Show images list output
                self.show_parameter_by_name("images")
            else:
                # Hide images list output
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
        
        # Validate height and width if using custom dimensions
        aspect_ratio = self.get_parameter_value("aspect_ratio")
        if aspect_ratio == "Use height and width":
            height = self.get_parameter_value("height")
            width = self.get_parameter_value("width")
            
            if height < 512 or height > 2048:
                exceptions.append(ValueError(f"{self.name}: Height must be between 512 and 2048 pixels (current: {height})"))
            elif height % 8 != 0:
                exceptions.append(ValueError(f"{self.name}: Height must be a multiple of 8 (current: {height})"))
                
            if width < 512 or width > 2048:
                exceptions.append(ValueError(f"{self.name}: Width must be between 512 and 2048 pixels (current: {width})"))
            elif width % 8 != 0:
                exceptions.append(ValueError(f"{self.name}: Width must be a multiple of 8 (current: {width})"))
        
        return exceptions if exceptions else None

    def process(self) -> AsyncResult[None]:
        """Process the image generation request asynchronously."""
        yield lambda: self._process()

    def _process(self) -> None:
        """Main processing method for image generation."""
        # Get parameters
        params = self._get_parameters()
        
        # Validate API key
        api_key = self._validate_api_key()
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        self._log(f"Generating image with Minimax API using model: {params['model']}")
        
        try:
            # Submit request
            response = self._submit_request(params, headers)
            if response:
                self._handle_response(response)
            else:
                self._set_safe_defaults()
        except Exception as e:
            self._log(f"Error in image generation: {e}")
            self._set_safe_defaults()
            raise

    def _get_parameters(self) -> dict[str, Any]:
        """Get and validate all parameters."""
        params = {
            "prompt": self.get_parameter_value("prompt") or "",
            "model": self.get_parameter_value("model") or "image-01",
            "aspect_ratio": self.get_parameter_value("aspect_ratio") or "1:1 (1024x1024)",
            "seed": self.get_parameter_value("seed") or -1,
            "num_images": self.get_parameter_value("num_images") or 1,
            "prompt_optimizer": self.get_parameter_value("prompt_optimizer") or False,
            "height": self.get_parameter_value("height") or 1024,
            "width": self.get_parameter_value("width") or 1024,
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

    def _submit_request(self, params: dict[str, Any], headers: dict[str, str]) -> dict[str, Any] | None:
        """Submit the image generation request to Minimax API."""
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
            response_json = response.json()
            self._log("Request submitted successfully")
            return response_json
            
        except requests.exceptions.HTTPError as e:
            self._log(f"HTTP error: {e.response.status_code} - {e.response.text}")
            msg = f"{self.name} API error: {e.response.status_code}"
            raise RuntimeError(msg) from e
        except Exception as e:
            self._log(f"Request failed: {e}")
            msg = f"{self.name} request failed: {e}"
            raise RuntimeError(msg) from e

    def _build_payload(self, params: dict[str, Any]) -> dict[str, Any]:
        """Build the API request payload."""
        payload = {
            "prompt": params["prompt"],
            "model": params["model"],
            "n": params["num_images"],
        }
        
        # Handle aspect ratio vs height/width
        if params["aspect_ratio"] == "Use height and width":
            # Use custom height and width (ensure they're multiples of 8)
            height = params["height"]
            width = params["width"]
            
            # Round to nearest multiple of 8
            height = ((height + 4) // 8) * 8
            width = ((width + 4) // 8) * 8
            
            payload["height"] = height
            payload["width"] = width
        else:
            # Use aspect ratio - extract just the ratio part (before the space)
            aspect_ratio = params["aspect_ratio"]
            if " " in aspect_ratio:
                aspect_ratio = aspect_ratio.split(" ")[0]
            payload["aspect_ratio"] = aspect_ratio
        
        # Add optional parameters
        if params["seed"] != -1:
            payload["seed"] = params["seed"]
            
        if params["prompt_optimizer"]:
            payload["prompt_optimizer"] = True
        
        return payload

    def _log_request(self, payload: dict[str, Any]) -> None:
        """Log the request payload with sensitive data sanitized."""
        with suppress(Exception):
            sanitized_payload = deepcopy(payload)
            
            # Truncate long prompts for logging
            prompt = sanitized_payload.get("prompt", "")
            if len(prompt) > PROMPT_TRUNCATE_LENGTH:
                sanitized_payload["prompt"] = prompt[:PROMPT_TRUNCATE_LENGTH] + "..."
            
            self._log(f"Request payload: {_json.dumps(sanitized_payload, indent=2)}")

    def _handle_response(self, response: dict[str, Any]) -> None:
        """Handle the API response and set output parameters."""
        
        self.parameter_output_values["provider_response"] = response
        
        # Extract generation ID if available
        generation_id = response.get("id", response.get("request_id", ""))
        self.parameter_output_values["generation_id"] = str(generation_id)
        
        # Extract image URLs from response
        # Minimax API returns image URLs in 'data.image_urls' field
        if "data" in response and response["data"]:
            image_data = response["data"]
            if "image_urls" in image_data and isinstance(image_data["image_urls"], list) and len(image_data["image_urls"]) > 0:
                image_urls = image_data["image_urls"]
                self._log(f"Found {len(image_urls)} image URLs in response")
                
                # Save all images
                saved_images = []
                for i, image_url in enumerate(image_urls):
                    if image_url:
                        saved_image = self._save_image_from_url(image_url, index=i)
                        if saved_image:
                            saved_images.append(saved_image)
                    else:
                        self._log(f"Empty image URL found at index {i}")
                
                # Set outputs based on number of images
                if saved_images:
                    # Always set the first image as the primary output
                    self.parameter_output_values["image_url"] = saved_images[0]
                    
                    # Set the images list output
                    self.parameter_output_values["images"] = saved_images
                    
                    self._log(f"Successfully saved {len(saved_images)} images")
                else:
                    self._log("No images were successfully saved")
                    self.parameter_output_values["image_url"] = None
                    self.parameter_output_values["images"] = []
            else:
                self._log("No image_urls array found in response data")
                self.parameter_output_values["image_url"] = None
                self.parameter_output_values["images"] = []
        else:
            self._log("No data field in response")
            self.parameter_output_values["image_url"] = None
            self.parameter_output_values["images"] = []

    def _save_image_from_url(self, image_url: str, index: int = 0) -> ImageUrlArtifact | None:
        """Save image from URL to static storage and return ImageUrlArtifact."""
        try:
            self._log(f"Processing generated image URL {index + 1}")
            
            # Download image bytes
            image_bytes = self._download_bytes_from_url(image_url)
            if image_bytes:
                # Generate filename with timestamp and index
                timestamp = int(time.time())
                filename = f"minimax_image_{timestamp}_{index + 1}.jpg"
                
                # Save to static storage
                from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
                
                static_files_manager = GriptapeNodes.StaticFilesManager()
                saved_url = static_files_manager.save_static_file(image_bytes, filename)
                
                # Create and return ImageUrlArtifact
                image_artifact = ImageUrlArtifact(
                    value=saved_url, 
                    name=filename
                )
                self._log(f"Saved image {index + 1} to static storage as {filename}")
                return image_artifact
            else:
                # Fallback to original URL if download fails
                image_artifact = ImageUrlArtifact(value=image_url)
                self._log(f"Using original image URL for image {index + 1} (download failed)")
                return image_artifact
                
        except Exception as e:
            self._log(f"Failed to save image {index + 1} from URL: {e}")
            # Fallback to original URL
            return ImageUrlArtifact(value=image_url)

    def _set_safe_defaults(self) -> None:
        """Set safe default values for all outputs."""
        self.parameter_output_values["image_url"] = None
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["images"] = []

    @staticmethod
    def _download_bytes_from_url(url: str) -> bytes | None:
        """Download image bytes from URL."""
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            return resp.content
        except Exception:
            return None