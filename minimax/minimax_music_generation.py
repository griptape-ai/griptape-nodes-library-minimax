from __future__ import annotations

import json as _json
import logging
import time
from contextlib import suppress
from copy import deepcopy
from typing import Any

import requests
from griptape.artifacts import AudioUrlArtifact

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, DataNode
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

logger = logging.getLogger(__name__)

__all__ = ["MinimaxMusicGeneration"]

# Define constants
PROMPT_TRUNCATE_LENGTH = 100
DEFAULT_TIMEOUT = 120  # Music generation may take longer

# Model options for Minimax music generation
MODEL_OPTIONS = ["music-1.5"]

# Sample rate options
SAMPLE_RATE_OPTIONS = [16000, 24000, 32000, 44100]

# Bitrate options
BITRATE_OPTIONS = [32000, 64000, 128000, 256000]

# Audio format options
AUDIO_FORMAT_OPTIONS = ["mp3", "wav", "pcm"]


class MinimaxMusicGeneration(DataNode):
    """Generate music using Minimax music generation API.
    
    This node uses the Minimax API to generate music from text prompts and lyrics.
    
    Inputs:
        - prompt (str): Description of the music style, mood, and scenario (10-300 characters)
        - lyrics (str): Lyrics of the song with optional structure tags (10-3000 characters)
        - model (str): Model to use for generation (currently only music-1.5)
        - sample_rate (int): Audio sampling rate (16000, 24000, 32000, 44100)
        - bitrate (int): Audio bitrate (32000, 64000, 128000, 256000)
        - format (str): Audio output format (mp3, wav, pcm)
        
    Outputs:
        - audio_url (AudioUrlArtifact): Generated music as URL artifact
        - provider_response (dict): Full API response
    """
    
    SERVICE_NAME = "Minimax"
    API_KEY_NAME = "MINIMAX_API_KEY"
    API_BASE_URL = "https://api.minimax.io/v1/music_generation"
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "Audio Generation"
        self.description = "Generate music using Minimax music generation API"
        
        # Prompt parameter
        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str"],
                type="str",
                tooltip="A description of the music, specifying style, mood, and scenario. For example: 'Pop, melancholic, perfect for a rainy night'. Length: 10-300 characters.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True,
                    "placeholder_text": "Describe the music style, mood, and scenario...",
                    "display_name": "Prompt",
                },
            )
        )
        
        # Lyrics parameter
        self.add_parameter(
            Parameter(
                name="lyrics",
                input_types=["str"],
                type="str",
                tooltip="Lyrics of the song. Use \\n to separate lines. You may add structure tags like [Intro], [Verse], [Chorus], [Bridge], [Outro] to enhance the arrangement. Length: 10-3000 characters.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True,
                    "placeholder_text": "[Verse]\\nYour lyrics here...\\n\\n[Chorus]\\nMore lyrics...",
                    "display_name": "Lyrics",
                },
            )
        )
        
        # Model selection
        self.add_parameter(
            Parameter(
                name="model",
                input_types=["str"],
                type="str",
                default_value="music-1.5",
                tooltip="Model to use for music generation",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=MODEL_OPTIONS)},
                ui_options={"display_name": "Model"},
            )
        )
        
        # Audio settings in a parameter group
        from griptape_nodes.exe_types.core_types import ParameterGroup
        
        with ParameterGroup(name="Audio Settings") as audio_group:
            Parameter(
                name="sample_rate",
                input_types=["int"],
                type="int",
                default_value=44100,
                tooltip="Audio sampling rate. Higher values provide better quality but larger file sizes.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=SAMPLE_RATE_OPTIONS)},
                ui_options={"display_name": "Sample Rate (Hz)"},
            )
            
            Parameter(
                name="bitrate",
                input_types=["int"],
                type="int",
                default_value=128000,
                tooltip="Audio bitrate. Higher values provide better quality but larger file sizes.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=BITRATE_OPTIONS)},
                ui_options={"display_name": "Bitrate (bps)"},
            )
            
            Parameter(
                name="format",
                input_types=["str"],
                type="str",
                default_value="mp3",
                tooltip="Audio output format",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=AUDIO_FORMAT_OPTIONS)},
                ui_options={"display_name": "Format"},
            )
        
        audio_group.ui_options = {"collapsed": True}
        self.add_node_element(audio_group)
        
        # OUTPUTS
        self.add_parameter(
            Parameter(
                name="audio_url",
                output_type="AudioUrlArtifact",
                type="AudioUrlArtifact",
                tooltip="Generated music as URL artifact",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"is_full_width": True, "pulse_on_run": True},
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
        
        # Validate prompt
        prompt = self.get_parameter_value("prompt")
        if not prompt or not prompt.strip():
            exceptions.append(ValueError(f"{self.name}: Prompt is required"))
        elif len(prompt) < 10:
            exceptions.append(ValueError(f"{self.name}: Prompt must be at least 10 characters (current: {len(prompt)} characters)"))
        elif len(prompt) > 300:
            exceptions.append(ValueError(f"{self.name}: Prompt must be 300 characters or less (current: {len(prompt)} characters)"))
        
        # Validate lyrics
        lyrics = self.get_parameter_value("lyrics")
        if not lyrics or not lyrics.strip():
            exceptions.append(ValueError(f"{self.name}: Lyrics are required"))
        elif len(lyrics) < 10:
            exceptions.append(ValueError(f"{self.name}: Lyrics must be at least 10 characters (current: {len(lyrics)} characters)"))
        elif len(lyrics) > 3000:
            exceptions.append(ValueError(f"{self.name}: Lyrics must be 3000 characters or less (current: {len(lyrics)} characters)"))
        
        return exceptions if exceptions else None

    def process(self) -> AsyncResult[None]:
        """Process the music generation request asynchronously."""
        yield lambda: self._process()

    def _process(self) -> None:
        """Main processing method for music generation."""
        # Get parameters
        params = self._get_parameters()
        
        # Validate API key
        api_key = self._validate_api_key()
        
        # Prepare headers
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        # Submit request and get result
        try:
            response_data = self._submit_request(params, headers)
            if not response_data:
                return
            
            self.parameter_output_values["provider_response"] = response_data
            
            # Check status
            data = response_data.get("data", {})
            status = data.get("status")
            
            if status == 2:  # Completed
                self._log("Music generation completed successfully")
                audio_url = data.get("audio")
                
                if audio_url:
                    # Download and save the audio
                    self._save_audio_from_url(audio_url, params["format"])
                else:
                    self._log("No audio URL in response")
                    self._set_safe_defaults()
            else:
                self._log(f"Unexpected status: {status}")
                self._set_safe_defaults()
                
        except Exception as e:
            self._log(f"Error in music generation: {e}")
            self._set_safe_defaults()
            raise

    def _get_parameters(self) -> dict[str, Any]:
        """Get and validate all parameters."""
        params = {
            "prompt": self.get_parameter_value("prompt") or "",
            "lyrics": self.get_parameter_value("lyrics") or "",
            "model": self.get_parameter_value("model") or "music-1.5",
            "sample_rate": self.get_parameter_value("sample_rate") or 44100,
            "bitrate": self.get_parameter_value("bitrate") or 128000,
            "format": self.get_parameter_value("format") or "mp3",
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
        """Submit the music generation request to Minimax API."""
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
            self._log(f"Received response from API")
            
            # Check base_resp for errors
            base_resp = response_data.get("base_resp", {})
            status_code = base_resp.get("status_code")
            
            if status_code != 0:
                status_msg = base_resp.get("status_msg", "Unknown error")
                self._log(f"API error: {status_code} - {status_msg}")
                raise RuntimeError(f"Music generation failed: {status_msg} (code: {status_code})")
            
            return response_data
            
        except requests.RequestException as e:
            self._log(f"Request failed: {e}")
            msg = f"{self.name} request failed: {e}"
            raise RuntimeError(msg) from e

    def _build_payload(self, params: dict[str, Any]) -> dict[str, Any]:
        """Build the API request payload."""
        payload = {
            "model": params["model"],
            "prompt": params["prompt"],
            "lyrics": params["lyrics"],
            "stream": False,
            "output_format": "url",
            "audio_setting": {
                "sample_rate": params["sample_rate"],
                "bitrate": params["bitrate"],
                "format": params["format"],
            }
        }
        
        return payload

    def _log_request(self, payload: dict[str, Any]) -> None:
        """Log the request payload with sensitive data sanitized."""
        with suppress(Exception):
            sanitized_payload = deepcopy(payload)
            if "prompt" in sanitized_payload and len(sanitized_payload["prompt"]) > PROMPT_TRUNCATE_LENGTH:
                sanitized_payload["prompt"] = sanitized_payload["prompt"][:PROMPT_TRUNCATE_LENGTH] + "..."
            if "lyrics" in sanitized_payload and len(sanitized_payload["lyrics"]) > PROMPT_TRUNCATE_LENGTH:
                sanitized_payload["lyrics"] = sanitized_payload["lyrics"][:PROMPT_TRUNCATE_LENGTH] + "..."
            
            self._log(f"Request payload: {_json.dumps(sanitized_payload, indent=2)}")

    def _save_audio_from_url(self, audio_url: str, audio_format: str) -> None:
        """Save audio from URL to static storage."""
        try:
            self._log("Downloading generated audio from URL")
            
            # Download audio bytes
            audio_bytes = self._download_bytes_from_url(audio_url)
            if audio_bytes:
                # Generate filename with timestamp
                filename = f"minimax_music_{int(time.time())}.{audio_format}"
                
                # Save to static storage
                from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
                
                static_files_manager = GriptapeNodes.StaticFilesManager()
                saved_url = static_files_manager.save_static_file(audio_bytes, filename)
                
                # Create AudioUrlArtifact
                self.parameter_output_values["audio_url"] = AudioUrlArtifact(
                    value=saved_url, 
                    name=filename
                )
                self._log(f"Saved audio to static storage as {filename}")
            else:
                # Fallback to original URL if download fails
                self.parameter_output_values["audio_url"] = AudioUrlArtifact(value=audio_url)
                self._log("Using original audio URL (download failed)")
                
        except Exception as e:
            self._log(f"Failed to save audio from URL: {e}")
            # Fallback to original URL
            self.parameter_output_values["audio_url"] = AudioUrlArtifact(value=audio_url)

    def _set_safe_defaults(self) -> None:
        """Set safe default values for all outputs."""
        self.parameter_output_values["audio_url"] = None
        self.parameter_output_values["provider_response"] = None

    @staticmethod
    def _download_bytes_from_url(url: str) -> bytes | None:
        """Download audio bytes from URL."""
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            return resp.content
        except Exception:
            return None

