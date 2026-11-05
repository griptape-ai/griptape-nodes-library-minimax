from __future__ import annotations

import json as _json
import logging
from contextlib import suppress
from copy import deepcopy
from typing import Any

from griptape.artifacts import AudioUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, DataNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from minimax_errors import API_KEY_URL, request_json, result_error

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

# Default output filename stem; the extension follows the format parameter
DEFAULT_FILENAME_STEM = "minimax_music"


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
        self._output_file = ProjectFileParameter(
            node=self, name="output_file", default_filename=f"{DEFAULT_FILENAME_STEM}.mp3"
        )
        self._output_file.add_parameter()
        # Runs after ProjectFileParameter's reset, so the editor shows the format-matched default
        self.get_parameter_by_name("output_file").on_incoming_connection_removed.append(
            lambda *_: self._sync_default_filename_extension(always_publish=True)
        )

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

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Keep a default output filename's extension in step with the audio format."""
        if parameter.name in ("format", "output_file"):
            self._sync_default_filename_extension()

        return super().after_value_set(parameter, value)

    def _sync_default_filename_extension(self, *, always_publish: bool = False) -> None:
        audio_format = self.get_parameter_value("format")
        if audio_format not in AUDIO_FORMAT_OPTIONS:
            return
        default_filenames = {f"{DEFAULT_FILENAME_STEM}.{fmt}" for fmt in AUDIO_FORMAT_OPTIONS}
        filename = f"{DEFAULT_FILENAME_STEM}.{audio_format}"
        current = self.get_parameter_value("output_file")
        if not isinstance(current, str) or current not in default_filenames:
            return
        if current != filename:
            self.set_parameter_value("output_file", filename)
        if current != filename or always_publish:
            self.publish_update_to_parameter("output_file", filename)

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

        # Validate prompt
        prompt = self.get_parameter_value("prompt")
        if not prompt or not prompt.strip():
            exceptions.append(NodeError("'Prompt' is empty. Enter a prompt and run the node again."))
        elif len(prompt) < 10:
            exceptions.append(NodeError(f"'Prompt' must be at least 10 characters. It has {len(prompt)}."))
        elif len(prompt) > 300:
            exceptions.append(NodeError(f"'Prompt' must be 300 characters or less. It has {len(prompt)}."))

        # Validate lyrics
        lyrics = self.get_parameter_value("lyrics")
        if not lyrics or not lyrics.strip():
            exceptions.append(NodeError("'Lyrics' is empty. Enter the song lyrics and run the node again."))
        elif len(lyrics) < 10:
            exceptions.append(NodeError(f"'Lyrics' must be at least 10 characters. It has {len(lyrics)}."))
        elif len(lyrics) > 3000:
            exceptions.append(NodeError(f"'Lyrics' must be 3000 characters or less. It has {len(lyrics)}."))

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

            self.parameter_output_values["provider_response"] = response_data

            # Check status
            data = response_data.get("data") or {}
            status = data.get("status")

            if status == 2:  # Completed
                self._log("Music generation completed successfully")
                audio_url = data.get("audio")

                if not audio_url:
                    raise result_error("audio URL", response=response_data)

                # Download and save the audio
                self._save_audio_from_url(audio_url, params["format"])
            else:
                raise NodeError(
                    "Minimax did not return finished audio. Run the node again.",
                    fields={"status": str(status)},
                    response=response_data,
                )

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
        api_key = GriptapeNodes.SecretsManager().get_secret(self.API_KEY_NAME)
        if not api_key:
            self._set_safe_defaults()
            raise NodeError.from_missing_secret(self.API_KEY_NAME, key_url=API_KEY_URL)
        return api_key

    def _submit_request(self, params: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        """Submit the music generation request to Minimax API."""
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
        self._log("Received response from API")
        return response_data

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
            },
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
        """Save audio from URL to project storage."""
        self._log("Downloading generated audio from URL")

        # Download audio bytes
        audio_bytes = File(audio_url).read_bytes()

        # Save to project storage
        saved = self._output_file.build_file().write_bytes(audio_bytes)

        # Create AudioUrlArtifact
        self.parameter_output_values["audio_url"] = AudioUrlArtifact(value=saved.location, name=saved.location)
        self._log(f"Saved audio to project storage as {saved.location}")

    def _set_safe_defaults(self) -> None:
        """Set safe default values for all outputs."""
        self.parameter_output_values["audio_url"] = None
        self.parameter_output_values["provider_response"] = None
