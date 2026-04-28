"""Fish Audio TTS provider implementation.

API: https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech
Auth: Bearer token via FISH_AUDIO_API_KEY env var
"""

import os
import requests
from typing import Dict, List, Optional
from ..base import TTSProvider
from ...utils.config_conversation import load_conversation_config

_FISH_AUDIO_API_URL = "https://api.fish.audio/v1/tts"
_DEFAULT_MODEL = "s2-pro"


class FishAudioTTS(TTSProvider):
    """Fish Audio Text-to-Speech provider.

    Supports high-quality Chinese TTS and voice cloning via reference_id.
    Each call to generate_audio corresponds to one dialogue turn (one speaker
    block), which is already the natural granularity used by text_to_speech.py.
    """

    PROVIDER_SSML_TAGS: List[str] = ["p", "s"]

    def __init__(self, api_key: Optional[str] = None, model: str = _DEFAULT_MODEL):
        """
        Initialize Fish Audio TTS provider.

        Args:
            api_key: Fish Audio API key. Falls back to FISH_AUDIO_API_KEY env var.
            model: TTS model to use. Defaults to "s2-pro" (recommended).
        """
        self.api_key = api_key or os.environ.get("FISH_AUDIO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Fish Audio API key must be provided or set in FISH_AUDIO_API_KEY"
            )
        self.model = model

        # Load per-voice speed settings from conversation_config.yaml
        try:
            conv_config = load_conversation_config()
            fishaudio_cfg = conv_config.get("text_to_speech", {}).get("fishaudio", {})
            self.voice_speeds: Dict[str, float] = {
                str(k): float(v)
                for k, v in fishaudio_cfg.get("voice_speeds", {}).items()
            }
        except Exception:
            self.voice_speeds = {}

    def get_supported_tags(self) -> List[str]:
        return self.PROVIDER_SSML_TAGS

    def generate_audio(
        self,
        text: str,
        voice: str,
        model: str,
        voice2: str = None,
    ) -> bytes:
        """Generate audio using Fish Audio API.

        Called once per speaker turn. The caller (text_to_speech.py) already
        splits the transcript into individual Person1/Person2 turns, so this
        method simply forwards the text as-is to the Fish Audio API.

        Args:
            text: Text to convert to speech (one dialogue turn).
            voice: Fish Audio voice model ID (reference_id).
                   Pass an empty string to use the default system voice.
            model: TTS model name (e.g. "s2-pro", "s1"). Overrides constructor default.
            voice2: Unused (single-speaker provider).

        Returns:
            Raw MP3 audio bytes.

        Raises:
            RuntimeError: If the API request fails.
        """
        self.validate_parameters(text, voice or "_default_", model or self.model)

        effective_model = model or self.model
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "model": effective_model,
        }
        payload: dict = {
            "text": text,
            "format": "mp3",
            "mp3_bitrate": 128,
            "normalize": True,
            "latency": "normal",
        }
        if voice:
            payload["reference_id"] = voice
        speed = self.voice_speeds.get(voice, 1.0)
        if speed != 1.0:
            payload["speed"] = speed

        try:
            response = requests.post(
                _FISH_AUDIO_API_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return response.content
        except requests.HTTPError as e:
            raise RuntimeError(
                f"Fish Audio API error {response.status_code}: {response.text}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to generate audio with Fish Audio: {e}") from e
