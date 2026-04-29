"""Fish Audio TTS provider implementation.

API: https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech
Auth: Bearer token via FISH_AUDIO_API_KEY env var
"""

import logging
import os
import re
import requests
from typing import Dict, List, Optional
from ..base import TTSProvider
from ...utils.config_conversation import load_conversation_config

logger = logging.getLogger(__name__)

_FISH_AUDIO_API_URL = "https://api.fish.audio/v1/tts"
_DEFAULT_MODEL = "s2-pro"

# ---------------------------------------------------------------------------
# Emotion annotation prompt for Fish Audio S2-Pro [bracket] syntax
# S2-Pro accepts free-form natural-language emotion cues in square brackets,
# e.g.  [feel excited and energetic]  or  [calm and thoughtful].
# ---------------------------------------------------------------------------
_ANNOTATION_SYSTEM_PROMPT = """\
You are an emotion annotation assistant for a podcast TTS engine using Fish Audio S2-Pro.

S2-Pro supports natural-language emotion cues in square brackets, e.g.:
  [feel excited and energetic] Welcome to today's episode!
  [calm and thoughtful] That is a really interesting point.
  [curious] Have you ever wondered why this happens?

Rules:
1. Add one [emotion bracket] at the START of each Person's turn (right after the opening tag).
2. For long turns where the emotional tone shifts mid-way, you may insert an additional [emotion bracket] inline at the transition point — but use this sparingly.
3. Bracket content must be a SHORT natural-language description, ≤8 words.
4. Do NOT change, reorder, or paraphrase any dialogue text — only insert brackets.
5. Keep ALL <Person1>...</Person1> and <Person2>...</Person2> tags exactly as-is.
6. Return ONLY the annotated transcript, no explanation, no code fences.
"""

_ANNOTATION_USER_PROMPT = "Annotate the following podcast transcript:\n\n{transcript}"


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

    def _get_annotation_model(self) -> str:
        """Return the LLM model name to use for emotion annotation."""
        try:
            conv_config = load_conversation_config()
            fishaudio_cfg = conv_config.get("text_to_speech", {}).get("fishaudio", {})
            return fishaudio_cfg.get("annotation_model", "gemini-3-flash-preview")
        except Exception:
            return "gemini-3-flash-preview"

    def _annotation_enabled(self) -> bool:
        """Return True unless emotion_annotation is explicitly set to false in config."""
        try:
            conv_config = load_conversation_config()
            fishaudio_cfg = conv_config.get("text_to_speech", {}).get("fishaudio", {})
            return bool(fishaudio_cfg.get("emotion_annotation", True))
        except Exception:
            return True

    def preprocess_transcript(self, text: str) -> str:
        """Annotate the full transcript with S2-Pro [bracket] emotion cues.

        Uses a single LLM call over the entire transcript so the model can
        assign emotions with full dialogue context.  Falls back to the original
        text if annotation fails or is disabled.

        Args:
            text: Raw transcript with <Person1>/<Person2> tags.

        Returns:
            Transcript with [emotion bracket] annotations inserted before each turn.
        """
        if not self._annotation_enabled():
            return text

        try:
            import litellm
        except ImportError:
            logger.warning("litellm not installed; skipping emotion annotation")
            return text

        llm_base_url = os.environ.get("LLM_BASE_URL")
        llm_api_key = os.environ.get("LLM_API_KEY")
        annotation_model = self._get_annotation_model()

        # Prefix model with "openai/" when routing through an OpenAI-compatible aggregator
        model_id = f"openai/{annotation_model}" if llm_base_url else annotation_model

        kwargs: dict = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": _ANNOTATION_SYSTEM_PROMPT},
                {"role": "user", "content": _ANNOTATION_USER_PROMPT.format(transcript=text)},
            ],
            "temperature": 0.4,
        }
        if llm_base_url:
            kwargs["api_base"] = llm_base_url
            kwargs["api_key"] = llm_api_key

        try:
            response = litellm.completion(**kwargs)
            annotated = response.choices[0].message.content.strip()
            # Sanity-check: annotated text must still contain the Person tags
            if "<Person1>" not in annotated or "<Person2>" not in annotated:
                logger.warning("Emotion annotation returned malformed transcript; using original")
                return text
            logger.info("Emotion annotation applied to transcript (model=%s)", annotation_model)
            return annotated
        except Exception as exc:
            logger.warning("Emotion annotation failed (%s); using original transcript", exc)
            return text

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
            # speed must be nested inside prosody per Fish Audio API spec
            payload["prosody"] = {"speed": speed}

        last_exc: Exception | None = None
        for attempt in range(1, 4):
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
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                logger.warning(
                    "Fish Audio connection error (attempt %d/3): %s", attempt, e
                )
                if attempt < 3:
                    import time
                    time.sleep(2 ** attempt)  # 2s, 4s back-off
            except Exception as e:
                raise RuntimeError(f"Failed to generate audio with Fish Audio: {e}") from e

        raise RuntimeError(
            f"Failed to generate audio with Fish Audio after 3 attempts: {last_exc}"
        ) from last_exc
