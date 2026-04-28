"""
FastAPI implementation for Podcastify podcast generation service.

This module provides REST endpoints for podcast generation and audio serving,
with configuration management and temporary file handling.
"""

# Disable LangSmith tracing before any langchain import to suppress APIKeyWarning
import os
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import shutil
import yaml
import logging
from typing import Dict, Any
from pathlib import Path
from ..client import generate_podcast
import uvicorn

# Use uvicorn's logger so output always appears in the terminal regardless of
# how the Python root logger is configured at startup time.
logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Cloudflare R2 (S3-compatible) storage — optional, falls back to local disk
# ---------------------------------------------------------------------------
_R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL")
_R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
_R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
_R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")
_R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL")

_r2_enabled = all([_R2_ENDPOINT_URL, _R2_ACCESS_KEY_ID, _R2_SECRET_ACCESS_KEY, _R2_BUCKET_NAME, _R2_PUBLIC_URL])
_s3_client = None

if _r2_enabled:
    try:
        import boto3
        _s3_client = boto3.client(
            "s3",
            endpoint_url=_R2_ENDPOINT_URL,
            aws_access_key_id=_R2_ACCESS_KEY_ID,
            aws_secret_access_key=_R2_SECRET_ACCESS_KEY,
        )
    except Exception as _e:
        print(f"Warning: Failed to initialize R2 client: {_e}")
        _r2_enabled = False


def load_base_config() -> Dict[Any, Any]:
    config_path = Path(__file__).parent.parent / "conversation_config.yaml"
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Warning: Could not load base config: {e}")
        return {}

def merge_configs(base_config: Dict[Any, Any], user_config: Dict[Any, Any]) -> Dict[Any, Any]:
    """Merge user configuration with base configuration, preferring user values."""
    merged = base_config.copy()
    
    # Handle special cases for nested dictionaries
    if 'text_to_speech' in merged and 'text_to_speech' in user_config:
        merged['text_to_speech'].update(user_config.get('text_to_speech', {}))
    
    # Update top-level keys
    for key, value in user_config.items():
        if key != 'text_to_speech':  # Skip text_to_speech as it's handled above
            if value is not None:  # Only update if value is not None
                merged[key] = value
                
    return merged

app = FastAPI()

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_audio")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/generate")
def generate_podcast_endpoint(data: dict):
    """"""
    try:
        # API keys are read from environment variables (configured via Railway Variables).
        # They are NOT accepted from the request body to avoid security risks.
        logger.info("[1/4] Request received — parsing config...")
        # Load base configuration
        base_config = load_base_config()
        
        # Get TTS model and its configuration from base config
        tts_model = data.get('tts_model', base_config.get('text_to_speech', {}).get('default_tts_model', 'openai'))
        tts_base_config = base_config.get('text_to_speech', {}).get(tts_model, {})
        
        # Get voices (use user-provided voices or fall back to defaults)
        voices = data.get('voices', {})
        default_voices = tts_base_config.get('default_voices', {})
        
        # Prepare user configuration
        # Resolve style preset: merge preset instructions with any extra user_instructions
        style_key = data.get('style')
        style_presets = base_config.get('style_presets', {})
        preset_instructions = style_presets.get(style_key, '') if style_key else ''
        extra_instructions = data.get('user_instructions', base_config.get('user_instructions', ''))
        merged_instructions = '\n\n'.join(filter(None, [preset_instructions, extra_instructions]))

        user_config = {
            'creativity': float(data.get('creativity', base_config.get('creativity', 0.7))),
            'conversation_style': data.get('conversation_style', base_config.get('conversation_style', [])),
            'roles_person1': data.get('roles_person1', base_config.get('roles_person1')),
            'roles_person2': data.get('roles_person2', base_config.get('roles_person2')),
            'dialogue_structure': data.get('dialogue_structure', base_config.get('dialogue_structure', [])),
            'podcast_name': data.get('name', base_config.get('podcast_name')),
            'podcast_tagline': data.get('tagline', base_config.get('podcast_tagline')),
            'output_language': data.get('output_language', base_config.get('output_language', 'English')),
            'user_instructions': merged_instructions,
            'engagement_techniques': data.get('engagement_techniques', base_config.get('engagement_techniques', [])),
            'text_to_speech': {
                'default_tts_model': tts_model,
                'model': tts_base_config.get('model'),
                'default_voices': {
                    'question': voices.get('question', default_voices.get('question')),
                    'answer': voices.get('answer', default_voices.get('answer'))
                }
            }
        }

        # print(user_config)

        # Merge configurations
        conversation_config = merge_configs(base_config, user_config)

        # print(conversation_config)
        

        # Generate podcast
        logger.info("[2/4] Starting content generation (LLM) — this may take 1-3 minutes...")
        result = generate_podcast(
            urls=data.get('urls', []),
            conversation_config=conversation_config,
            tts_model=tts_model,
            longform=bool(data.get('is_long_form', False)),
        )
        logger.info("[3/4] Content generation done — processing audio output...")
        # Handle the result
        if isinstance(result, str) and os.path.isfile(result):
            source_path = result
        elif hasattr(result, 'audio_path'):
            source_path = result.audio_path
        else:
            raise HTTPException(status_code=500, detail="Invalid result format")

        filename = f"podcast_{os.urandom(8).hex()}.mp3"

        if _r2_enabled:
            # Store under YYYY/MM/DD/ prefix for easy filtering
            from datetime import datetime, timezone
            date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
            r2_key = f"{date_prefix}/{filename}"

            logger.info("[4/4] Uploading to R2: %s", r2_key)
            _s3_client.upload_file(
                source_path,
                _R2_BUCKET_NAME,
                r2_key,
                ExtraArgs={"ContentType": "audio/mpeg"},
            )
            public_url = f"{_R2_PUBLIC_URL.rstrip('/')}/{r2_key}"
            logger.info("Done — audioUrl: %s", public_url)
            return {"audioUrl": public_url}
        else:
            # Fallback: store locally and return relative path
            output_path = os.path.join(TEMP_DIR, filename)
            shutil.copy2(source_path, output_path)
            logger.info("[4/4] Done — saved locally: %s", filename)
            return {"audioUrl": f"/audio/{filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{filename}")
def serve_audio(filename: str):
    """ Get File Audio From ther Server"""
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/health")
def healthcheck():
    return {
        "status": "healthy",
        "r2_storage": "enabled" if _r2_enabled else "disabled (using local storage fallback)",
    }

@app.get("/styles")
def list_styles():
    """Return available style preset keys."""
    config = load_base_config()
    presets = config.get('style_presets', {})
    return {"styles": list(presets.keys())}

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host=host, port=port)
