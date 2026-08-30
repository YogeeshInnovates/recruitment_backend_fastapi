import os

import httpx

GROQ_ACCOUNT_KEYS_ENV = "GROQ_ACCOUNT_KEYS"
GROQ_SINGLE_KEY_ENV = "GROQ_API_KEY"
GROQ_MODEL_ENV = "GROQ_TRANSCRIBE_MODEL"
GROQ_TRANSCRIBE_MODEL = os.getenv(GROQ_MODEL_ENV, "whisper-large-v3-turbo")
GROQ_TRANSCRIBE_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"


def _groq_keys():
    account_keys = os.getenv(GROQ_ACCOUNT_KEYS_ENV, "")
    keys = [k.strip() for k in account_keys.split(",") if k.strip()]
    single_key = os.getenv(GROQ_SINGLE_KEY_ENV, "").strip()
    if not keys and single_key:
        keys = [single_key]
    return keys


def transcribe_audio(file_bytes: bytes, filename: str = "audio.webm",
                     content_type: str = "audio/webm") -> dict:
    keys = _groq_keys()
    if not keys:
        return {"text": ""}

    last_error = None
    for key in keys:
        try:
            files = {"file": (filename, file_bytes, content_type)}
            data = {"model": GROQ_TRANSCRIBE_MODEL}
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    GROQ_TRANSCRIBE_ENDPOINT,
                    headers={"Authorization": f"Bearer {key}"},
                    data=data,
                    files=files,
                )
            if response.status_code == 200:
                payload = response.json()
                return {"text": (payload.get("text") or "").strip()}
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as e:
            last_error = str(e)

    return {"text": "", "error": f"transcription_failed: {last_error}"}