import json
import logging
import os
import re

import httpx

from services.ai_engine import screen_resume as heuristic_screen_resume

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_ACCOUNT_KEYS = os.getenv("GEMINI_ACCOUNT_KEYS", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta")

FREE_MODELS = ["gemini-2.0-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]


def _gemini_keys():
    keys = []
    if GEMINI_API_KEY:
        keys.append(GEMINI_API_KEY)
    if GEMINI_ACCOUNT_KEYS:
        keys.extend(k.strip() for k in GEMINI_ACCOUNT_KEYS.split(",") if k.strip())
    return keys


def gemini_generate(prompt, temperature=0.3, max_output_tokens=4096):
    """Call the Gemini generateContent REST API. Returns plain text or None."""
    keys = _gemini_keys()
    if not keys:
        logger.warning("No Gemini API keys configured")
        return None
    models_to_try = [GEMINI_MODEL] + [m for m in FREE_MODELS if m != GEMINI_MODEL]
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    for model in models_to_try:
        for key in keys:
            try:
                url = f"{GEMINI_ENDPOINT}/models/{model}:generateContent?key={key}"
                resp = httpx.post(url, json=body, timeout=90)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates") or []
                    if not candidates:
                        continue
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts).strip()
                    if text:
                        return text
                else:
                    logger.warning("Gemini %s key %s status %s: %s", model, key[:8], resp.status_code, resp.text[:200])
            except Exception as e:
                logger.warning("Gemini %s error: %s", model, e)
    return None


def extract_json(text):
    """Robustly pull the first JSON object out of model output text."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


_SCREENING_PROMPT = """You are a senior technical recruiter screening a candidate resume against a job description.
Return ONLY a valid JSON object (no markdown, no commentary) with exactly these keys:
{{
  "score": <number 0-100>,
  "analysis": "<2-4 sentence summary of overall fit>",
  "matched_skills": ["<skill>", ...],
  "missing_skills": ["<skill>", ...],
  "recommendation": "<STRONG_SHORTLIST | SHORTLIST | MAYBE | REJECT>"
}}

JOB DESCRIPTION:
{job}

CANDIDATE RESUME:
{resume}"""


def screen_resume_smart(resume_text, job_description):
    """LLM-powered resume screening with a deterministic heuristic fallback."""
    try:
        prompt = _SCREENING_PROMPT.format(job=job_description, resume=resume_text)
        raw = gemini_generate(prompt, temperature=0.2)
        data = extract_json(raw)
        if data and "score" in data:
            try:
                score = max(0.0, min(100.0, float(data.get("score", 0))))
            except (TypeError, ValueError):
                score = 50.0
            matched = data.get("matched_skills") or []
            missing = data.get("missing_skills") or []
            if isinstance(matched, str):
                matched = [matched]
            if isinstance(missing, str):
                missing = [missing]
            return {
                "score": score,
                "analysis": str(data.get("analysis") or ""),
                "matched_skills": [str(s) for s in matched][:25],
                "missing_skills": [str(s) for s in missing][:25],
                "recommendation": str(data.get("recommendation") or "MAYBE"),
            }
        logger.info("Gemini screening returned no usable JSON; falling back to heuristic.")
    except Exception as e:
        logger.warning("LLM screening failed (%s); falling back to heuristic.", e)
    return heuristic_screen_resume(resume_text, job_description)
