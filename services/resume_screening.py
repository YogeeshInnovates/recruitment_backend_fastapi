import hashlib
import json
import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from services.llm_screening import gemini_generate, extract_json
from services.groq_interviewer import get_pinecone_index, chunk_text
from services.ai_engine import screen_resume as heuristic_screen_resume
from services.screening_scheduler import compute_schedule

logger = logging.getLogger(__name__)

MAX_RESUMES = 10

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "screening_cache.jsonl"
BATCH_FILE = DATA_DIR / "screening_batches.jsonl"
JD_CHAR_LIMIT = 8000
_gemini_file_lock = threading.Lock()

RESUME_SCREENING_PROMPT = """You will receive ONE job description and A LIST of up to 10 candidate resumes. Analyze EACH resume independently against the SAME job description. Return ONLY a valid JSON array — no markdown, no preamble, no explanation. Do not compare candidates against each other — score each purely against the JD.

For EACH candidate, evaluate using these weighted factors:
1. Skills match (40%)
2. Experience relevance (30%)
3. Project/impact quality (20%)
4. Education/certifications (10%)

Classify experience_level:
- \"fresher\": 0–1 years OR no professional experience
- \"mid\": 2–5 years
- \"senior\": 6+ years OR title indicates lead/senior/architect

Assign interview_duration_minutes strictly by experience_level:
- fresher: 15 | mid: 25 | senior: 30
(hr_round_minutes is always 15 regardless of level)

Return EXACTLY this JSON array schema (one object per candidate, same order as input):
[
  {{
    \"candidate_id\": \"<pass-through from input>\",
    \"candidate_name\": \"<from resume>\",
    \"score\": <integer 0-100>,
    \"score_breakdown\": {{
      \"skills_match\": <0-40>,
      \"experience_relevance\": <0-30>,
      \"project_quality\": <0-20>,
      \"education\": <0-10>
    }},
    \"experience_level\": \"fresher | mid | senior\",
    \"interview_duration_minutes\": <15|25|30>,
    \"hr_round_minutes\": 15,
    \"matched_skills\": [\"...\"],
    \"missing_skills\": [\"...\"],
    \"summary\": \"<2-3 line justification>\"
  }}
]

Be strict and consistent across all candidates — apply the same standard to every resume, don't grade on a curve relative to the batch. If a resume is a poor fit, score below 40.

INPUT FORMAT YOU WILL RECEIVE:
Job Description: <jd_text>
Resumes:
[
  {{\"candidate_id\": \"c1\", \"resume_text\": \"...\"}},
  {{\"candidate_id\": \"c2\", \"resume_text\": \"...\"}}
]"""

DURATION_BY_LEVEL = {"fresher": 15, "mid": 25, "senior": 30}


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _extract_json_array(text: Optional[str]):
    """Pull the first top-level JSON array out of model output."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def _load_cache() -> Dict[str, Dict[str, Any]]:
    _ensure_data_dir()
    cache: Dict[str, Dict[str, Any]] = {}
    if not CACHE_FILE.exists():
        return cache
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = f"{entry.get('jd_hash')}:{entry.get('resume_hash')}"
                    cache[key] = entry.get("result") or {}
                except Exception:
                    continue
    except Exception as e:
        logger.warning("Failed to load screening cache: %s", e)
    return cache


def _persist_cache_entry(jd_hash: str, resume_hash: str, result: Dict[str, Any]):
    _ensure_data_dir()
    entry = {"jd_hash": jd_hash, "resume_hash": resume_hash, "result": result, "saved_at": datetime.utcnow().isoformat()}
    with _gemini_file_lock:
        try:
            with open(CACHE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning("Failed to persist screening cache entry: %s", e)


def persist_batch_report(report: Dict[str, Any]):
    _ensure_data_dir()
    with _gemini_file_lock:
        try:
            with open(BATCH_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(report) + "\n")
        except Exception as e:
            logger.warning("Failed to persist screening batch report: %s", e)


def _index_jd_in_pinecone(batch_id: str, job_description: str) -> None:
    try:
        index = get_pinecone_index()
        namespace = f"screening_{batch_id}"
        chunks = chunk_text(job_description or "", max_tokens=500)
        records = [
            {
                "_id": f"jd-{i}",
                "text": chunk[:1500],
                "chunk_index": i,
                "source": "job_description",
            }
            for i, chunk in enumerate(chunks)
        ]
        if records:
            index.upsert_records(namespace=namespace, records=records)
            logger.info("Indexed %d JD chunks into namespace %s", len(records), namespace)
    except Exception as e:
        logger.warning("Failed to index JD into Pinecone (non-fatal): %s", e)


def _jd_context(job_description: str, batch_id: str) -> str:
    if len(job_description or "") <= JD_CHAR_LIMIT:
        return (job_description or "").strip()
    try:
        index = get_pinecone_index()
        namespace = f"screening_{batch_id}"
        from pinecone import SearchQuery
        results = index.search_records(
            namespace=namespace,
            query=SearchQuery(inputs={"text": (job_description or "")[:2000]}, top_k=8),
            fields=["text"],
        )
        chunks = [h["fields"]["text"] for h in results.result.hits if "text" in h.get("fields", {})]
        joined = "\n\n".join(chunks) if chunks else ""
        return joined[:JD_CHAR_LIMIT] if joined else (job_description or "")[:JD_CHAR_LIMIT]
    except Exception as e:
        logger.warning("Pinecone JD retrieval failed (non-fatal): %s", e)
        return (job_description or "")[:JD_CHAR_LIMIT]


def _heuristic_result(candidate: Dict[str, Any], jd_text: str) -> Dict[str, Any]:
    fallback = heuristic_screen_resume(candidate.get("resume_text", ""), jd_text)
    level = "mid"
    score = float(fallback.get("score", 0) or 0)
    if score < 40:
        level = "fresher"
    elif score >= 75:
        level = "senior"
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "candidate_name": candidate.get("name", ""),
        "score": int(round(score)),
        "score_breakdown": {
            "skills_match": 0,
            "experience_relevance": 0,
            "project_quality": 0,
            "education": 0,
        },
        "experience_level": level,
        "interview_duration_minutes": DURATION_BY_LEVEL[level],
        "hr_round_minutes": 15,
        "matched_skills": fallback.get("matched_skills", []) or [],
        "missing_skills": fallback.get("missing_skills", []) or [],
        "summary": fallback.get("analysis", "") or "",
        "source": "heuristic",
    }


def _normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    level = str(item.get("experience_level", "mid") or "mid").lower().strip()
    if level not in DURATION_BY_LEVEL:
        level = "mid"
    duration = int(item.get("interview_duration_minutes") or DURATION_BY_LEVEL[level])
    if duration not in (15, 25, 30):
        duration = DURATION_BY_LEVEL[level]
    breakdown = item.get("score_breakdown") or {}
    if not isinstance(breakdown, dict):
        breakdown = {}
    bd = {
        "skills_match": int(breakdown.get("skills_match", 0) or 0),
        "experience_relevance": int(breakdown.get("experience_relevance", 0) or 0),
        "project_quality": int(breakdown.get("project_quality", 0) or 0),
        "education": int(breakdown.get("education", 0) or 0),
    }
    matched = item.get("matched_skills") or []
    missing = item.get("missing_skills") or []
    if isinstance(matched, str):
        matched = [matched]
    if isinstance(missing, str):
        missing = [missing]
    return {
        "candidate_id": str(item.get("candidate_id", "")),
        "candidate_name": str(item.get("candidate_name", "")),
        "score": max(0, min(100, int(item.get("score", 0) or 0))),
        "score_breakdown": bd,
        "experience_level": level,
        "interview_duration_minutes": duration,
        "hr_round_minutes": 15,
        "matched_skills": [str(s) for s in matched][:25],
        "missing_skills": [str(s) for s in missing][:25],
        "summary": str(item.get("summary", "") or ""),
        "source": "gemini",
    }


async def screen_resume_batch(
    batch_id: str,
    job_description: str,
    role: str,
    round_name: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Screen up to 10 resumes against a single JD in one Gemini batch call."""
    if not candidates:
        raise ValueError("At least one candidate resume is required")
    if len(candidates) > MAX_RESUMES:
        raise ValueError(f"Maximum {MAX_RESUMES} resumes allowed")

    jd_text = (job_description or "").strip()
    if not jd_text:
        raise ValueError("Job description is required")

    jd_hash = _sha(jd_text)
    _ensure_data_dir()
    _index_jd_in_pinecone(batch_id, job_description)
    jd_ctx = _jd_context(job_description, batch_id)

    cache = _load_cache()
    results: List[Dict[str, Any]] = []
    uncached_candidates: List[Dict[str, Any]] = []
    used_cache = False

    for cand in candidates:
        resume_hash = _sha(cand.get("resume_text", ""))
        cached = cache.get(f"{jd_hash}:{resume_hash}")
        if cached:
            used_cache = True
            item = _normalize_item(cached)
            item["source"] = "cache"
            item["candidate_id"] = cand.get("candidate_id", "")
            item["candidate_name"] = item["candidate_name"] or cand.get("name", "")
            results.append(item)
        else:
            uncached_candidates.append({**cand, "resume_hash": resume_hash})

    source = "gemini"
    if uncached_candidates:
        prepared = [
            {"candidate_id": c.get("candidate_id", ""), "resume_text": c.get("resume_text", "")}
            for c in uncached_candidates
        ]
        resume_block = json.dumps(prepared, ensure_ascii=False)
        prompt = f"{RESUME_SCREENING_PROMPT}\n\nJob Description:\n{jd_ctx}\n\nResumes:\n{resume_block}"

        items: Optional[List[Dict[str, Any]]] = None
        raw = gemini_generate(prompt, temperature=0.2, max_output_tokens=8192)
        items = _extract_json_array(raw)
        expected_ids = set(c.get("candidate_id", "") for c in uncached_candidates)
        valid = (
            items is not None
            and isinstance(items, list)
            and len(items) == len(uncached_candidates)
            and {str(i.get("candidate_id", "")) for i in items} == expected_ids
        )
        if not valid:
            logger.warning("Batch screening validation failed — retrying once. raw=%s", (raw or "")[:300])
            prompt_retry = (
                "Previous response was rejected: must return ONLY a JSON array with exactly "
                f"{len(uncached_candidates)} objects, one per candidate, using the EXACT "
                "candidate_id values passed in, in the same order. No markdown, no commentary.\n\n"
                f"{prompt}"
            )
            raw2 = gemini_generate(prompt_retry, temperature=0.2, max_output_tokens=8192)
            items = _extract_json_array(raw2)
            valid = (
                items is not None
                and isinstance(items, list)
                and len(items) == len(uncached_candidates)
                and {str(i.get("candidate_id", "")) for i in items} == expected_ids
            )

        by_id = {}
        if valid:
            for it in items:
                by_id[str(it.get("candidate_id", ""))] = it

        for cand in uncached_candidates:
            if valid and cand.get("candidate_id") in by_id:
                item = _normalize_item(by_id[cand.get("candidate_id")])
                item["candidate_id"] = cand.get("candidate_id", "")
                item["candidate_name"] = item["candidate_name"] or cand.get("name", "")
            else:
                logger.error("Gemini batch screening failed for %d candidates; using heuristic fallback", len(uncached_candidates))
                source = "heuristic"
                item = _heuristic_result(cand, jd_text)
            _persist_cache_entry(jd_hash, cand["resume_hash"], item)
            results.append(item)

    for rank, item in enumerate(sorted(results, key=lambda r: (-r["score"], r["candidate_name"])), start=1):
        item["rank"] = rank

    candidates_out = [{"candidate_id": i["candidate_id"], **i} for i in results]
    schedule = compute_schedule(candidates_out)

    report = {
        "batch_id": batch_id,
        "job_description": jd_text,
        "role": role,
        "round": round_name,
        "source": source,
        "created_at": datetime.utcnow().isoformat(),
        "cached": used_cache,
        "candidates": candidates_out,
        "schedule": schedule,
    }
    persist_batch_report(report)

    return {
        "batch_id": batch_id,
        "status": "screened",
        "source": source,
        "candidates": candidates_out,
        "schedule": schedule,
    }