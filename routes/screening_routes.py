import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from models.schemas import (
    ScreenRunRequest,
    ScreenRunResponse,
    ScreenConfirmRequest,
    ScreenConfirmResponse,
    ScreeningReport,
)
from services.resume_screening import screen_resume_batch, persist_batch_report, DATA_DIR

logger = logging.getLogger(__name__)

router = APIRouter()

_CONFIRMED_FILE = DATA_DIR / "screening_confirmed.jsonl"


def _mark_confirmed(batch_id: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIRMED_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"batch_id": batch_id, "confirmed_at": __import__("datetime").datetime.utcnow().isoformat()}) + "\n")


def _load_report(batch_id: str):
    batch_file = DATA_DIR / "screening_batches.jsonl"
    if not batch_file.exists():
        return None
    try:
        with open(batch_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get("batch_id") == batch_id:
                    return entry
    except Exception as e:
        logger.error("Failed to read screening batches: %s", e)
    return None


@router.post("/screening/run", response_model=ScreenRunResponse)
async def screening_run(request: ScreenRunRequest):
    try:
        candidates = [c.model_dump() for c in request.candidates]
        result = await screen_resume_batch(
            batch_id=request.batch_id,
            job_description=request.job_description,
            role=request.role,
            round_name=request.round,
            candidates=candidates,
        )
        return ScreenRunResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Screening run failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")


@router.post("/screening/confirm", response_model=ScreenConfirmResponse)
async def screening_confirm(request: ScreenConfirmRequest):
    if not request.batch_id:
        raise HTTPException(status_code=400, detail="batch_id is required")
    report = _load_report(request.batch_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"No screening batch found for {request.batch_id}")
    _mark_confirmed(request.batch_id)
    return ScreenConfirmResponse(
        batch_id=request.batch_id,
        status="confirmed",
        candidates=report.get("candidates", []),
        schedule=report.get("schedule", []),
    )


@router.get("/screening/result/{batch_id}", response_model=ScreeningReport)
async def screening_result(batch_id: str):
    report = _load_report(batch_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"No screening batch found for {batch_id}")
    return ScreeningReport(
        batch_id=report.get("batch_id", batch_id),
        job_description=report.get("job_description", ""),
        role=report.get("role", ""),
        round=report.get("round", ""),
        source=report.get("source", "gemini"),
        created_at=report.get("created_at", ""),
        candidates=report.get("candidates", []),
        schedule=report.get("schedule", []),
    )