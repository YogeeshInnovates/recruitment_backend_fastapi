from fastapi import APIRouter, HTTPException, UploadFile, File
from models.schemas import (
    ResumeScreenRequest,
    ResumeScreenResponse,
    JobMatchRequest,
    JobMatchResponse,
    JobMatchItem,
    ApplicationAnalyzeRequest,
    ApplicationAnalyzeResponse,
)
from services.ai_engine import screen_resume, match_jobs, analyze_application
from services.llm_screening import screen_resume_smart
from services.audio_transcription import transcribe_audio

router = APIRouter()


@router.post("/screen-resume", response_model=ResumeScreenResponse)
async def screen_resume_endpoint(request: ResumeScreenRequest):
    try:
        result = screen_resume_smart(request.resume_text, request.job_description)
        return ResumeScreenResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/match-jobs", response_model=JobMatchResponse)
async def match_jobs_endpoint(request: JobMatchRequest):
    try:
        jobs_dicts = [j.model_dump() for j in request.jobs]
        matches = match_jobs(request.resume_text, jobs_dicts)
        return JobMatchResponse(matches=[JobMatchItem(**m) for m in matches])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-application", response_model=ApplicationAnalyzeResponse)
async def analyze_application_endpoint(request: ApplicationAnalyzeRequest):
    try:
        result = analyze_application(
            request.resume_text, request.job_description, request.cover_letter
        )
        return ApplicationAnalyzeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")
    try:
        result = transcribe_audio(
            data,
            filename=file.filename or "audio.webm",
            content_type=file.content_type or "audio/webm",
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
