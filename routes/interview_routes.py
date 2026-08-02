from fastapi import APIRouter, HTTPException
from models.schemas import (
    InterviewSetupRequest,
    InterviewSetupResponse,
    InterviewChatRequest,
    InterviewChatResponse,
    InterviewScoreRequest,
    InterviewScoreResponse,
    BatchIndexRequest,
    BatchIndexResponse,
)
from services.groq_interviewer import (
    setup_interview,
    chat_with_groq,
    score_interview,
    index_resume_batch,
)

router = APIRouter()


@router.post("/index-batch", response_model=BatchIndexResponse)
async def index_resume_batch_endpoint(request: BatchIndexRequest):
    try:
        result = await index_resume_batch(
            batch_id=request.batch_id,
            job_description=request.job_description,
            role=request.role,
            round_name=request.round,
            candidates=[c.model_dump() for c in request.candidates],
        )
        return BatchIndexResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch index failed: {str(e)}")


@router.post("/setup", response_model=InterviewSetupResponse)
async def interview_setup(request: InterviewSetupRequest):
    try:
        namespaces = await setup_interview(
            interview_id=request.interview_id,
            job_description=request.job_description,
            candidate_resume_text=request.candidate_resume_text,
            candidate_name=request.candidate_name,
            max_questions=request.max_questions,
            round=request.round,
        )
        return InterviewSetupResponse(
            status="ready",
            interview_id=request.interview_id,
            namespaces=namespaces,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interview setup failed: {str(e)}")


@router.post("/chat", response_model=InterviewChatResponse)
async def interview_chat(request: InterviewChatRequest):
    try:
        result = await chat_with_groq(
            interview_id=request.interview_id,
            latest_user_message=request.latest_user_message,
            question_number=request.question_number,
        )
        return InterviewChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interview chat error: {str(e)}")


@router.post("/score", response_model=InterviewScoreResponse)
async def score_interview_endpoint(request: InterviewScoreRequest):
    try:
        result = await score_interview(
            interview_id=request.interview_id,
            job_description=request.job_description,
            candidate_resume=request.candidate_resume,
        )
        return InterviewScoreResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
