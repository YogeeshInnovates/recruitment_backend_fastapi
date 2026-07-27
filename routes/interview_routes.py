from fastapi import APIRouter, HTTPException
from models.schemas import (
    InterviewChatRequest,
    InterviewChatResponse,
    InterviewScoreRequest,
    InterviewScoreResponse,
)
from services.groq_interviewer import chat_with_groq, score_interview

router = APIRouter()


@router.post("/chat", response_model=InterviewChatResponse)
async def interview_chat(request: InterviewChatRequest):
    try:
        history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]
        result = await chat_with_groq(
            conversation_history=history,
            job_description=request.job_description,
            candidate_resume=request.candidate_resume,
            candidate_name=request.candidate_name,
            question_number=request.question_number,
        )
        return InterviewChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interview chat error: {str(e)}")


@router.post("/score", response_model=InterviewScoreResponse)
async def score_interview_endpoint(request: InterviewScoreRequest):
    try:
        history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]
        result = await score_interview(
            conversation_history=history,
            job_description=request.job_description,
            candidate_resume=request.candidate_resume,
        )
        return InterviewScoreResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
