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
    InterviewReportRequest,
    InterviewReportResponse,
)
from services.report_analyzer import generate_interview_report
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


@router.post("/report", response_model=InterviewReportResponse)
async def interview_report_endpoint(request: InterviewReportRequest):
    try:
        result = generate_interview_report(
            interview_id=request.interview_id,
            job_description=request.job_description,
            candidate_resume=request.candidate_resume,
            questions=request.questions,
            answers=request.answers,
            behavior_events=request.behavior_events,
            evidence_count=request.evidence_count,
        )
        return InterviewReportResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.post("/score-qa")
async def score_qa_endpoint(request: dict):
    from services.qa_scorer import score_qa_pairs

    raw_pairs = request.get("qa_pairs") or []
    try:
        pairs = []
        for q in raw_pairs:
            if isinstance(q, dict):
                pairs.append([q.get("question", ""), q.get("answer", "")])
            elif isinstance(q, list):
                pairs.append([str(q[0]) if len(q) > 0 else "", str(q[1]) if len(q) > 1 else ""])
            else:
                pairs.append(["", ""])
    except Exception:
        raise HTTPException(status_code=400, detail="qa_pairs must be a list of {question, answer} objects")

    result = score_qa_pairs(
        candidate_name=request.get("candidate_name", ""),
        job_title=request.get("job_title", ""),
        round_name=request.get("round", ""),
        qa_pairs=pairs,
    )
    return result
