from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ResumeScreenRequest(BaseModel):
    resume_text: str
    job_description: str


class ResumeScreenResponse(BaseModel):
    score: float
    analysis: str
    matched_skills: List[str]
    missing_skills: List[str]


class JobItem(BaseModel):
    job_id: str
    title: str
    description: str
    required_skills: str


class JobMatchRequest(BaseModel):
    resume_text: str
    jobs: List[JobItem]


class JobMatchItem(BaseModel):
    job_id: str
    title: str
    score: float
    reason: str


class JobMatchResponse(BaseModel):
    matches: List[JobMatchItem]


class ApplicationAnalyzeRequest(BaseModel):
    resume_text: str
    job_description: str
    cover_letter: str


class ApplicationAnalyzeResponse(BaseModel):
    overall_score: float
    resume_score: float
    cover_letter_score: float
    recommendation: str
    strengths: List[str]
    weaknesses: List[str]
    summary: str


class InterviewSetupRequest(BaseModel):
    interview_id: str
    job_description: str
    candidate_resume_text: str
    candidate_name: str
    max_questions: int = 15


class InterviewSetupResponse(BaseModel):
    status: str
    interview_id: str
    namespaces: List[str]


class InterviewChatRequest(BaseModel):
    interview_id: str
    latest_user_message: str
    question_number: int = 1


class InterviewChatResponse(BaseModel):
    response: str
    question_number: int
    current_difficulty: str
    running_score: int
    is_finished: bool = False


class InterviewScoreRequest(BaseModel):
    interview_id: str
    job_description: str
    candidate_resume: str


class InterviewScoreResponse(BaseModel):
    overall_score: float
    technical_score: float
    communication_score: float
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str
    summary: str


class CandidateResumeItem(BaseModel):
    candidate_id: str
    name: str
    email: str
    resume_text: str


class BatchIndexRequest(BaseModel):
    batch_id: str
    job_description: str
    role: str
    round: str
    candidates: List[CandidateResumeItem]


class BatchIndexCandidateResult(BaseModel):
    candidate_id: str
    name: str
    chunks_indexed: int


class BatchIndexResponse(BaseModel):
    status: str
    namespace: str
    job_chunks: int
    candidates: List[BatchIndexCandidateResult]


class HealthResponse(BaseModel):
    status: str
    service: str
