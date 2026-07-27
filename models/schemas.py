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


class InterviewStartRequest(BaseModel):
    job_description: str
    candidate_resume: str
    candidate_name: str
    interview_type: str = "technical"


class ChatMessage(BaseModel):
    role: str
    content: str


class InterviewChatRequest(BaseModel):
    conversation_history: List[ChatMessage]
    job_description: str
    candidate_resume: str
    candidate_name: str
    question_number: int = 1


class InterviewChatResponse(BaseModel):
    response: str
    question_number: int
    is_finished: bool = False


class InterviewScoreRequest(BaseModel):
    conversation_history: List[ChatMessage]
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


class HealthResponse(BaseModel):
    status: str
    service: str
