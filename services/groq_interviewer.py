import os
import json
import httpx
from typing import List, Dict

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT_TEMPLATE = """You are a professional AI interviewer conducting a job interview for the position: {job_description}.

The candidate's resume:
{candidate_resume}

Candidate name: {candidate_name}

RULES:
1. Start by greeting the candidate warmly by name
2. Ask one question at a time
3. Base questions on the job requirements and the candidate's resume
4. Be professional but friendly
5. Ask follow-up questions based on their answers
6. Cover technical skills, experience, and behavioral questions
7. After {max_questions} questions, wrap up the interview politely
8. Keep responses conversational and natural
9. Do NOT reveal you are an AI - conduct the interview as a real interviewer would
10. Start with an ice-breaker question, then move to technical/experience questions

When you have asked enough questions (around {max_questions}), end with "Thank you for your time. The interview is now complete." and set is_finished to true.
"""


async def chat_with_groq(
    conversation_history: List[Dict],
    job_description: str,
    candidate_resume: str,
    candidate_name: str,
    question_number: int,
    max_questions: int = 10,
) -> Dict:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        job_description=job_description,
        candidate_resume=candidate_resume,
        candidate_name=candidate_name,
        max_questions=max_questions,
    )

    messages = [{"role": "system", "content": system_prompt}]

    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    if len(conversation_history) == 0:
        messages.append({
            "role": "user",
            "content": "Please start the interview now.",
        })

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
            },
        )

        if response.status_code != 200:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")

        data = response.json()
        ai_response = data["choices"][0]["message"]["content"]

        is_finished = "interview is now complete" in ai_response.lower()

        return {
            "response": ai_response,
            "question_number": question_number,
            "is_finished": is_finished,
        }


async def score_interview(
    conversation_history: List[Dict],
    job_description: str,
    candidate_resume: str,
) -> Dict:
    score_prompt = f"""Analyze this interview and provide scores.

Job Description: {job_description}
Candidate Resume: {candidate_resume}

Interview Transcript:
"""
    for msg in conversation_history:
        role_label = "INTERVIEWER" if msg["role"] == "assistant" else "CANDIDATE"
        score_prompt += f"\n{role_label}: {msg['content']}"

    score_prompt += """

Provide your analysis as JSON with these exact fields:
{
    "overall_score": <0-100>,
    "technical_score": <0-100>,
    "communication_score": <0-100>,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "recommendation": "HIRE" or "MAYBE" or "NO_HIRE",
    "summary": "brief summary of interview performance"
}
"""

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are an expert interview analyst. Respond only with valid JSON."},
                    {"role": "user", "content": score_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 1024,
            },
        )

        if response.status_code != 200:
            raise Exception(f"Groq API error during scoring: {response.status_code} - {response.text}")

        data = response.json()
        result_text = data["choices"][0]["message"]["content"]

        try:
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            result = json.loads(result_text[start:end])
        except (json.JSONDecodeError, ValueError):
            result = {
                "overall_score": 50.0,
                "technical_score": 50.0,
                "communication_score": 50.0,
                "strengths": ["Unable to parse scoring response"],
                "weaknesses": ["Unable to parse scoring response"],
                "recommendation": "MAYBE",
                "summary": "Score parsing failed. Manual review required.",
            }

        return result
