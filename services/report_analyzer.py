import json
import logging
import re
from typing import List, Optional

from services.llm_screening import gemini_generate, extract_json

logger = logging.getLogger(__name__)

_RECOMMENDATION_BY_SCORE = [
    (85, "SHORTLIST"),
    (70, "MAYBE"),
    (0, "REJECT"),
]


def _grade_answers(questions, answers, job_description):
    job_keywords = set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]{1,}", job_description.lower()))
    job_keywords.discard("the")
    job_keywords.discard("and")
    total_questions = len(questions)
    answered = sum(1 for a in answers if a and len(a.strip()) >= 40)
    if total_questions == 0:
        return 0.0, 0.0
    coverage = answered / max(total_questions, 1)
    keyword_hits = 0
    for answer in answers:
        tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]{1,}", answer.lower()))
        keyword_hits += len(tokens & job_keywords)
    depth = min(1.0, keyword_hits / max(total_questions * 3, 1))
    overall = round(min(100.0, 45 + coverage * 35 + depth * 20), 1)
    technical = round(min(100.0, 40 + coverage * 30 + depth * 30), 1)
    communication = round(min(100.0, 50 + coverage * 40), 1)
    return overall, technical, communication


def _pick_recommendation(score):
    for threshold, label in _RECOMMENDATION_BY_SCORE:
        if score >= threshold:
            return label
    return "REJECT"


def _strip_prefix(question):
    return re.sub(r"^\s*(Q\d*[.):\-]|\d+[.):\-])\s*", "", question or "").strip()


def fallback_interview_report(interview_id, questions, answers, job_description, evidence_count=0):
    overall, technical, communication = _grade_answers(questions, answers, job_description)
    strengths = ["Answered " + str(sum(1 for a in answers if a and a.strip()))
                 + " of " + str(len(questions)) + " questions substantively"]
    weaknesses = ["Could not provide answers for every question" if overall < 70 else
                  "Answers were concise and could include more depth"]
    recommendation = _pick_recommendation(overall)
    verdict_summary = (
        f"Candidate scored {overall}/100 overall. "
        f"Technical {technical}/100, communication {communication}/100. "
        f"Recommendation: {recommendation}."
    )
    return {
        "interview_id": str(interview_id),
        "overall_score": overall,
        "technical_score": technical,
        "communication_score": communication,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "topic_breakdown": {},
        "behavior_summary": _behavior_summary(evidence_count),
        "severity": "NONE",
        "suspicious_event_count": 0,
        "evidence_count": evidence_count,
        "recommendation": recommendation,
        "verdict_summary": verdict_summary,
        "final_note": "Deterministic report (LLM not available).",
        "source": "fallback",
    }


def _behavior_summary(evidence_count):
    if evidence_count <= 0:
        return "No unusual behavior captured during this interview."
    if evidence_count < 3:
        return "A few behavior flags were captured; interview mostly normal."
    return "Repeated behavior flags were captured; further human review recommended."


_REPORT_PROMPT = """You are an experienced technical interviewer producing a final interview evaluation report.
Return ONLY a valid JSON object (no markdown, no commentary) with exactly these keys:
{{
  "overall_score": <number 0-100>,
  "technical_score": <number 0-100>,
  "communication_score": <number 0-100>,
  "strengths": ["<strength>", ...],
  "weaknesses": ["<weakness>", ...],
  "topic_breakdown": {{"<topic>": <number 0-100>, ...}},
  "recommendation": "<SHORTLIST | MAYBE | REJECT>",
  "verdict_summary": "<2-4 sentence verdict>",
  "final_note": "<short closing note for the recruiter>"
}}

JOB DESCRIPTION:
{job}

CANDIDATE RESUME (summary):
{resume}

QUESTIONS AND ANSWERS:
{qa}

BEHAVIOR OBSERVATIONS (from proctoring):
{behavior}"""


def generate_interview_report(interview_id, job_description="", candidate_resume="",
                              questions=None, answers=None, behavior_events=None,
                              evidence_count=0):
    questions = questions or []
    answers = answers or []
    behavior_events = behavior_events or {}

    total_behavior = sum(int(v) for v in behavior_events.values() if isinstance(v, (int, float)))
    severity = "NONE"
    if total_behavior > 0:
        if total_behavior >= 8 or evidence_count >= 5:
            severity = "HIGH"
        elif total_behavior >= 4 or evidence_count >= 3:
            severity = "MEDIUM"
        else:
            severity = "LOW"

    try:
        qa_blocks = []
        for i, question in enumerate(questions):
            answer = answers[i] if i < len(answers) else ""
            qa_blocks.append(f"Q{i + 1}: {_strip_prefix(question)}\nA{i + 1}: {answer}")
        qa_text = "\n\n".join(qa_blocks) or "No Q&A provided."

        behavior_text = (
            "Proctoring flags by event type: " + json.dumps(behavior_events)
            + f"; evidence snapshots captured: {evidence_count}."
            if total_behavior else f"No behavior flags. Evidence snapshots: {evidence_count}."
        )

        prompt = _REPORT_PROMPT.format(
            job=job_description or "Not provided",
            resume=(candidate_resume or "")[:4000],
            qa=qa_text[:12000],
            behavior=behavior_text,
        )
        raw = gemini_generate(prompt, temperature=0.3, max_output_tokens=4096)
        data = extract_json(raw)
        if data and "overall_score" in data:
            def clamp_score(value):
                try:
                    return max(0.0, min(100.0, float(value)))
                except (TypeError, ValueError):
                    return 0.0

            overall = clamp_score(data.get("overall_score"))
            technical = clamp_score(data.get("technical_score"))
            communication = clamp_score(data.get("communication_score"))
            recommendation = str(data.get("recommendation") or _pick_recommendation(overall))
            strengths = [str(s) for s in (data.get("strengths") or [])]
            weaknesses = [str(s) for s in (data.get("weaknesses") or [])]
            topic_breakdown = data.get("topic_breakdown") or {}
            if isinstance(topic_breakdown, list):
                topic_breakdown = {}
            return {
                "interview_id": str(interview_id),
                "overall_score": overall,
                "technical_score": technical,
                "communication_score": communication,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "topic_breakdown": topic_breakdown,
                "behavior_summary": _behavior_summary(evidence_count),
                "severity": severity,
                "suspicious_event_count": total_behavior,
                "evidence_count": evidence_count,
                "recommendation": recommendation,
                "verdict_summary": str(data.get("verdict_summary") or ""),
                "final_note": str(data.get("final_note") or ""),
                "source": "gemini",
            }
        logger.info("Gemini report returned no usable JSON; using deterministic fallback.")
    except Exception as e:
        logger.warning("LLM report failed (%s); using deterministic fallback.", e)

    report = fallback_interview_report(
        interview_id, questions, answers, job_description, evidence_count
    )
    report["behavior_summary"] = _behavior_summary(evidence_count)
    report["severity"] = severity
    report["suspicious_event_count"] = total_behavior
    return report
