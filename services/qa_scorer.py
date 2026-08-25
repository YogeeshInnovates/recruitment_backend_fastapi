import logging

from services.llm_screening import gemini_generate, extract_json

logger = logging.getLogger(__name__)

_QA_SCORE_PROMPT = """You are a strict, experienced technical interviewer evaluating a real voice interview transcript.
Candidate: {name}
Job Role: {role}
Round: {round}

Below are the interview question-answer pairs. Answers are speech-to-text transcriptions -
completely ignore grammar, spelling or speech recognition noise; judge ONLY:
- technical correctness and depth of the answer
- relevance to the question asked
- real understanding vs vague/generic talk

Score each answer 0-10 where:
0 = no answer / irrelevant / "no answer received"
1-3 = barely relevant, very shallow
4-6 = partially correct, some understanding
7-8 = good solid answer with clear points
9-10 = excellent, deep, specific answer

The TOTAL score out of 100 must reflect REAL-WORLD interview judgement:
- A candidate who answered only 1 of many questions but answered brilliantly should still score LOW overall (coverage matters), e.g. 15-25.
- A candidate who answered most questions well scores high.
- Empty or "no answer" responses earn 0.
Do NOT simply average - weight coverage and quality together like a human interviewer would.

Return ONLY valid JSON (no markdown, no commentary):
{{"items":[{{"index":<question number starting at 1>,"score":<0-10>,"remark":"<one short sentence>"}}],"total_score":<0-100>,"verdict":"<2-3 sentence honest overall assessment>"}}

INTERVIEW TRANSCRIPT:
{qa}
"""


def score_qa_pairs(candidate_name, job_title, round_name, qa_pairs):
    """qa_pairs: list of [question, answer]. Returns dict with items/total_score/verdict/success."""
    result = {"items": [], "total_score": 0, "verdict": "", "success": False}

    if not qa_pairs:
        result["verdict"] = "No questions were recorded for this interview."
        result["success"] = True
        return result

    qa_text = ""
    for i, (q, a) in enumerate(qa_pairs, start=1):
        ans = a.strip() if a else ""
        qa_text += f"Q{i}: {q}\nA{i}: {ans if ans else '(NO ANSWER GIVEN)'}\n\n"

    prompt = _QA_SCORE_PROMPT.format(
        name=candidate_name or "Unknown",
        role=job_title or "Software Engineer",
        round=round_name or "Technical Interview",
        qa=qa_text,
    )

    raw = gemini_generate(prompt, temperature=0.2, max_output_tokens=2048)
    data = extract_json(raw)
    if not data or "items" not in data:
        logger.warning("QA scoring: Gemini returned no usable JSON")
        # deterministic fallback: coverage * rough quality guess
        answered = sum(1 for _, a in qa_pairs if a and len(a.strip()) > 15)
        result["total_score"] = int(answered * 100 / len(qa_pairs))
        result["verdict"] = "AI analysis unavailable - score based on answer coverage only."
        return result

    try:
        for item in data.get("items", []):
            idx = int(item.get("index", 0))
            if 1 <= idx <= len(qa_pairs):
                q, a = qa_pairs[idx - 1]
            else:
                q, a = "", ""
            result["items"].append({
                "question": q,
                "answer": a,
                "score": max(0, min(10, int(item.get("score", 0)))),
                "remark": str(item.get("remark", "")),
            })
        result["total_score"] = max(0, min(100, int(data.get("total_score", 0))))
        result["verdict"] = str(data.get("verdict", ""))
        result["success"] = True
    except Exception as e:
        logger.error("QA scoring parse error: %s", e)
        answered = sum(1 for _, a in qa_pairs if a and len(a.strip()) > 15)
        result["total_score"] = int(answered * 100 / len(qa_pairs))
        result["verdict"] = "AI analysis incomplete - fallback score applied."
    return result
