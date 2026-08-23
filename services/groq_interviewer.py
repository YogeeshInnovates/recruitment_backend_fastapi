import os
import re
import json
import logging
import asyncio
import httpx
from typing import List, Dict, Any, Optional, TypedDict, Annotated

import tiktoken
from pinecone import Pinecone, SearchQuery

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

try:
    from langgraph import MemorySaver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_ACCOUNT_KEYS_ENV = os.getenv("GROQ_ACCOUNT_KEYS", "")
if GROQ_ACCOUNT_KEYS_ENV:
    GROQ_KEYS = [k.strip() for k in GROQ_ACCOUNT_KEYS_ENV.split(",") if k.strip()]
else:
    fallback_key = os.getenv("GROQ_API_KEY", "")
    GROQ_KEYS = [fallback_key] if fallback_key else []

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "interview-vectors")

SPRING_BOOT_CALLBACK_URL = os.getenv(
    "SPRING_BOOT_CALLBACK_URL",
    "http://host.docker.internal:8080/api/interview/callback",
)

SLIDING_WINDOW_TURNS = 3

# ---------------------------------------------------------------------------
# Global lazy-init clients
# ---------------------------------------------------------------------------
_pinecone_client: Optional[Pinecone] = None
_pinecone_index = None


def get_pinecone_index():
    global _pinecone_client, _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    _pinecone_index = _pinecone_client.Index(PINECONE_INDEX_NAME)
    return _pinecone_index


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
_tokenizer = tiktoken.get_encoding("cl100k_base")


def num_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def chunk_text(text: str, max_tokens: int = 500) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks: List[str] = []
    buffer: List[str] = []
    buffer_size = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = num_tokens(para)
        if para_tokens > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent_tokens = num_tokens(sent)
                if buffer_size + sent_tokens > max_tokens and buffer:
                    chunks.append(" ".join(buffer))
                    buffer = []
                    buffer_size = 0
                buffer.append(sent)
                buffer_size += sent_tokens
        else:
            if buffer_size + para_tokens > max_tokens and buffer:
                chunks.append(" ".join(buffer))
                buffer = []
                buffer_size = 0
            buffer.append(para)
            buffer_size += para_tokens
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks if chunks else [text[:max_tokens * 4]]


# ---------------------------------------------------------------------------
# Multi-key LLM invocation with rotation
# ---------------------------------------------------------------------------
def get_llm_instance(api_key: str, temperature: float, max_tokens: Optional[int] = None):
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=25,
        reasoning_format="hidden",
    )


async def ainvoke_with_rotation(
    messages: List[BaseMessage],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> AIMessage:
    if not GROQ_KEYS:
        raise RuntimeError("No GROQ API keys configured.")
    last_exception: Optional[Exception] = None
    for idx, api_key in enumerate(GROQ_KEYS):
        try:
            llm = get_llm_instance(api_key, temperature, max_tokens)
            return await llm.ainvoke(messages)
        except Exception as e:
            last_exception = e
            masked = api_key[:8] + "..." if len(api_key) > 8 else "***"
            logger.warning("Groq key %d (%s) failed: %s. Rotating...", idx, masked, e)
            continue
    logger.error("All Groq keys exhausted. Final: %s", last_exception)
    raise last_exception


# ---------------------------------------------------------------------------
# Pinecone ingestion pipeline (called once at setup)
# ---------------------------------------------------------------------------
async def ingest_interview_context(
    interview_id: str,
    job_description: str,
    candidate_resume_text: str,
) -> List[str]:
    combined = (
        f"[JOB DESCRIPTION]\n{job_description}\n\n"
        f"[RESUME]\n{candidate_resume_text}"
    )
    chunks = chunk_text(combined, max_tokens=500)
    logger.info("Ingested %d chunks for interview %s", len(chunks), interview_id)

    index = get_pinecone_index()
    namespace = f"interview_{interview_id}"

    records = [
        {
            "_id": f"chunk-{i}",
            "text": chunks[i][:1500],
            "chunk_index": i,
            "source": "interview_context",
        }
        for i in range(len(chunks))
    ]

    index.upsert_records(namespace=namespace, records=records)
    logger.info("Ingested %d records into namespace %s", len(records), namespace)
    return [namespace]


async def query_relevant_chunks(
    interview_id: str,
    query_text: str,
    k: int = 2,
) -> List[str]:
    index = get_pinecone_index()
    namespace = f"interview_{interview_id}"
    results = index.search_records(
        namespace=namespace,
        query=SearchQuery(
            inputs={"text": query_text},
            top_k=k,
        ),
        fields=["text"],
    )
    chunks = []
    for hit in results.result.hits:
        if "text" in hit.get("fields", {}):
            chunks.append(hit["fields"]["text"])
    return chunks


async def delete_interview_namespace(interview_id: str) -> None:
    index = get_pinecone_index()
    namespace = f"interview_{interview_id}"
    try:
        index.delete(delete_all=True, namespace=namespace)
        logger.info("Cleaned up Pinecone namespace %s", namespace)
    except Exception as e:
        logger.warning("Failed to clean up namespace %s: %s", namespace, e)


# ---------------------------------------------------------------------------
# Batch resume ingestion (multi-resume setup, keeps strict candidate mapping)
# ---------------------------------------------------------------------------
async def index_resume_batch(
    batch_id: str,
    job_description: str,
    role: str,
    round_name: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    index = get_pinecone_index()
    namespace = f"batch_{batch_id}"
    records: List[Dict[str, Any]] = []

    jd_chunks = chunk_text(job_description or "", max_tokens=500)
    for i, chunk in enumerate(jd_chunks):
        records.append({
            "_id": f"job-{i}",
            "text": chunk[:1500],
            "chunk_index": i,
            "source": "job",
            "batch_id": batch_id,
            "role": role or "",
            "round": round_name or "",
        })

    per_candidate: List[Dict[str, Any]] = []
    for cand in candidates:
        candidate_id = str(cand.get("candidate_id", ""))
        name = cand.get("name", "")
        email = cand.get("email", "")
        resume_text = cand.get("resume_text", "")
        chunks = chunk_text(resume_text, max_tokens=500)
        added = 0
        for i, chunk in enumerate(chunks):
            records.append({
                "_id": f"candidate-{candidate_id}-{i}",
                "text": chunk[:1500],
                "chunk_index": i,
                "source": "resume",
                "candidate_id": candidate_id,
                "candidate_name": name,
                "candidate_email": email,
                "batch_id": batch_id,
                "role": role or "",
                "round": round_name or "",
            })
            added += 1
        per_candidate.append({
            "candidate_id": candidate_id,
            "name": name,
            "chunks_indexed": added,
        })
        logger.info("Indexed %d chunks for candidate %s (%s)", added, candidate_id, name)

    if records:
        index.upsert_records(namespace=namespace, records=records)
        logger.info("Indexed batch %s: %d records into namespace %s", batch_id, len(records), namespace)

    return {
        "status": "indexed",
        "namespace": namespace,
        "job_chunks": len(jd_chunks),
        "candidates": per_candidate,
    }


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------
class InterviewState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    interview_id: str
    job_description: str
    question_number: int
    current_difficulty: str
    running_score: int
    question_scores: List[int]
    max_questions: int
    is_finished: bool
    round: str


# ---------------------------------------------------------------------------
# Node: interviewer (JIT RAG injector)
# ---------------------------------------------------------------------------
COMPACT_SYSTEM_TEMPLATE = """You are a strict AI interviewer. Ask questions only. Never answer them.

JOB: {job_description}
CANDIDATE (from resume): {rag_context}

INTERVIEW FLOW - follow these phases naturally:
Phase 1 (Q1-2): Ask about the candidate's background, experience from their resume.
Phase 2 (Q3-5): Deep-dive into specific projects and technical skills listed on resume.
Phase 3 (Q6-8): Ask job-description specific questions — scenario-based, real-world problems.
Phase 4 (Q9+): Advanced/challenging questions. Push to see depth of knowledge.

STRICT RULES:
1. ONE short specific question at a time (max 2 sentences, under 25 words).
2. Greet warmly on Q1 only.
3. After answer, acknowledge briefly (1-3 words like "Got it" or "Thanks"), then ask the next question immediately.
4. NEVER explain, show code, give examples, or answer your own question.
5. NEVER give hints or corrections.
6. If candidate says "ask me to repeat": say "Could you please repeat your answer?"
7. If candidate says "let me ask something else": say "I see, let's move on" and ask next question.
8. If candidate says "end the interview" or doesn't respond: say "That's alright. Thank you for your time. The interview is now complete."
9. If candidate says "I don't know": say "That's okay, let's move on."
10. End naturally: "Thank you for your time. We'll review your responses and get back to you." when all phases feel covered."""


HR_ROUND_TEMPLATE = """You are a warm, professional HUMAN RESOURCES interviewer conducting an initial HR screening round. Ask questions only. Never answer them.

JOB: {job_description}
CANDIDATE (from resume): {rag_context}

HR INTERVIEW FLOW - follow these phases naturally:
Phase 1 (Q1-2): Introductions - greet warmly, ask the candidate to introduce themselves and walk through their background and experience from their resume.
Phase 2 (Q3-5): Motivation and character - why this role/company, career goals, strengths and weaknesses, how they handle pressure and feedback.
Phase 3 (Q6-8): Behavioral and situational questions - teamwork, conflict, communication, adaptability, past experiences.
Phase 4 (Q9+): Logistics and closing - notice period, expected salary, current location, work mode preference; then close warmly.

STRICT RULES:
1. ONE short, conversational question at a time (max 2 sentences).
2. Greet warmly on Q1 only.
3. After an answer, acknowledge naturally ("Got it", "That's great", "I see") then ask the next question immediately.
4. NEVER ask technical, coding, or hard-skill questions in this HR round.
5. NEVER give answers, hints, or corrections.
6. Keep the tone friendly, professional, and human - talk like a real HR person getting to know a candidate's character.
7. If candidate says "end the interview" or doesn't respond: say "That's alright. Thank you for your time. The interview is now complete."
8. End naturally: "Thank you for your time. We'll review your responses and get back to you soon." when all phases feel covered."""


MANAGERIAL_ROUND_TEMPLATE = """You are a senior interviewer conducting a MANAGERIAL round for a leadership/people-management role. Ask questions only. Never answer them.

JOB: {job_description}
CANDIDATE (from resume): {rag_context}

MANAGERIAL INTERVIEW FLOW - follow these phases naturally:
Phase 1 (Q1-2): Leadership background - teams managed, management experience from their resume.
Phase 2 (Q3-5): People management scenarios - delegating, mentoring, conflict resolution, giving feedback, hiring/firing.
Phase 3 (Q6-8): Strategic and cross-functional questions - aligning the team with business goals, driving results, decision-making.
Phase 4 (Q9+): Advanced leadership challenges and culture fit.

STRICT RULES:
1. ONE short, focused question at a time (max 2 sentences).
2. Greet warmly on Q1 only.
3. After an answer, acknowledge briefly then ask the next question immediately.
4. NEVER answer your own questions or give hints.
5. Focus on leadership, ownership, teamwork, and business impact - not pure coding.
6. If candidate says "end the interview" or doesn't respond: say "That's alright. Thank you for your time. The interview is now complete."
7. End naturally: "Thank you for your time. We'll review your responses and get back to you soon." when all phases feel covered."""


def _round_kind(round_name: str) -> str:
    r = (round_name or "").lower()
    if "hr" in r or "human resource" in r or "screening" in r or "fitment" in r:
        return "hr"
    if "manager" in r or "leadership" in r or "lead" in r:
        return "managerial"
    return "technical"


async def call_interviewer(state: InterviewState) -> Dict[str, Any]:
    messages = state["messages"]
    if not messages:
        return {"messages": [], "question_number": 0}

    latest_msg = messages[-1].content if messages else ""
    interview_id = state.get("interview_id", "")
    q_num = state["question_number"] + 1

    if q_num <= 2:
        phase = "1 - Resume warm-up"
    elif q_num <= 5:
        phase = "2 - Skills deep-dive"
    elif q_num <= 8:
        phase = "3 - Job scenarios"
    else:
        phase = "4 - Advanced challenges"

    rag_chunks = await query_relevant_chunks(
        interview_id,
        latest_msg,
        k=2,
    )
    rag_context = "\n\n".join(rag_chunks) if rag_chunks else "No specific context retrieved."
    job_desc = state.get("job_description", "")

    round_kind = _round_kind(state.get("round", ""))
    if round_kind == "hr":
        system = HR_ROUND_TEMPLATE.format(
            job_description=job_desc[:800],
            rag_context=rag_context[:1000],
        )
    elif round_kind == "managerial":
        system = MANAGERIAL_ROUND_TEMPLATE.format(
            job_description=job_desc[:800],
            rag_context=rag_context[:1000],
        )
    elif state["question_number"] <= 5:
        system = COMPACT_SYSTEM_TEMPLATE.format(
            job_description=job_desc[:800],
            rag_context=rag_context[:1000],
        )
    else:
        system = (
            f"You are a strict AI interviewer. Ask questions only.\n"
            f"Context: {rag_context[:400]}\n"
            f"Rules: 1) One short question at a time. 2) After answer, acknowledge briefly then ask next question immediately. 3) Never give code or explanations."
        )

    truncated = apply_sliding_window(messages, turns=SLIDING_WINDOW_TURNS)

    prompt = [SystemMessage(content=system)] + truncated

    response: AIMessage = await ainvoke_with_rotation(prompt, temperature=0.7)

    cleaned = strip_thinking(response.content)
    response = AIMessage(content=cleaned)

    return {
        "messages": [response],
        "question_number": state["question_number"] + 1,
    }


def apply_sliding_window(messages: List[BaseMessage], turns: int = 3) -> List[BaseMessage]:
    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_indices) <= turns:
        return messages[-max(1, len(messages)):]
    cutoff = human_indices[-turns]
    return messages[cutoff:]


def strip_thinking(text: str) -> str:
    if isinstance(text, list):
        text = " ".join(str(part) for part in text)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<\|thought\|>.*?<\|/thought\|>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"===.*?===", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\*{3,}.*?\*{3,}", "", cleaned, flags=re.DOTALL)
    if not cleaned.strip():
        return "Got it."
    return cleaned.strip()


def _get_latest_qa(messages: List[BaseMessage]) -> tuple:
    answer = None
    answer_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            answer = messages[i].content
            answer_idx = i
            break
    if answer is None or answer_idx is None:
        return None, None
    for j in range(answer_idx - 1, -1, -1):
        if isinstance(messages[j], AIMessage):
            return messages[j].content, answer
    return None, answer


ANSWER_EVALUATION_TEMPLATE = """You are a fair, strict technical interviewer evaluating a candidate's answer during a job interview.

INTERVIEWER QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

Evaluate how well the candidate answered. Judge:
- Correctness: is the technical content accurate?
- Completeness: did they cover the main points a good answer should cover?
- Clarity: is it understandable and coherent?

RULES:
- Give PARTIAL CREDIT. A partially correct answer (e.g. only 2 of 5 key points) must get a proportional score, NEVER 0.
- A short but accurate and complete answer should score well. Length is NOT a factor.
- Do NOT penalize for brevity. Do NOT reward rambling.
- Score 0 = completely wrong/irrelevant. Score 10 = excellent, comprehensive, accurate.

Respond with ONLY valid JSON: {{"score": <integer 0-10>, "difficulty": "Easy"|"Medium"|"Hard", "reason": "<one short sentence>"}}"""


async def _evaluate_answer_llm(question: str, answer: str) -> Dict[str, Any]:
    prompt = ANSWER_EVALUATION_TEMPLATE.format(question=question, answer=answer)
    response = await ainvoke_with_rotation([HumanMessage(content=prompt)], temperature=0.2)
    text = strip_thinking(response.content)
    start = text.find("{")
    end = text.rfind("}") + 1
    parsed = json.loads(text[start:end]) if start != -1 and end > start else {}
    if "score" not in parsed:
        raise ValueError(f"Evaluation response missing 'score': {text[:200]}")
    return parsed


# ---------------------------------------------------------------------------
# Node: evaluator (per-answer human-like scoring + adaptive difficulty + cleanup)
# ---------------------------------------------------------------------------
async def evaluator(state: InterviewState) -> Dict[str, Any]:
    messages = state["messages"]
    current_difficulty = state.get("current_difficulty", "Medium")
    running_score = state.get("running_score", 0)
    question_scores = list(state.get("question_scores", []))

    is_finished = False

    if state["question_number"] < 6:
        is_finished = False
    elif len(messages) >= 1 and state["question_number"] >= 6:
        last_ai = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                last_ai = m.content.lower()
                break
        if "thank you for your time" in last_ai or "interview is now complete" in last_ai or "we'll" in last_ai:
            is_finished = True

    if not is_finished:
        last_user = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user = m.content.lower()
                break
        if "end the interview" in last_user or "stop the interview" in last_user:
            is_finished = True

    if not is_finished:
        question, answer = _get_latest_qa(messages)
        if question and answer:
            try:
                evaluation = await _evaluate_answer_llm(question, answer)
                score = evaluation.get("score")
                if score is not None:
                    score = max(0, min(10, int(round(float(score)))))
                    running_score = min(state.get("max_questions", 15) * 10, running_score + score)
                    question_scores.append(score)
                if evaluation.get("difficulty") in ("Easy", "Medium", "Hard"):
                    current_difficulty = evaluation["difficulty"]
            except Exception as e:
                logger.warning("Per-answer evaluation failed for interview %s: %s", state.get("interview_id"), e)

    return {
        "current_difficulty": current_difficulty,
        "running_score": running_score,
        "question_scores": question_scores,
        "is_finished": is_finished,
    }


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def build_interview_graph() -> StateGraph:
    workflow = StateGraph(InterviewState)
    workflow.add_node("interviewer", call_interviewer)
    workflow.add_node("evaluator", evaluator)
    workflow.add_edge(START, "interviewer")
    workflow.add_edge("interviewer", "evaluator")
    workflow.add_edge("evaluator", END)
    return workflow


memory_saver = MemorySaver()
interview_agent = build_interview_graph().compile(checkpointer=memory_saver)


# ---------------------------------------------------------------------------
# Public API: setup_interview
# ---------------------------------------------------------------------------
async def setup_interview(
    interview_id: str,
    job_description: str,
    candidate_resume_text: str,
    candidate_name: str,
    max_questions: int = 15,
    round: str = "Technical Round 1",
) -> List[str]:
    namespaces = await ingest_interview_context(
        interview_id=interview_id,
        job_description=job_description,
        candidate_resume_text=candidate_resume_text,
    )
    agent_config = {"configurable": {"thread_id": interview_id}}
    initial_state: InterviewState = {
        "messages": [
            HumanMessage(content=f"Candidate name: {candidate_name}. Please start the interview.")
        ],
        "interview_id": interview_id,
        "job_description": job_description[:800],
        "question_number": 0,
        "current_difficulty": "Medium",
        "running_score": 0,
        "question_scores": [],
        "max_questions": max_questions,
        "is_finished": False,
        "round": round,
    }
    await interview_agent.ainvoke(initial_state, config=agent_config)
    return namespaces


# ---------------------------------------------------------------------------
# Public API: chat_with_groq (subsequent turns)
# ---------------------------------------------------------------------------
async def chat_with_groq(
    interview_id: str,
    latest_user_message: str,
    question_number: int,
) -> Dict[str, Any]:
    config = {"configurable": {"thread_id": interview_id}}

    state = interview_agent.get_state(config)
    if not state or not state.values:
        logger.warning("No session for %s — re-creating from scratch", interview_id)
        fresh_state: InterviewState = {
            "messages": [],
            "interview_id": interview_id,
            "job_description": "",
            "question_number": question_number,
            "current_difficulty": "Medium",
            "running_score": 0,
            "question_scores": [],
            "max_questions": 15,
            "is_finished": False,
            "round": "Technical Round 1",
        }
        await interview_agent.ainvoke(fresh_state, config=config)
        state = interview_agent.get_state(config)

    current = state.values
    user_msg = HumanMessage(content=latest_user_message)

    inputs: InterviewState = {
        "messages": [user_msg],
        "interview_id": current.get("interview_id", interview_id),
        "job_description": current.get("job_description", ""),
        "question_number": current.get("question_number", question_number),
        "current_difficulty": current.get("current_difficulty", "Medium"),
        "running_score": current.get("running_score", 0),
        "question_scores": list(current.get("question_scores", [])),
        "max_questions": current.get("max_questions", 15),
        "is_finished": current.get("is_finished", False),
        "round": current.get("round", "Technical Round 1"),
    }

    result = await interview_agent.ainvoke(inputs, config=config)

    ai_response = result["messages"][-1].content if result.get("messages") else ""
    ai_response = strip_thinking(ai_response)
    updated_qn = result.get("question_number", question_number)
    difficulty = result.get("current_difficulty", "Medium")
    score = result.get("running_score", 0)
    is_finished = result.get("is_finished", False)

    if is_finished:
        asyncio.ensure_future(delete_interview_namespace(interview_id))
        asyncio.ensure_future(_auto_score_and_callback(interview_id, result))

    return {
        "response": ai_response,
        "question_number": updated_qn,
        "current_difficulty": difficulty,
        "running_score": score,
        "is_finished": is_finished,
    }


async def _auto_score_and_callback(interview_id: str, final_state: Dict[str, Any]) -> None:
    try:
        messages = final_state.get("messages", [])
        transcript_lines = []
        for m in messages:
            if isinstance(m, HumanMessage):
                transcript_lines.append(f"CANDIDATE: {m.content}")
            elif isinstance(m, AIMessage):
                transcript_lines.append(f"INTERVIEWER: {m.content}")
        transcript = "\n".join(transcript_lines)

        analysis = await _run_scoring(transcript)

        question_scores = list(final_state.get("question_scores", []))
        accumulated = final_state.get("running_score", 0)
        normalized = None
        if question_scores:
            normalized = round((sum(question_scores) / len(question_scores)) * 10)

        holistic_overall = None
        try:
            raw = analysis.get("overall_score")
            if raw is not None:
                holistic_overall = float(raw)
        except (TypeError, ValueError):
            holistic_overall = None

        if normalized is not None and holistic_overall is not None:
            overall_score = int(round(0.7 * normalized + 0.3 * holistic_overall))
        elif normalized is not None:
            overall_score = normalized
        elif holistic_overall is not None:
            overall_score = int(round(holistic_overall))
        else:
            overall_score = 50

        payload = {
            "interview_id": interview_id,
            "transcript_sheet": transcript,
            "overall_score": overall_score,
            "question_scores": question_scores,
            "accumulated_score": accumulated,
            **analysis,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                SPRING_BOOT_CALLBACK_URL,
                json=payload,
                timeout=30.0,
            )
            if resp.status_code == 200:
                logger.info("Scoring callback succeeded for %s", interview_id)
                try:
                    memory_saver.delete_thread(interview_id)
                    logger.info("Purged MemorySaver state for %s", interview_id)
                except Exception as purge_err:
                    logger.warning("Purge failed for %s: %s", interview_id, purge_err)
            else:
                logger.error("Scoring callback failed: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logger.error("Auto scoring/callback failed for %s: %s", interview_id, e)


async def _run_scoring(transcript: str) -> Dict[str, Any]:
    score_prompt = f"""Analyze this interview transcript and return feedback as valid JSON with these exact fields: overall_score (0-100), technical_score (0-100), communication_score (0-100), strengths (list), weaknesses (list), recommendation (HIRE/MAYBE/NO_HIRE), summary (string).

Transcript:
{transcript}

Respond only with valid JSON."""
    try:
        response = await ainvoke_with_rotation(
            [HumanMessage(content=score_prompt)],
            temperature=0.3,
        )
        import json as _json
        text = response.content
        start = text.find("{")
        end = text.rfind("}") + 1
        return _json.loads(text[start:end])
    except Exception as e:
        logger.warning("Scoring parse failed: %s. Returning defaults.", e)
        return {
            "overall_score": 50, "technical_score": 50, "communication_score": 50,
            "strengths": ["Default"], "weaknesses": ["Default"],
            "recommendation": "MAYBE", "summary": "Auto-scoring failed.",
        }


# ---------------------------------------------------------------------------
# Public API: score_interview (explicit call)
# ---------------------------------------------------------------------------
async def score_interview(
    interview_id: str,
    job_description: str = "",
    candidate_resume: str = "",
) -> Dict[str, Any]:
    config = {"configurable": {"thread_id": interview_id}}
    state = interview_agent.get_state(config)
    if not state or not state.values:
        raise RuntimeError(f"No interview session found for {interview_id}.")
    messages = state.values.get("messages", [])
    transcript_lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            transcript_lines.append(f"CANDIDATE: {m.content}")
        elif isinstance(m, AIMessage):
            transcript_lines.append(f"INTERVIEWER: {m.content}")
    transcript = "\n".join(transcript_lines)
    return await _run_scoring(transcript)
