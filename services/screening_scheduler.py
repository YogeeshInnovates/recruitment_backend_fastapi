import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

SCREENING_ZONE = ZoneInfo("Asia/Kolkata")
FIRST_SLOT_DELAY_MINUTES = 8
GAP_MINUTES = 8


def compute_schedule(
    ranked: List[Dict[str, Any]],
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Pre-calculate fixed interview slots for ranked candidates.

    Rules:
      - first slot starts 8 minutes from now (demo-friendly; no waiting
        until the next morning working window)
      - back-to-back slots with an 8-minute gap between allocations
      - slot length = the candidate's interview_duration_minutes
      - each candidate starts one-by-one based on their experience level,
        so a candidate who finishes early does NOT trigger the next one early
    """
    if now is None:
        now = datetime.now(SCREENING_ZONE)
    else:
        now = now.astimezone(SCREENING_ZONE)

    base = (now + timedelta(minutes=FIRST_SLOT_DELAY_MINUTES)).replace(second=0, microsecond=0)
    cursor = base
    slots: List[Dict[str, Any]] = []

    for item in ranked:
        duration = int(item.get("interview_duration_minutes") or 25)
        if duration <= 0:
            duration = 25

        start = cursor
        end = start + timedelta(minutes=duration)

        slots.append({
            "candidate_id": item.get("candidate_id", ""),
            "candidate_name": item.get("candidate_name", ""),
            "rank": item.get("rank", 0),
            "score": item.get("score", 0),
            "date": start.strftime("%A, %B %d, %Y"),
            "start_time": start.strftime("%I:%M %p").lstrip("0"),
            "end_time": end.strftime("%I:%M %p").lstrip("0"),
            "start_at": start.isoformat(),
            "duration_minutes": duration,
        })

        cursor = end + timedelta(minutes=GAP_MINUTES)

    logger.info("Computed %d screening slots starting at %s", len(slots), slots[0]["start_at"] if slots else "n/a")
    return slots