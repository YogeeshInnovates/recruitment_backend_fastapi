import logging
from datetime import datetime, timedelta, time as dtime
from typing import List, Dict, Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

SCREENING_ZONE = ZoneInfo("Asia/Kolkata")
WINDOW_START = dtime(10, 0)
WINDOW_END = dtime(19, 0)
GAP_MINUTES = 5


def _next_window_start(now: datetime) -> datetime:
    today_10 = datetime.combine(now.date(), WINDOW_START, tzinfo=SCREENING_ZONE)
    if now.hour < WINDOW_START.hour:
        return today_10
    return today_10 + timedelta(days=1)


def compute_schedule(
    ranked: List[Dict[str, Any]],
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Pre-calculate fixed interview slots for ranked candidates.

    Rules:
      - back-to-back slots with a 5-minute gap
      - slot length = the candidate's interview_duration_minutes
      - working window 10:00 AM - 7:00 PM in Asia/Kolkata
      - if a slot would start or extend past 7:00 PM, it (and everyone after
        in ranked order) rolls over to the next day starting at 10:00 AM
    """
    if now is None:
        now = datetime.now(SCREENING_ZONE)
    else:
        now = now.astimezone(SCREENING_ZONE)

    cursor = _next_window_start(now)
    slots: List[Dict[str, Any]] = []

    for item in ranked:
        duration = int(item.get("interview_duration_minutes") or 25)
        if duration <= 0:
            duration = 25

        while True:
            start = cursor
            end = start + timedelta(minutes=duration)
            day_windows_end = datetime.combine(start.date(), WINDOW_END, tzinfo=SCREENING_ZONE)
            if start.time() >= WINDOW_END or end > day_windows_end:
                cursor = datetime.combine(start.date() + timedelta(days=1), WINDOW_START, tzinfo=SCREENING_ZONE)
                continue
            break

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