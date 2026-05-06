import json
from datetime import datetime, timezone, timedelta
from google import genai
from config import GEMINI_API_KEY
import supabase_client

_client = genai.Client(api_key=GEMINI_API_KEY)

_SUMMARY_PROMPT = """
You are a professional safeguarding and SEL (Social Emotional Learning) analyst
reviewing a child's emotional check-in data for the past 7 days.

SESSION: {session_id}
PERIOD: {period_start} to {period_end}
TOTAL CHECK-INS: {entry_count}

DAILY CHECK-IN DATA:
{entries_block}

SAFEGUARDING ALERTS THIS PERIOD: {alert_count}
{alerts_block}

Generate a structured teacher-facing weekly emotional summary.
Respond with ONLY a valid JSON object (no markdown, no extra text) with these exact keys:

- "dominant_emotions": list of strings, e.g. ["joy (3)", "sadness (2)"] — up to 3 most frequent
- "sentiment_arc": one of "improving", "declining", "stable", "mixed"
- "arc_explanation": 1-sentence explanation of the arc classification
- "avg_drift_score": average drift_score as a float, or null if no drift data
- "drift_concern": boolean — true if avg drift > 0.6 or any single drift > 0.85
- "casel_patterns": list of strings, e.g. ["self-awareness (4)", "self-management (2)"]
- "safeguarding_highlights": string — brief summary of any alerts, or "None this week"
- "narrative_summary": 3-4 sentence teacher-facing paragraph covering overall emotional
  wellbeing, notable patterns, and context
- "recommended_action": one of "no_action", "monitor_closely", "schedule_checkin", "escalate"
- "recommended_action_rationale": 1-sentence rationale for the recommended action
"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _fetch_weekly_data(session_id: str) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    all_entries = supabase_client.get_history(session_id)
    entries = [
        e for e in all_entries
        if e.get("created_at") and datetime.fromisoformat(
            e["created_at"].replace("Z", "+00:00")
        ) >= cutoff
    ]
    all_alerts = supabase_client.get_session_alerts(session_id)
    alerts = [
        a for a in all_alerts
        if a.get("triggered_at") and datetime.fromisoformat(
            a["triggered_at"].replace("Z", "+00:00")
        ) >= cutoff
    ]
    period_start = entries[0]["created_at"][:10] if entries else ""
    period_end = entries[-1]["created_at"][:10] if entries else ""
    return {
        "entries": entries,
        "period_start": period_start,
        "period_end": period_end,
        "alert_count": len(alerts),
        "alerts": alerts,
    }


def _format_entries_block(entries: list[dict]) -> str:
    lines = []
    for e in entries:
        date = e.get("created_at", "")[:10]
        emotion = e.get("emotion", "unknown")
        sentiment = e.get("sentiment", "unknown")
        score = e.get("sentiment_score", 0.0)
        drift = e.get("drift_score")
        drift_str = f"{drift:.3f}" if drift is not None else "n/a"
        transcript = (e.get("transcript") or e.get("user_input", ""))[:80]
        lines.append(
            f'[{date}] emotion={emotion} | sentiment={sentiment} (score={score:.2f}) '
            f'| drift={drift_str} | transcript="{transcript}"'
        )
    return "\n".join(lines) if lines else "(no entries)"


def _format_alerts_block(alerts: list[dict]) -> str:
    if not alerts:
        return "(none)"
    lines = []
    for a in alerts:
        date = a.get("triggered_at", "")[:10]
        severity = a.get("severity", "unknown").upper()
        reason = a.get("reason", "")
        ack = "acknowledged" if a.get("acknowledged_at") else "unacknowledged"
        lines.append(f"[{date}] severity={severity} | {ack} | reason={reason}")
    return "\n".join(lines)


def generate_weekly_summary(session_id: str) -> dict:
    """Generate a weekly emotional summary for a session and save it to Supabase.

    Returns a dict with all summary fields, or {"error": "..."} if no data found.
    """
    data = _fetch_weekly_data(session_id)
    if not data["entries"]:
        return {"error": "No check-ins found for this session in the last 7 days."}

    entries_block = _format_entries_block(data["entries"])
    alerts_block = _format_alerts_block(data["alerts"])

    prompt = _SUMMARY_PROMPT.format(
        session_id=session_id,
        period_start=data["period_start"],
        period_end=data["period_end"],
        entry_count=len(data["entries"]),
        entries_block=entries_block,
        alert_count=data["alert_count"],
        alerts_block=alerts_block,
    )

    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    summary_data = _parse_json(response.text)

    supabase_client.save_session_summary(
        session_id=session_id,
        period_start=data["period_start"] or datetime.now(timezone.utc).date().isoformat(),
        period_end=data["period_end"] or datetime.now(timezone.utc).date().isoformat(),
        summary_text=response.text,
    )

    summary_data["session_id"] = session_id
    summary_data["period_start"] = data["period_start"]
    summary_data["period_end"] = data["period_end"]
    return summary_data
