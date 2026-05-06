from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY

_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BASELINE_MIN_ENTRIES = 3


def save_conversation(
    session_id: str,
    user_input: str,
    emotion: str,
    casel_competency: str,
    gemini_response: str,
    transcript: str = "",
    sentiment: str = "",
    sentiment_score: float = 0.0,
    keywords: list[str] | None = None,
    embedding: list[float] | None = None,
    drift_score: float | None = None,
    context_response: str | None = None,
    flag_triggered: bool = False,
) -> str:
    import json
    row = {
        "session_id": session_id,
        "user_input": user_input,
        "transcript": transcript,
        "emotion": emotion,
        "casel_competency": casel_competency,
        "gemini_response": gemini_response,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "keywords": json.dumps(keywords or []),
        "flag_triggered": flag_triggered,
    }
    if embedding is not None:
        row["embedding"] = embedding
    if drift_score is not None:
        row["drift_score"] = drift_score
    if context_response is not None:
        row["context_response"] = context_response
    result = _client.table("conversations").insert(row).execute()
    return result.data[0]["id"]


def get_baseline(session_id: str) -> dict | None:
    """Compute the personal emotional baseline for a session from all prior entries.

    Returns None when fewer than BASELINE_MIN_ENTRIES entries with embeddings exist.
    """
    history = get_history(session_id)
    entries = [e for e in history if e.get("embedding")]
    if len(entries) < BASELINE_MIN_ENTRIES:
        return None
    embeddings = [e["embedding"] for e in entries]
    dim = len(embeddings[0])
    centroid = [sum(emb[i] for emb in embeddings) / len(embeddings) for i in range(dim)]
    avg_sentiment = sum(e["sentiment_score"] for e in entries) / len(entries)
    emotion_distribution: dict[str, int] = {}
    for e in entries:
        emo = e.get("emotion", "unknown")
        emotion_distribution[emo] = emotion_distribution.get(emo, 0) + 1
    return {"centroid": centroid, "avg_sentiment": avg_sentiment, "emotion_distribution": emotion_distribution}


def get_history(session_id: str) -> list[dict]:
    result = (
        _client.table("conversations")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
    return result.data


def save_safeguarding_alert(
    session_id: str,
    conversation_id: str,
    severity: str,
    reason: str,
) -> str:
    result = (
        _client.table("safeguarding_alerts")
        .insert({
            "session_id": session_id,
            "conversation_id": conversation_id,
            "severity": severity,
            "reason": reason,
        })
        .execute()
    )
    return result.data[0]["id"]


def get_unacknowledged_alerts() -> list[dict]:
    result = (
        _client.table("safeguarding_alerts")
        .select("id, session_id, severity, reason, triggered_at")
        .is_("acknowledged_at", "null")
        .order("triggered_at", desc=True)
        .execute()
    )
    return result.data


def get_session_emotional_history(session_id: str, limit: int = 7) -> list[dict]:
    history = get_history(session_id)
    recent = history[-limit:] if len(history) >= limit else history
    return [
        {
            "emotion": e.get("emotion", ""),
            "sentiment": e.get("sentiment", ""),
            "sentiment_score": e.get("sentiment_score", 0.0),
            "drift_score": e.get("drift_score"),
            "created_at": e.get("created_at", ""),
        }
        for e in recent
    ]


def acknowledge_alert(alert_id: str) -> bool:
    from datetime import datetime, timezone
    result = (
        _client.table("safeguarding_alerts")
        .update({"acknowledged_at": datetime.now(timezone.utc).isoformat()})
        .eq("id", alert_id)
        .execute()
    )
    return len(result.data) > 0


def save_session_summary(
    session_id: str,
    period_start: str,
    period_end: str,
    summary_text: str,
) -> str:
    result = (
        _client.table("session_summaries")
        .insert({
            "session_id": session_id,
            "period_start": period_start,
            "period_end": period_end,
            "summary_text": summary_text,
        })
        .execute()
    )
    return result.data[0]["id"]


def get_session_summaries(session_id: str) -> list[dict]:
    result = (
        _client.table("session_summaries")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(5)
        .execute()
    )
    return result.data


def get_session_alerts(session_id: str) -> list[dict]:
    result = (
        _client.table("safeguarding_alerts")
        .select("id, session_id, severity, reason, triggered_at, acknowledged_at")
        .eq("session_id", session_id)
        .order("triggered_at", desc=True)
        .execute()
    )
    return result.data


def search_similar(
    query_embedding: list[float],
    session_id: str,
    limit: int = 5,
) -> list[dict]:
    """Return the most semantically similar past entries for a session.

    Calls the match_conversations Postgres function (defined in supabase_setup.sql),
    which ranks rows by cosine distance using pgvector.

    Args:
        query_embedding: A 768-dimension embedding of the query text.
        session_id: Only search within this session's entries.
        limit: Maximum number of results to return (default 5).

    Returns:
        List of dicts ordered by ascending cosine_distance (0.0 = identical,
        2.0 = opposite). Each dict includes all conversation fields plus
        'cosine_distance'. Convert to similarity via: 1 - cosine_distance.
    """
    result = _client.rpc(
        "match_conversations",
        {
            "query_embedding": query_embedding,
            "match_session_id": session_id,
            "match_count": limit,
        },
    ).execute()
    return result.data
