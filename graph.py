from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
import gemini_client
import supabase_client
import audio_recorder


class EmotionState(TypedDict):
    session_id: str
    user_input: str       # Raw text input (or transcript for voice)
    transcript: str       # Verbatim speech-to-text (voice only; empty for text input)
    emotion: str
    casel_competency: str
    gemini_response: str
    sentiment: str
    sentiment_score: float
    keywords: list[str]
    embedding: list[float]    # 768-dimension semantic vector (Week 4)
    similar_entries: list[dict]   # top-3 semantically similar past entries (Week 5)
    context_response: str         # history-aware Gemini response (Week 5)
    drift_score: float | None     # 0.0–1.0 personal drift; None if < 3 prior entries (Week 5)
    baseline: dict | None         # centroid + avg_sentiment + emotion_distribution (Week 5)
    flag_triggered: bool          # True if >= 1 safeguarding rule fired (Week 6)
    flag_severity: str | None     # "low" | "medium" | "high" | None (Week 6)
    flag_reason: str | None       # human-readable summary of fired rules (Week 6)


_DRIFT_EMBEDDING_WEIGHT = 0.6
_DRIFT_SENTIMENT_WEIGHT = 0.4


# ── Text pipeline nodes ───────────────────────────────────────────────────────

def analyze_node(state: EmotionState) -> dict:
    """Call Gemini with plain text and return emotion analysis."""
    result = gemini_client.analyze_emotion(state["user_input"])
    return {
        "emotion": result["emotion"],
        "casel_competency": result["casel_competency"],
        "gemini_response": result["response"],
        "sentiment": result["sentiment"],
        "sentiment_score": result["sentiment_score"],
        "keywords": result["keywords"],
    }


# ── Audio pipeline node ───────────────────────────────────────────────────────

def record_node(state: EmotionState) -> dict:
    """Record audio from the mic, transcribe via Gemini, and analyze emotion."""
    audio_path = audio_recorder.record_audio(duration_seconds=5)
    result = gemini_client.analyze_emotion_from_audio(audio_path)
    return {
        "user_input": result["transcript"],   # store transcript as user_input for DB
        "transcript": result["transcript"],
        "emotion": result["emotion"],
        "casel_competency": result["casel_competency"],
        "gemini_response": result["response"],
        "sentiment": result["sentiment"],
        "sentiment_score": result["sentiment_score"],
        "keywords": result["keywords"],
    }


# ── Embedding node ────────────────────────────────────────────────────────────

def embed_node(state: EmotionState) -> dict:
    """Generate a semantic embedding for the user's input."""
    embedding = gemini_client.generate_embedding(state["user_input"])
    return {"embedding": embedding}


# ── RAG context retrieval node ────────────────────────────────────────────────

def retrieve_context_node(state: EmotionState) -> dict:
    """Fetch personal baseline and top-3 similar entries; generate RAG response."""
    baseline = supabase_client.get_baseline(state["session_id"])
    if baseline is None:
        return {"similar_entries": [], "context_response": state["gemini_response"], "baseline": None}
    similar_entries = supabase_client.search_similar(state["embedding"], state["session_id"], limit=3)
    context_response = gemini_client.generate_context_response(
        text=state["user_input"],
        emotion=state["emotion"],
        sentiment=state["sentiment"],
        sentiment_score=state["sentiment_score"],
        similar_entries=similar_entries,
        emotion_distribution=baseline["emotion_distribution"],
    )
    return {"similar_entries": similar_entries, "context_response": context_response, "baseline": baseline}


# ── Drift scoring node ────────────────────────────────────────────────────────

def score_drift_node(state: EmotionState) -> dict:
    """Compute normalised drift score (0.0–1.0) vs. personal baseline."""
    baseline = state.get("baseline")
    if baseline is None:
        return {"drift_score": None}
    current_emb = state["embedding"]
    centroid = baseline["centroid"]
    dot = sum(a * b for a, b in zip(current_emb, centroid))
    norm_c = sum(x * x for x in current_emb) ** 0.5
    norm_b = sum(x * x for x in centroid) ** 0.5
    if norm_c == 0 or norm_b == 0:
        cosine_dist = 1.0
    else:
        cosine_dist = 1.0 - dot / (norm_c * norm_b)
    embedding_component = cosine_dist / 2.0
    sentiment_component = abs(state["sentiment_score"] - baseline["avg_sentiment"])
    drift_score = round(
        _DRIFT_EMBEDDING_WEIGHT * embedding_component + _DRIFT_SENTIMENT_WEIGHT * sentiment_component, 4
    )
    return {"drift_score": max(0.0, min(1.0, drift_score))}


# ── Safeguarding flag agent node ──────────────────────────────────────────────

_DISTRESS_EMOTIONS = {"sadness", "fear"}


def flag_agent_node(state: EmotionState) -> dict:
    """Evaluate four behavioural safeguarding rules against session history."""
    history = supabase_client.get_history(state["session_id"])
    recent_7 = history[-7:]
    recent_5 = recent_7[-5:]
    recent_3 = recent_7[-3:]
    current_drift = state.get("drift_score")
    rules_fired: list[str] = []

    # Rule 1: sustained distress — 3 most-recent saved entries + current = 4 consecutive negative
    chain = list(recent_3) + [{"sentiment": state["sentiment"]}]
    if len(chain) == 4 and all(e.get("sentiment") == "negative" for e in chain):
        rules_fired.append("Sustained distress: 4 consecutive negative sentiment entries")

    # Rule 2: escalating drift — drift > 0.7 AND monotonically increasing over last 3 saved + current
    if current_drift is not None and current_drift > 0.7:
        dw = [e["drift_score"] for e in recent_3 if e.get("drift_score") is not None]
        dw.append(current_drift)
        if len(dw) >= 3 and all(dw[i] < dw[i + 1] for i in range(len(dw) - 1)):
            rules_fired.append(
                f"Escalating drift: score {current_drift:.2f} with increasing trend"
            )

    # Rule 3: repeated high-distress emotions — sadness/fear >= 3 times in last 5 saved + current
    ew3 = [e.get("emotion", "") for e in recent_5] + [state["emotion"]]
    distress_count = sum(1 for e in ew3 if e in _DISTRESS_EMOTIONS)
    if distress_count >= 3:
        rules_fired.append(
            f"Repeated high-distress emotions: {distress_count} occurrences of "
            f"sadness/fear in last {len(ew3)} entries"
        )

    # Rule 4: emotion volatility — 3+ distinct emotions + majority negative sentiment
    ew4 = [e.get("emotion", "") for e in recent_5] + [state["emotion"]]
    sw = [e.get("sentiment", "") for e in recent_5] + [state["sentiment"]]
    sc = [e.get("sentiment_score", 0.0) for e in recent_5] + [state["sentiment_score"]]
    avg_score = sum(sc) / len(sc) if sc else 0.0
    neg_count = sum(1 for s in sw if s == "negative")
    if len(set(ew4)) >= 3 and neg_count > len(sw) / 2 and avg_score > 0.5:
        rules_fired.append(
            f"Emotion volatility: {len(set(ew4))} distinct emotions with "
            f"predominantly negative sentiment (avg score {avg_score:.2f})"
        )

    if not rules_fired:
        return {"flag_triggered": False, "flag_severity": None, "flag_reason": None}

    n = len(rules_fired)
    severity = "low" if n == 1 else "medium" if n == 2 else "high"
    if any("Escalating drift" in r for r in rules_fired) and current_drift and current_drift > 0.8:
        severity = "high"

    return {
        "flag_triggered": True,
        "flag_severity": severity,
        "flag_reason": "; ".join(rules_fired),
    }


# ── Shared save node ──────────────────────────────────────────────────────────

def save_node(state: EmotionState) -> dict:
    """Persist the conversation to Supabase; conditionally write a safeguarding alert."""
    conversation_id = supabase_client.save_conversation(
        session_id=state["session_id"],
        user_input=state["user_input"],
        transcript=state["transcript"],
        emotion=state["emotion"],
        casel_competency=state["casel_competency"],
        gemini_response=state["gemini_response"],
        sentiment=state["sentiment"],
        sentiment_score=state["sentiment_score"],
        keywords=state["keywords"],
        embedding=state.get("embedding"),
        drift_score=state.get("drift_score"),
        context_response=state.get("context_response"),
        flag_triggered=state.get("flag_triggered", False),
    )
    if state.get("flag_triggered") and state.get("flag_severity"):
        supabase_client.save_safeguarding_alert(
            session_id=state["session_id"],
            conversation_id=conversation_id,
            severity=state["flag_severity"],
            reason=state["flag_reason"],
        )
    return {}


# ── Text graph:  START → analyze → embed → retrieve_context → score_drift → flag_agent → save → END ──

_text_builder = StateGraph(EmotionState)
_text_builder.add_node("analyze", analyze_node)
_text_builder.add_node("embed", embed_node)
_text_builder.add_node("retrieve_context", retrieve_context_node)
_text_builder.add_node("score_drift", score_drift_node)
_text_builder.add_node("flag_agent", flag_agent_node)
_text_builder.add_node("save", save_node)
_text_builder.add_edge(START, "analyze")
_text_builder.add_edge("analyze", "embed")
_text_builder.add_edge("embed", "retrieve_context")
_text_builder.add_edge("retrieve_context", "score_drift")
_text_builder.add_edge("score_drift", "flag_agent")
_text_builder.add_edge("flag_agent", "save")
_text_builder.add_edge("save", END)
text_graph = _text_builder.compile()

# ── Audio graph: START → record → embed → retrieve_context → score_drift → flag_agent → save → END ──

_audio_builder = StateGraph(EmotionState)
_audio_builder.add_node("record", record_node)
_audio_builder.add_node("embed", embed_node)
_audio_builder.add_node("retrieve_context", retrieve_context_node)
_audio_builder.add_node("score_drift", score_drift_node)
_audio_builder.add_node("flag_agent", flag_agent_node)
_audio_builder.add_node("save", save_node)
_audio_builder.add_edge(START, "record")
_audio_builder.add_edge("record", "embed")
_audio_builder.add_edge("embed", "retrieve_context")
_audio_builder.add_edge("retrieve_context", "score_drift")
_audio_builder.add_edge("score_drift", "flag_agent")
_audio_builder.add_edge("flag_agent", "save")
_audio_builder.add_edge("save", END)
audio_graph = _audio_builder.compile()

# ── Post-audio graph (web): START → embed → retrieve_context → score_drift → flag_agent → save → END
# Used by the web /check-in route after analyze_emotion_from_audio has already populated the state.
# Skips analyze_node so the audio model's emotion/prosody analysis is preserved.

_post_audio_builder = StateGraph(EmotionState)
_post_audio_builder.add_node("embed", embed_node)
_post_audio_builder.add_node("retrieve_context", retrieve_context_node)
_post_audio_builder.add_node("score_drift", score_drift_node)
_post_audio_builder.add_node("flag_agent", flag_agent_node)
_post_audio_builder.add_node("save", save_node)
_post_audio_builder.add_edge(START, "embed")
_post_audio_builder.add_edge("embed", "retrieve_context")
_post_audio_builder.add_edge("retrieve_context", "score_drift")
_post_audio_builder.add_edge("score_drift", "flag_agent")
_post_audio_builder.add_edge("flag_agent", "save")
_post_audio_builder.add_edge("save", END)
post_audio_graph = _post_audio_builder.compile()
