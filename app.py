import os
import subprocess
import tempfile
import uuid

from flask import Flask, render_template, request, jsonify

import gemini_client
import supabase_client
from graph import text_graph

app = Flask(__name__)


def _initial_state(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "user_input": "",
        "transcript": "",
        "emotion": "",
        "casel_competency": "",
        "gemini_response": "",
        "sentiment": "",
        "sentiment_score": 0.0,
        "keywords": [],
        "embedding": [],
        "similar_entries": [],
        "context_response": "",
        "drift_score": None,
        "baseline": None,
        "flag_triggered": False,
        "flag_severity": None,
        "flag_reason": None,
    }


# ── Kid-facing check-in page ──────────────────────────────────────────────────

@app.route("/")
def index():
    session_id = request.args.get("session_id") or str(uuid.uuid4())
    return render_template("index.html", session_id=session_id)


@app.route("/check-in", methods=["POST"])
def check_in():
    session_id = request.form.get("session_id") or str(uuid.uuid4())
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio received"}), 400

    webm_path = None
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            audio_file.save(tmp.name)
            webm_path = tmp.name

        wav_path = webm_path.replace(".webm", ".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path,
             "-ar", "44100", "-ac", "1", wav_path],
            check=True, capture_output=True,
        )

        # Transcribe + first-pass emotion via Gemini (also deletes wav_path)
        audio_result = gemini_client.analyze_emotion_from_audio(wav_path)
        wav_path = None  # deleted inside analyze_emotion_from_audio

        # Run the rest of the pipeline (embed → retrieve_context → drift → flag → save)
        state = _initial_state(session_id)
        state["user_input"] = audio_result["transcript"]
        state["transcript"] = audio_result["transcript"]
        final_state = text_graph.invoke(state)

        response_text = final_state.get("context_response") or final_state.get("gemini_response", "")
        return jsonify({
            "emotion": final_state["emotion"],
            "casel_competency": final_state["casel_competency"],
            "response": response_text,
            "sentiment": final_state["sentiment"],
            "drift_score": final_state.get("drift_score"),
            "flag_triggered": final_state.get("flag_triggered", False),
            "transcript": final_state["transcript"],
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    finally:
        if webm_path and os.path.exists(webm_path):
            os.remove(webm_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


# ── Teacher dashboard ─────────────────────────────────────────────────────────

@app.route("/teacher")
def teacher():
    alerts = supabase_client.get_unacknowledged_alerts()
    return render_template("teacher.html", alerts=alerts)


@app.route("/teacher/acknowledge", methods=["POST"])
def acknowledge():
    data = request.get_json(silent=True) or {}
    alert_id = data.get("alert_id")
    if not alert_id:
        return jsonify({"success": False, "error": "alert_id required"}), 400
    success = supabase_client.acknowledge_alert(alert_id)
    return jsonify({"success": success})


@app.route("/teacher/history")
def session_history():
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    entries = supabase_client.get_session_emotional_history(session_id, limit=14)
    return jsonify(entries)


@app.route("/teacher/summary")
def session_summary():
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    import summariser_agent
    result = summariser_agent.generate_weekly_summary(session_id)
    return jsonify(result)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
