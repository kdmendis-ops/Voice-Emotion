import uuid
from graph import text_graph, audio_graph
import teacher_dashboard


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


def _print_result(result: dict, show_transcript: bool = False) -> None:
    if show_transcript and result.get("transcript"):
        print(f"\nTranscript        : {result['transcript']}")
    print(f"Emotion detected  : {result['emotion']}")
    print(f"CASEL competency  : {result['casel_competency']}")
    if result.get("context_response") and result.get("similar_entries"):
        print(f"Context response  : {result['context_response']}")
    else:
        print(f"Gemini            : {result['gemini_response']}")
    if result.get("sentiment"):
        score = result.get("sentiment_score", 0.0)
        print(f"Sentiment         : {result['sentiment']} ({score:.2f})")
    if result.get("keywords"):
        print(f"Keywords          : {', '.join(result['keywords'])}")
    if result.get("drift_score") is not None:
        print(f"Drift score       : {result['drift_score']:.4f}")
    if result.get("flag_triggered"):
        severity = result.get("flag_severity", "unknown").upper()
        reason = result.get("flag_reason", "")
        print(f"Safeguarding flag : {severity} — {reason}")
    print()


def main():
    session_id = str(uuid.uuid4())

    print("╔══════════════════════════════════╗")
    print("║     Voice Emotion Analyzer       ║")
    print("╚══════════════════════════════════╝")
    print()

    while True:
        print("  [1]  Voice input  (record mic)")
        print("  [2]  Text input")
        print("  [3]  Teacher Dashboard")
        print("  [q]  Quit")
        choice = input("\nChoose an option: ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            break

        elif choice == "3":
            teacher_dashboard.run_dashboard()

        elif choice == "1":
            # ── Voice mode ────────────────────────────────────────────────
            state = _initial_state(session_id)
            result = audio_graph.invoke(state)
            _print_result(result, show_transcript=True)

        elif choice == "2":
            # ── Text mode ─────────────────────────────────────────────────
            user_input = input("You: ").strip()
            if not user_input:
                print("No input detected. Try again.\n")
                continue

            state = _initial_state(session_id)
            state["user_input"] = user_input
            result = text_graph.invoke(state)
            _print_result(result, show_transcript=False)

        else:
            print("Invalid option. Please choose 1, 2, 3, or q.\n")


if __name__ == "__main__":
    main()
