import supabase_client


# ── Formatting helpers ────────────────────────────────────────────────────────

def _severity_label(severity: str) -> str:
    return {"low": "[LOW]", "medium": "[MEDIUM]", "high": "** HIGH **"}.get(
        severity, severity.upper()
    )


def _format_timestamp(ts: str) -> str:
    return ts[:16].replace("T", " ") if ts else "unknown"


def _drift_bar(score: float | None, width: int = 12) -> str:
    if score is None:
        return "(no baseline)"
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.3f}"


# ── View 1: Unacknowledged alerts ─────────────────────────────────────────────

def show_unacknowledged_alerts() -> list[dict]:
    alerts = supabase_client.get_unacknowledged_alerts()
    print("\n" + "═" * 64)
    print("  ACTIVE SAFEGUARDING ALERTS  (unacknowledged)")
    print("═" * 64)
    if not alerts:
        print("  No active alerts. All clear.\n")
        return alerts
    for i, alert in enumerate(alerts, 1):
        print(f"\n  [{i}] Session  : {alert['session_id']}")
        print(f"      Severity : {_severity_label(alert['severity'])}")
        print(f"      Triggered: {_format_timestamp(alert['triggered_at'])}")
        print(f"      Reason   : {alert['reason']}")
    print()
    return alerts


# ── View 2: Emotional history for a session ───────────────────────────────────

def show_session_history(session_id: str) -> None:
    entries = supabase_client.get_session_emotional_history(session_id, limit=7)
    print("\n" + "═" * 64)
    print(f"  EMOTIONAL HISTORY — Session: {session_id}")
    print("═" * 64)
    if not entries:
        print("  No entries found for this session.\n")
        return
    print(f"  {'Date':<12} {'Emotion':<12} {'Sentiment':<10} {'Score':>5}  Drift")
    print("  " + "─" * 60)
    for e in entries:
        date = _format_timestamp(e["created_at"])[:10]
        emotion = e["emotion"][:11]
        sentiment = e["sentiment"][:9]
        score = f"{e['sentiment_score']:.2f}"
        drift = _drift_bar(e["drift_score"])
        print(f"  {date:<12} {emotion:<12} {sentiment:<10} {score:>5}  {drift}")
    print()


# ── View 3: Acknowledge alert ─────────────────────────────────────────────────

def acknowledge_alert_interactive(alerts: list[dict]) -> None:
    if not alerts:
        print("  No alerts to acknowledge.\n")
        return
    print("  Enter the alert number to acknowledge (or 'back' to return):")
    choice = input("  > ").strip().lower()
    if choice == "back":
        return
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(alerts):
            print("  Invalid number.\n")
            return
        alert = alerts[idx]
        success = supabase_client.acknowledge_alert(alert["id"])
        if success:
            print(f"\n  Alert acknowledged for session {alert['session_id']}.\n")
        else:
            print("\n  Could not acknowledge — alert not found.\n")
    except ValueError:
        print("  Please enter a valid number.\n")


# ── Main dashboard entry point ────────────────────────────────────────────────

def _print_summary(summary: dict) -> None:
    print("\n" + "═" * 64)
    print("  WEEKLY EMOTIONAL SUMMARY")
    print("═" * 64)
    print(f"  Session    : {summary.get('session_id', '—')}")
    print(f"  Period     : {summary.get('period_start', '?')} → {summary.get('period_end', '?')}")
    print(f"  Sentiment arc      : {summary.get('sentiment_arc', '—').upper()}")
    print(f"  Arc explanation    : {summary.get('arc_explanation', '—')}")
    avg_drift = summary.get("avg_drift_score")
    drift_str = f"{avg_drift:.3f}" if avg_drift is not None else "N/A"
    print(f"  Avg drift score    : {drift_str}")
    if summary.get("drift_concern"):
        print("  Drift concern      : YES — review closely")
    emotions = summary.get("dominant_emotions", [])
    print(f"  Dominant emotions  : {', '.join(str(e) for e in emotions) or '—'}")
    casel = summary.get("casel_patterns", [])
    print(f"  CASEL patterns     : {', '.join(str(c) for c in casel) or '—'}")
    print(f"  Safeguarding       : {summary.get('safeguarding_highlights', 'None')}")
    print(f"\n  Narrative:\n  {summary.get('narrative_summary', '—')}")
    action = summary.get("recommended_action", "—").upper()
    rationale = summary.get("recommended_action_rationale", "—")
    print(f"\n  Recommended action : {action}")
    print(f"  Rationale          : {rationale}")
    print()


def run_dashboard() -> None:
    while True:
        print("\n╔══════════════════════════════════╗")
        print("║       Teacher Dashboard          ║")
        print("╚══════════════════════════════════╝")
        print("  [1]  View active alerts")
        print("  [2]  View session emotional history")
        print("  [3]  Acknowledge an alert")
        print("  [4]  Generate session summary (last 7 days)")
        print("  [b]  Back to main menu")
        choice = input("\nChoose an option: ").strip().lower()

        if choice == "b":
            break

        elif choice == "1":
            show_unacknowledged_alerts()

        elif choice == "2":
            session_id = input("  Enter session ID: ").strip()
            if session_id:
                show_session_history(session_id)
            else:
                print("  No session ID entered.\n")

        elif choice == "3":
            alerts = show_unacknowledged_alerts()
            acknowledge_alert_interactive(alerts)

        elif choice == "4":
            import summariser_agent
            session_id = input("  Enter session ID: ").strip()
            if not session_id:
                print("  No session ID entered.\n")
                continue
            print(f"\n  Generating weekly summary for session: {session_id}")
            print("  (This may take a few seconds...)\n")
            result = summariser_agent.generate_weekly_summary(session_id)
            if "error" in result:
                print(f"  {result['error']}\n")
            else:
                _print_summary(result)

        else:
            print("  Invalid option.\n")
