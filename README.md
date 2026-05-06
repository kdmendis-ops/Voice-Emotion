# Voice Emotion Check-in Pipeline

> A seven-agent AI pipeline that processes children's daily voice check-ins — transcribing, analysing emotions, detecting personal drift, flagging safeguarding concerns, generating weekly summaries for teachers, and presenting a kid-friendly web interface for recording.

---

## The Problem

Young children can't reliably express emotions through text or multiple-choice forms. Voice is natural, low-friction, and rich with emotional signal. No SEL platform processes it seriously. This project builds the one that does.

---

## What It Does

A child records a short daily voice note via the web interface. The system processes it through a LangGraph pipeline:

1. **Record / Analyse** — sends audio to Gemini 2.5 Flash Lite, which returns a transcript, emotion, CASEL competency, sentiment score, and keywords in a single API call; text input goes through Gemini 2.5 Flash directly
2. **Embed** — generates a 768-dimension semantic vector via `gemini-embedding-001`
3. **RAG Contextualiser** — retrieves the child's three most emotionally similar past entries from pgvector and generates a history-aware empathetic response; establishes a personal emotional baseline
4. **Drift Scorer** — computes a 0.0–1.0 deviation score measuring how far today's entry is from the child's personal baseline (cosine distance + sentiment divergence)
5. **Flag Agent** — evaluates four behavioural safeguarding rules (sustained distress, escalating drift, repeated high-distress emotions, emotion volatility) and raises severity-graded alerts
6. **Save** — persists the full result to Supabase; conditionally writes safeguarding alerts
7. **Summariser Agent** — generates weekly teacher-facing narrative summaries per session: dominant emotions, sentiment arc, drift concerns, CASEL patterns, safeguarding highlights, and a recommended action

**Interfaces:**
- **Kid-facing web app** — colourful single-page UI with a large record button, 5-second countdown, pulse animation, and an empathetic response after each check-in
- **Teacher dashboard (web)** — Bootstrap 5 interface for viewing active alerts, session emotional history, and generating weekly summaries via AJAX
- **Teacher dashboard (CLI)** — original terminal-based interface for alerts, history, acknowledgement, and summary generation

---

## What Makes This Technically Interesting

**Gemini 2.5 Flash handles audio natively.** No separate transcription service — one API call returns language and emotional metadata directly from the audio file. This simplifies the architecture significantly.

**Drift detection is personal, not comparative.** The system builds a baseline per child and detects when they deviate from *themselves* — not from a class average. This is the hardest and most interesting part of the system.

**Safeguarding logic uses behavioural patterns, not keywords.** The flag agent evaluates temporal sequences across a session's history — four consecutive negative entries, monotonically escalating drift, repeated distress emotion clusters — none of which can be triggered by a single check-in.

---

## Tech Stack

| Component | Technology |
|---|---|
| AI / Multimodal Analysis | Google Gemini 2.5 Flash / Flash Lite |
| Embeddings | gemini-embedding-001 (768-dim via output_dimensionality) |
| Workflow Orchestration | LangGraph |
| Vector Memory / RAG | pgvector (via Supabase) |
| Database | Supabase (PostgreSQL) |
| Web Framework | Flask |
| Audio Capture (CLI) | sounddevice + soundfile |
| Audio Conversion (Web) | ffmpeg (subprocess) |
| Environment Config | python-dotenv |
| Deployment | Railway (nixpacks + ffmpeg) |
| Language | Python 3.13+ |

---

## Architecture

Two LangGraph pipelines share the same embed → retrieve_context → score_drift → flag_agent → save tail:

```
Kid web UI (browser)                 CLI — Text input
      │ WebM audio blob                    │
      ▼ ffmpeg → WAV                       ▼
┌──────────────┐                 ┌──────────────────┐
│ record node  │  Gemini 2.5     │  analyze node    │  Gemini 2.5
│              │  Flash Lite     │                  │  Flash
│  • transcribe│  (audio file)   │  • emotion       │  (text prompt)
│  • emotion   │                 │  • CASEL         │
│  • CASEL     │                 │  • sentiment     │
│  • sentiment │                 │  • keywords      │
│  • keywords  │                 └────────┬─────────┘
└──────┬───────┘                          │
       └──────────────┬───────────────────┘
                      ▼
             ┌────────────────────┐
             │    embed node      │  gemini-embedding-001
             │                    │  → 768-dim vector
             └────────┬───────────┘
                      ▼
             ┌────────────────────┐
             │ retrieve_context   │  pgvector cosine search
             │ node               │  → top-3 similar entries
             │                    │  → history-aware response
             └────────┬───────────┘
                      ▼
             ┌────────────────────┐
             │  score_drift node  │  0.6 × cosine_dist
             │                    │  + 0.4 × sentiment_delta
             └────────┬───────────┘
                      ▼
             ┌────────────────────┐
             │  flag_agent node   │  4 safeguarding rules
             │                    │  → severity: low/medium/high
             └────────┬───────────┘
                      ▼
             ┌────────────────────┐
             │    save node       │  Supabase (conversations
             │                    │  + safeguarding_alerts)
             └────────────────────┘

Teacher requests weekly summary
             ▼
     ┌───────────────────┐
     │  Summariser Agent │  Gemini 2.5 Flash
     │                   │  ← 7 days of history
     │  • sentiment arc  │  ← session alerts
     │  • drift concern  │  → session_summaries
     │  • narrative      │     table
     │  • recommended    │
     │    action         │
     └───────────────────┘
```

---

## Prerequisites

- Python 3.13+
- [Supabase](https://supabase.com) account with a project
- Google Gemini API key (via [Google AI Studio](https://aistudio.google.com))
- `ffmpeg` installed locally (`brew install ffmpeg` on macOS)
- A working microphone (for CLI voice input)

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd "Voice Emotion"

# 2. Create and activate a virtual environment
python3 -m venv project
source project/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install ffmpeg (macOS)
brew install ffmpeg
```

---

## Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
```

---

## Database Setup

1. Open the **SQL Editor** in your Supabase dashboard.
2. Run the full contents of `supabase_setup.sql` — this creates all tables, enables pgvector, adds the embedding column and IVFFlat index, installs the `match_conversations` RPC, and creates the `session_summaries` table used by the Summariser Agent.

---

## Usage

### Web interface (recommended)

```bash
python app.py
```

- **Kid check-in:** `http://localhost:PORT/?session_id=child-name`
- **Teacher dashboard:** `http://localhost:PORT/teacher`

The `session_id` in the URL is how you identify each child. Bookmark a stable URL per child (e.g. `/?session_id=alice-class-4b`).

### CLI interface

```bash
python main.py
```

```
╔══════════════════════════════════╗
║     Voice Emotion Analyzer       ║
╚══════════════════════════════════╝

  [1]  Voice input  (record mic)
  [2]  Text input
  [3]  Teacher Dashboard
  [q]  Quit
```

The Teacher Dashboard CLI offers:
- `[1]` View active safeguarding alerts
- `[2]` View session emotional history (last 7 entries)
- `[3]` Acknowledge an alert
- `[4]` Generate a weekly summary for any session

**Example check-in output:**

```
Transcript        : I didn't want to come to school today, nobody sat with me at lunch.
Emotion detected  : sadness
CASEL competency  : self-awareness
Context response  : It sounds like today felt really lonely. You've mentioned feeling left out a few times recently — that's hard, and it makes sense you feel that way.
Sentiment         : negative (0.81)
Keywords          : didn't want, school, nobody, lunch
Drift score       : 0.4821
```

---

## Safeguarding Rules

The flag agent evaluates four rules on every check-in. Rules can stack — more rules fired = higher severity.

| Rule | Trigger |
|---|---|
| Sustained distress | 4 consecutive entries with negative sentiment |
| Escalating drift | Drift score > 0.7 AND monotonically increasing over last 3 entries |
| Repeated high-distress | Sadness or fear in ≥ 3 of the last 6 entries |
| Emotion volatility | 3+ distinct emotions AND majority negative sentiment |

Severity: 1 rule = `low`, 2 rules = `medium`, 3+ rules = `high`. Escalating drift with score > 0.8 is always `high`.

---

## Project Structure

```
Voice Emotion/
├── app.py                # Flask web app (kid UI + teacher dashboard + API routes)
├── main.py               # CLI entry point and menu interface
├── audio_recorder.py     # Microphone capture → WAV file (CLI only)
├── gemini_client.py      # Gemini integration (audio + text analysis + embeddings)
├── graph.py              # LangGraph pipelines (text_graph + audio_graph)
├── summariser_agent.py   # Weekly summary generation via Gemini 2.5 Flash
├── supabase_client.py    # Supabase read/write + pgvector + summary operations
├── teacher_dashboard.py  # CLI teacher interface (alerts, history, summaries)
├── config.py             # .env loader for API credentials
├── supabase_setup.sql    # Database schema and all week migrations
├── requirements.txt      # Python dependencies
├── nixpacks.toml         # Railway deployment config (ffmpeg + start command)
├── templates/
│   ├── index.html        # Kid-facing recording page
│   └── teacher.html      # Teacher dashboard web UI
└── .env                  # API keys (not committed to version control)
```

---

## Build Milestones

| Week | Milestone | Status |
|---|---|---|
| 1 | Stack setup. Single Gemini agent running locally | ✅ Done |
| 2 | Audio pipeline built. Voice → Gemini → transcript end-to-end | ✅ Done |
| 3 | Emotion Analysis Agent. Sentiment and keyword extraction | ✅ Done |
| 4 | Historical entries embedded and stored in pgvector. Retrieval working | ✅ Done |
| 5 | RAG Contextualiser. Baseline establishment and drift scoring | ✅ Done |
| 6 | Flag Agent. Safeguarding alert logic and teacher dashboard | ✅ Done |
| 7 | Summariser Agent and kid-facing recording interface | ✅ Done |
| 8 | Deployed on Railway, documented, demo recorded | ⬜ Upcoming |

---

## What You'll Learn

- Multimodal AI with audio input (Gemini native audio)
- Embedding-based similarity search and RAG (pgvector)
- Longitudinal personal drift detection
- Temporal safeguarding rule evaluation
- LangGraph multi-node pipeline design
- Flask web app with browser audio capture (MediaRecorder API)
- Railway deployment with nixpacks

---

## License

MIT
