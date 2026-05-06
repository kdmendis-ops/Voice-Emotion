# Voice Emotion Check-in Pipeline

> A five-agent AI pipeline that processes children's daily voice check-ins — transcribing, analysing emotions, detecting personal drift, flagging safeguarding concerns, and generating weekly summaries for teachers.

---

## The Problem

Young children can't reliably express emotions through text or multiple-choice forms. Voice is natural, low-friction, and rich with emotional signal. No SEL platform processes it seriously. This project builds the one that does.

---

## What It Does

A child records a short daily voice note. The system processes it through a LangGraph pipeline:

**Live**
1. **Record / Analyse** — sends audio to Gemini 2.5 Flash Lite, which returns a transcript, emotion, CASEL competency, sentiment score, and keywords in a single API call; text input goes through Gemini 2.5 Flash directly
2. **Embed** — generates a 768-dimension semantic vector via `gemini-embedding-001`
3. **Save** — persists the full result (including embedding) to Supabase for retrieval

**Planned**
4. **RAG Contextualiser** — compares today's embedding against the child's history in pgvector to detect personal emotional drift
5. **Flag Agent** — raises safeguarding alerts for sustained distress patterns using conditional logic, not keyword matching
6. **Summariser** — generates weekly emotional pattern summaries per child for teacher review

---

## What Makes This Technically Interesting

**Gemini 2.5 Flash handles audio natively.** No separate transcription service — one API call returns language and emotional metadata directly from the audio file. This simplifies the architecture significantly.

**Drift detection is personal, not comparative.** The system builds a baseline per child and detects when they deviate from *themselves* — not from a class average. This is the hardest and most interesting part of the system.

---

## Tech Stack

| Component | Technology |
|---|---|
| AI / Multimodal Analysis | Google Gemini 2.5 Flash / Flash Lite |
| Embeddings | gemini-embedding-001 (768-dim via output_dimensionality) |
| Workflow Orchestration | LangGraph |
| Vector Memory / RAG | pgvector (via Supabase) |
| Database | Supabase (PostgreSQL) |
| Audio Capture | sounddevice + soundfile |
| Environment Config | python-dotenv |
| Deployment | Railway |
| Language | Python 3.13+ |

---

## Architecture

Two LangGraph pipelines share the same embed → save tail:

```
Voice input                          Text input
     │                                    │
     ▼                                    ▼
┌──────────────┐                 ┌──────────────────┐
│ record node  │  Gemini 2.5     │  analyze node    │  Gemini 2.5
│              │  Flash Lite     │                  │  Flash
│  • transcribe│  (audio file)   │  • emotion       │  (text prompt)
│  • emotion   │                 │  • CASEL         │
│  • CASEL     │                 │  • sentiment     │
│  • sentiment │                 │  • keywords      │
│  • keywords  │                 └────────┬─────────┘
└──────┬───────┘                          │
       │                                  │
       └──────────────┬───────────────────┘
                      ▼
             ┌────────────────┐
             │   embed node   │  gemini-embedding-001
             │                │  → 768-dim vector
             └────────┬───────┘
                      ▼
             ┌────────────────┐
             │   save node    │  Supabase (pgvector)
             └────────────────┘
```

**Upcoming nodes** (Weeks 5–8): RAG Contextualiser (drift scoring), Flag Agent (safeguarding alerts), Summariser Agent (weekly teacher report).

---

## Prerequisites

- Python 3.13+
- [Supabase](https://supabase.com) account with a project
- Google Gemini API key (via [Google AI Studio](https://aistudio.google.com))
- A working microphone (for voice input)

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
2. Run `supabase_setup.sql` — this creates the `conversations` table, enables the `pgvector` extension, adds the `embedding vector(768)` column, creates the IVFFlat cosine-similarity index, and installs the `match_conversations` RPC function used for semantic retrieval.

---

## Usage

```bash
python main.py
```

```
╔══════════════════════════════════╗
║     Voice Emotion Analyzer       ║
╚══════════════════════════════════╝

  [1]  Voice input  (record mic)
  [2]  Text input
  [q]  Quit
```

- **Option 1** — Records a voice note, transcribes and analyses it end-to-end through the full agent pipeline
- **Option 2** — Type input directly for analysis
- **Option q** — Exit

**Example output:**

```
Emotion detected  : sadness
CASEL competency  : self-awareness
Gemini            : It sounds like today felt really lonely. That feeling is valid and important.
Sentiment         : negative (0.81)
Keywords          : didn't want, school, nobody, lunch
```

---

## Project Structure

```
Voice Emotion/
├── main.py               # CLI entry point and menu interface
├── audio_recorder.py     # Microphone capture → WAV file
├── gemini_client.py      # Gemini integration (audio + text analysis)
├── graph.py              # LangGraph agent pipeline (5-node workflow)
├── supabase_client.py    # Supabase read/write + pgvector operations
├── config.py             # .env loader for API credentials
├── supabase_setup.sql    # Database schema and migrations
├── requirements.txt      # Python dependencies
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
| 5 | RAG Contextualiser. Baseline establishment and drift scoring | 🔄 In progress |
| 6 | Flag Agent. Safeguarding alert logic and teacher dashboard view | ⬜ Upcoming |
| 7 | Summariser Agent and kid-facing recording interface | ⬜ Upcoming |
| 8 | Deployed on Railway, documented, demo recorded | ⬜ Upcoming |

---

## What You'll Learn

- Multimodal AI with audio input
- Embedding-based similarity search
- Longitudinal pattern detection
- Conditional agent branching
- Safeguarding system design

---

## License

MIT
