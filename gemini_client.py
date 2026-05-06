import json
import os
import pathlib
from google import genai
from google.genai import types
from config import GEMINI_API_KEY

_client = genai.Client(api_key=GEMINI_API_KEY)

_PROMPT_TEMPLATE = """
Analyze the emotion expressed in the following text.

Text: "{text}"

Respond with ONLY a valid JSON object (no markdown, no extra text) with these keys:
- "emotion": the primary emotion detected (e.g. joy, sadness, anger, fear, surprise, disgust, neutral)
- "casel_competency": the most relevant CASEL SEL competency for this emotion.
  Choose exactly one of: self-awareness, self-management, social-awareness, relationship-skills, responsible-decision-making
- "response": a brief (1-2 sentence) empathetic reply to the person
- "sentiment": overall sentiment polarity — one of: positive, negative, neutral
- "sentiment_score": confidence score for the sentiment as a float between 0.0 and 1.0
- "keywords": a list of 3 to 5 key words or short phrases that capture the main topics of the text

Example output:
{{"emotion": "sadness", "casel_competency": "self-awareness", "response": "It sounds like you're going through a tough time. Recognizing that feeling is an important first step.", "sentiment": "negative", "sentiment_score": 0.82, "keywords": ["tough time", "feeling", "recognition"]}}
"""


_AUDIO_PROMPT = """
Listen to the audio recording and do the following in order:
1. Transcribe exactly what was said in the audio.
2. Analyze the emotion expressed in the transcribed speech.

Respond with ONLY a valid JSON object (no markdown, no extra text) with these keys:
- "transcript": the full verbatim transcription of the audio
- "emotion": the primary emotion detected (e.g. joy, sadness, anger, fear, surprise, disgust, neutral)
- "casel_competency": the most relevant CASEL SEL competency for this emotion.
  Choose exactly one of: self-awareness, self-management, social-awareness, relationship-skills, responsible-decision-making
- "response": a brief (1-2 sentence) empathetic reply to the person
- "sentiment": overall sentiment polarity — one of: positive, negative, neutral
- "sentiment_score": confidence score for the sentiment as a float between 0.0 and 1.0
- "keywords": a list of 3 to 5 key words or short phrases that capture the main topics of the audio

Example output:
{"transcript": "I feel really overwhelmed with everything going on.", "emotion": "fear", "casel_competency": "self-management", "response": "It sounds like things feel really heavy right now. Taking it one step at a time can help.", "sentiment": "negative", "sentiment_score": 0.91, "keywords": ["overwhelmed", "everything going on"]}
"""


_RAG_PROMPT_TEMPLATE = """
You are an empathetic AI supporting a child's emotional wellbeing.

The child has just said: "{text}"

Their emotional analysis today:
- Emotion: {emotion}
- Sentiment: {sentiment} ({sentiment_score})

Here are their 3 most emotionally similar past check-ins for context:
{history_context}

Their overall emotion history: {emotion_distribution}

Provide a warm, contextualised 2-3 sentence response that:
1. Acknowledges what they expressed today
2. Gently references any patterns you notice (e.g. "You've mentioned feeling this way before...")
3. Offers a supportive, age-appropriate reflection

If history_context is empty, respond normally without referencing history.
Respond with ONLY the plain reply text — no JSON, no labels.
"""


def _parse_json(raw: str) -> dict:
    """Strip markdown code fences if present, then parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def analyze_emotion_from_audio(audio_path: str) -> dict:
    """Upload an audio file to Gemini and return transcript + emotion analysis.

    Args:
        audio_path: Path to a WAV audio file.

    Returns:
        Dict with keys: transcript, emotion, casel_competency, response.
    """
    try:
        audio_file = _client.files.upload(file=pathlib.Path(audio_path))
        response = _client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[audio_file, _AUDIO_PROMPT],
        )
        return _parse_json(response.text)
    finally:
        # Always clean up the temp file, even if Gemini raises an error
        if os.path.exists(audio_path):
            os.remove(audio_path)


def analyze_emotion(text: str) -> dict:
    prompt = _PROMPT_TEMPLATE.format(text=text)
    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return _parse_json(response.text)


def generate_context_response(
    text: str,
    emotion: str,
    sentiment: str,
    sentiment_score: float,
    similar_entries: list[dict],
    emotion_distribution: dict,
) -> str:
    """Generate a history-aware empathetic response using RAG context."""
    history_context = "\n".join(
        f"- [{e['created_at'][:10]}] Emotion: {e['emotion']} | \"{e['user_input'][:80]}\""
        for e in similar_entries
    ) or "(no prior entries)"
    distribution_str = ", ".join(f"{k} x{v}" for k, v in emotion_distribution.items())
    prompt = _RAG_PROMPT_TEMPLATE.format(
        text=text,
        emotion=emotion,
        sentiment=sentiment,
        sentiment_score=f"{sentiment_score:.2f}",
        history_context=history_context,
        emotion_distribution=distribution_str,
    )
    response = _client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text.strip()


def generate_embedding(text: str) -> list[float]:
    """Generate a 768-dimension semantic embedding for the given text.

    Uses Google's text-embedding-004 model. The 768-dimension output matches
    the vector(768) column in Supabase and is used for pgvector similarity search.

    Args:
        text: The text to embed (user input or transcript).

    Returns:
        A list of 768 floats representing the semantic embedding.
    """
    response = _client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[text],
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    if not response.embeddings:
        raise ValueError("Gemini returned no embeddings for the provided text.")
    return response.embeddings[0].values
