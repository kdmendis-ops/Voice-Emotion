import tempfile
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 44100   # Hz
CHANNELS = 1          # Mono


def record_audio(duration_seconds: int = 5) -> str:
    """Record audio from the default microphone and save to a temp WAV file.

    Args:
        duration_seconds: How many seconds to record (default 5).

    Returns:
        The file path of the saved WAV file (string).
    """
    print(f"\nRecording for {duration_seconds} seconds... Speak now!")

    audio_data = sd.rec(
        frames=int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()  # Block until recording is complete
    print("Done recording.\n")

    # Write to a named temp file that persists until we explicitly delete it
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_data, SAMPLE_RATE)
    tmp.close()

    return tmp.name
