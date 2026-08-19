import os

from sarvamai import SarvamAI


# ============================================================
# SARVAM CLIENT
# ============================================================

API_KEY = os.getenv("SARVAM_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "SARVAM_API_KEY not set."
    )

client = SarvamAI(
    api_subscription_key=API_KEY
)


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

SUPPORTED_LANGUAGES = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
}


# ============================================================
# SPEECH TO TEXT
# ============================================================

def transcribe_audio(audio_path: str):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print("\n========== SARVAM STT ==========")
    print("Audio:", audio_path)
    print("Model: saaras")
    print("================================\n")

    try:

        with open(audio_path, "rb") as audio_file:

            response = client.speech_to_text.transcribe(
                file=audio_file,
                model="saaras",
                mode="transcribe",
                language_code="unknown"
            )

    except Exception as e:

        print(
            "\n========== STT ERROR =========="
        )

        print(
            "TYPE:",
            type(e).__name__
        )

        print(
            "ERROR:",
            repr(e)
        )

        print(
            "================================\n"
        )

        raise

    # ========================================================
    # EXTRACT RESPONSE
    # ========================================================

    transcript = getattr(
        response,
        "transcript",
        ""
    )

    language_code = getattr(
        response,
        "language_code",
        None
    )

    language_probability = getattr(
        response,
        "language_probability",
        None
    )

    transcript = (
        transcript or ""
    ).strip()

    # ========================================================
    # LANGUAGE NAME
    # ========================================================

    language = SUPPORTED_LANGUAGES.get(
        language_code,
        "Unknown"
    )

    # ========================================================
    # DEBUG
    # ========================================================

    print(
        "\n========== STT RESULT =========="
    )

    print(
        "Transcript:",
        transcript
    )

    print(
        "Language code:",
        language_code
    )

    print(
        "Language:",
        language
    )

    print(
        "Language probability:",
        language_probability
    )

    print(
        "================================\n"
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "transcript": transcript,
        "language_code": language_code,
        "language": language,
        "language_probability": language_probability,
    }