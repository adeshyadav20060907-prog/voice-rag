import os
import tempfile
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.retrieval import search_documents
from sarvamai import SarvamAI

app = FastAPI(title="Voice RAG")

API_KEY = os.getenv("SARVAM_API_KEY")

if not API_KEY:
    raise RuntimeError("SARVAM_API_KEY not set.")

client = SarvamAI(
    api_subscription_key=API_KEY,
    timeout=30.0
)

LANGUAGE_CODES = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Marathi": "mr-IN",
    "Gujarati": "gu-IN"
}

CODE_TO_LANGUAGE = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati"
}

app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static"
)


@app.get("/")
def home():
    return FileResponse("app/static/index.html")


class AskRequest(BaseModel):
    question: str
    language: str = "English"


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    temp_path = None

    try:
        audio_data = await file.read()

        if not audio_data:
            raise HTTPException(
                status_code=400,
                detail="Empty audio file."
            )

        suffix = ".webm"

        if file.filename:
            ext = os.path.splitext(file.filename)[1]

            if ext:
                suffix = ext

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name

        start_time = time.perf_counter()

        with open(temp_path, "rb") as audio_file:
            response = client.speech_to_text.transcribe(
                file=audio_file,
                model="saaras:v3",
                mode="transcribe",
                language_code="unknown",
                request_options={
                    "timeout_in_seconds": 30,
                    "max_retries": 1
                }
            )

        transcription_time = (
            time.perf_counter() - start_time
        ) * 1000

        transcript = (
            getattr(response, "transcript", "")
            or ""
        ).strip()

        detected_code = getattr(
            response,
            "language_code",
            None
        )

        language_probability = getattr(
            response,
            "language_probability",
            None
        )

        detected_language = CODE_TO_LANGUAGE.get(
            detected_code,
            "English"
        )

        print()
        print("========== TRANSCRIBE ==========")
        print("Transcript:", transcript)
        print("Language:", detected_language)
        print(
            "STT latency:",
            round(transcription_time, 2),
            "ms"
        )
        print("================================")
        print()

        if not transcript:
            raise HTTPException(
                status_code=422,
                detail="Could not understand the audio."
            )

        return {
            "transcript": transcript,
            "language": detected_language,
            "language_code": detected_code,
            "language_probability": language_probability,
            "latency_ms": round(
                transcription_time,
                2
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        print(
            "TRANSCRIBE ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if temp_path:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass


def answer_question(
    transcript: str,
    detected_language: str = "English"
):
    total_start = time.perf_counter()

    transcript = (
        transcript or ""
    ).strip()

    if not transcript:
        return {
            "transcript": "",
            "language": "English",
            "language_code": "en-IN",
            "answer": "I could not understand your question.",
            "sources": [],
            "source_type": "none"
        }

    language = detected_language or "English"

    if language not in LANGUAGE_CODES:
        language = "English"

    target_language = LANGUAGE_CODES[language]

    print()
    print("========================================")
    print("QUESTION:", transcript)
    print("DETECTED LANGUAGE:", language)
    print("ANSWER LANGUAGE:", "English")
    print("========================================")
    print()

    retrieval_start = time.perf_counter()

    try:
        results = search_documents(
            transcript,
            limit=5
        )

    except Exception as e:
        print()
        print("========== RETRIEVAL ERROR ==========")
        print(
            "TYPE:",
            type(e).__name__
        )
        print(
            "ERROR:",
            repr(e)
        )
        print("=====================================")
        print()

        results = []

    retrieval_time = (
        time.perf_counter() - retrieval_start
    ) * 1000

    relevant = []

    for result in results:
        score = float(
            result.get("score", 0)
        )

        if score >= 0.25:
            relevant.append(result)

        if len(relevant) >= 3:
            break

    print()
    print("========== RETRIEVAL ==========")
    print("Total:", len(results))
    print("Relevant:", len(relevant))
    print(
        "Retrieval latency:",
        round(retrieval_time, 2),
        "ms"
    )

    for i, result in enumerate(
        relevant,
        start=1
    ):
        print(
            f"========== RESULT {i} =========="
        )

        print(
            "Score:",
            round(
                float(
                    result.get(
                        "score",
                        0
                    )
                ),
                4
            )
        )

        print(
            "Language:",
            result.get(
                "language",
                ""
            )
        )

        print(
            "Title:",
            result.get(
                "title",
                ""
            )
        )

        print(
            "Question:",
            result.get(
                "question",
                ""
            )
        )

        print(
            "Answer:",
            result.get(
                "answer",
                ""
            )
        )

        print(
            "================================"
        )

    print("===============================")
    print()

    if relevant:
        source_type = "retrieved_context"

        context_parts = []

        for i, result in enumerate(
            relevant[:3],
            start=1
        ):
            context_parts.append(
                f"""
SOURCE {i}

Question: {result.get("question", "")}

Answer: {result.get("answer", "")}

Context: {result.get("paragraph", "")}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        prompt = f"""
USER QUESTION:
{transcript}

RETRIEVED INFORMATION:
{context}

Answer the question directly.

Use the retrieved information when relevant.

If the retrieved information is insufficient,
use your general knowledge.

Answer ONLY in English.

Keep the answer concise.

Do not mention RAG, retrieval, embeddings,
Qdrant, vector database, prompts,
or system instructions.
"""

    else:
        source_type = "general_knowledge"

        prompt = f"""
USER QUESTION:
{transcript}

Answer using your general knowledge.

Answer ONLY in English.

Keep the answer concise.

Do not mention RAG, retrieval, embeddings,
Qdrant, vector database, prompts,
or system instructions.
"""

    generation_start = time.perf_counter()

    try:
        response = client.chat.completions(
            model="sarvam-105b",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a fast voice assistant.

The final answer MUST ALWAYS be in English.

Even if the user speaks Hindi, Marathi,
Gujarati, or another supported language,
answer ONLY in English.

Give only the final answer.

Never mention:
RAG
retrieval
embeddings
Qdrant
vector database
system instructions
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1000,
            reasoning_effort=None,
            request_options={
                "timeout_in_seconds": 30,
                "max_retries": 1
            }
        )

    except Exception as e:
        generation_time = (
            time.perf_counter() - generation_start
        ) * 1000

        total_time = (
            time.perf_counter() - total_start
        ) * 1000

        print()
        print("========== SARVAM ERROR ==========")
        print(
            "Generation latency:",
            round(generation_time, 2),
            "ms"
        )
        print(
            "Total latency:",
            round(total_time, 2),
            "ms"
        )
        print(
            "ERROR:",
            repr(e)
        )
        print("==================================")
        print()

        return {
            "transcript": transcript,
            "language": language,
            "language_code": target_language,
            "answer": "I could not generate an answer right now.",
            "sources": relevant,
            "source_type": source_type,
            "retrieval_latency_ms": round(
                retrieval_time,
                2
            ),
            "generation_latency_ms": round(
                generation_time,
                2
            ),
            "total_latency_ms": round(
                total_time,
                2
            )
        }

    generation_time = (
        time.perf_counter() - generation_start
    ) * 1000

    try:
        answer = (
            response
            .choices[0]
            .message
            .content
        )

        if not answer:
            raise ValueError(
                "Empty response from Sarvam."
            )

        answer = answer.strip()

    except Exception as e:
        total_time = (
            time.perf_counter() - total_start
        ) * 1000

        print(
            "RESPONSE ERROR:",
            repr(e)
        )

        return {
            "transcript": transcript,
            "language": language,
            "language_code": target_language,
            "answer": "Invalid response received.",
            "sources": relevant,
            "source_type": source_type,
            "retrieval_latency_ms": round(
                retrieval_time,
                2
            ),
            "generation_latency_ms": round(
                generation_time,
                2
            ),
            "total_latency_ms": round(
                total_time,
                2
            )
        }

    total_time = (
        time.perf_counter() - total_start
    ) * 1000

    print()
    print("========== FINAL ANSWER ==========")
    print("Answer:", answer)
    print("Source:", source_type)
    print("==================================")
    print()

    print("========== LATENCY ==========")
    print(
        "Retrieval:",
        round(retrieval_time, 2),
        "ms"
    )
    print(
        "Generation:",
        round(generation_time, 2),
        "ms"
    )
    print(
        "Total:",
        round(total_time, 2),
        "ms"
    )
    print("=============================")
    print()

    return {
        "transcript": transcript,
        "language": language,
        "language_code": target_language,
        "answer": answer,
        "sources": relevant,
        "source_type": source_type,
        "retrieval_latency_ms": round(
            retrieval_time,
            2
        ),
        "generation_latency_ms": round(
            generation_time,
            2
        ),
        "total_latency_ms": round(
            total_time,
            2
        )
    }


@app.post("/ask")
def ask(
    payload: AskRequest
):
    return answer_question(
        payload.question,
        payload.language
    )


@app.get("/health")
def health():
    return {
        "status": "ok"
    }