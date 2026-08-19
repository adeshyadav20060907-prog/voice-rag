
import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sarvamai import SarvamAI


# ============================================================
# CONFIG
# ============================================================

COLLECTION_NAME = "multilingual_rag"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

QDRANT_PATH = "qdrant_data"


# ============================================================
# SARVAM CLIENT
# ============================================================

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if not SARVAM_API_KEY:
    raise RuntimeError(
        "SARVAM_API_KEY environment variable is not set."
    )

sarvam_client = SarvamAI(
    api_subscription_key=SARVAM_API_KEY
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading retrieval model...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Retrieval model loaded.")


# ============================================================
# QDRANT
# ============================================================

client = QdrantClient(
    path=QDRANT_PATH
)


# ============================================================
# LANGUAGE MAP
# ============================================================

LANGUAGE_CODES = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
}


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text: str) -> str:

    text = (text or "").strip()

    if not text:
        return "English"

    try:

        response = sarvam_client.text.identify_language(
            input=text[:1000]
        )

        language_code = getattr(
            response,
            "language_code",
            None
        )

        detected_language = LANGUAGE_CODES.get(
            language_code
        )

        if detected_language:
            return detected_language

        return "English"

    except Exception as e:

        print(
            "Language detection error:",
            repr(e)
        )

        return "English"


# ============================================================
# DOCUMENT SEARCH
# ============================================================

def search_documents(
    query: str,
    limit: int = 3
):

    query = (query or "").strip()

    if not query:
        return []

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).tolist()

    # ========================================================
    # QDRANT SEARCH
    # ========================================================

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True
    ).points

    output = []

    # ========================================================
    # PROCESS RESULTS
    # ========================================================

    for result in results:

        payload = result.payload or {}

        output.append(
            {
                "text": payload.get(
                    "text",
                    ""
                ),

                "question": payload.get(
                    "question",
                    ""
                ),

                "answer": payload.get(
                    "answer",
                    ""
                ),

                "paragraph": payload.get(
                    "paragraph",
                    ""
                ),

                "title": payload.get(
                    "title",
                    ""
                ),

                "language": payload.get(
                    "language",
                    ""
                ),

                "source": payload.get(
                    "source",
                    ""
                ),

                "score": float(
                    result.score
                )
            }
        )

    return output
