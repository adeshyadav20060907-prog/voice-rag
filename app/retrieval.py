import os
import re
import math
from collections import Counter

from qdrant_client import QdrantClient
from sarvamai import SarvamAI


# ============================================================
# CONFIG
# ============================================================

COLLECTION_NAME = "multilingual_rag"

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
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    text = str(text or "").lower()

    text = re.sub(
        r"[^\w\s\u0900-\u097f]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(text: str):
    text = normalize_text(text)

    if not text:
        return []

    return text.split()


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

        return LANGUAGE_CODES.get(
            language_code,
            "English"
        )

    except Exception as e:

        print(
            "Language detection error:",
            repr(e)
        )

        return "English"


# ============================================================
# SIMPLE MULTILINGUAL RELEVANCE SCORE
# ============================================================

def calculate_score(
    query_tokens,
    question,
    answer,
    paragraph,
    title
):

    query_set = set(query_tokens)

    if not query_set:
        return 0.0

    question_tokens = tokenize(question)
    answer_tokens = tokenize(answer)
    paragraph_tokens = tokenize(paragraph)
    title_tokens = tokenize(title)

    question_set = set(question_tokens)
    answer_set = set(answer_tokens)
    paragraph_set = set(paragraph_tokens)
    title_set = set(title_tokens)

    question_matches = len(
        query_set & question_set
    )

    answer_matches = len(
        query_set & answer_set
    )

    paragraph_matches = len(
        query_set & paragraph_set
    )

    title_matches = len(
        query_set & title_set
    )

    # Question is most important.
    score = 0.0

    score += (
        question_matches
        / max(len(query_set), 1)
    ) * 0.55

    score += (
        answer_matches
        / max(len(query_set), 1)
    ) * 0.20

    score += (
        paragraph_matches
        / max(len(query_set), 1)
    ) * 0.20

    score += (
        title_matches
        / max(len(query_set), 1)
    ) * 0.05

    # Small phrase bonus.
    normalized_query = normalize_text(
        " ".join(query_tokens)
    )

    normalized_question = normalize_text(
        question
    )

    if (
        normalized_query
        and normalized_query in normalized_question
    ):
        score += 0.25

    return min(score, 1.0)


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

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    try:

        # ----------------------------------------------------
        # Read payloads without loading SentenceTransformer.
        # ----------------------------------------------------

        offset = None

        candidates = []

        while True:

            points, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            for point in points:

                payload = point.payload or {}

                question = payload.get(
                    "question",
                    ""
                )

                answer = payload.get(
                    "answer",
                    ""
                )

                paragraph = payload.get(
                    "paragraph",
                    ""
                )

                title = payload.get(
                    "title",
                    ""
                )

                score = calculate_score(
                    query_tokens,
                    question,
                    answer,
                    paragraph,
                    title
                )

                if score <= 0:
                    continue

                candidates.append(
                    {
                        "text": payload.get(
                            "text",
                            ""
                        ),

                        "question": question,

                        "answer": answer,

                        "paragraph": paragraph,

                        "title": title,

                        "language": payload.get(
                            "language",
                            ""
                        ),

                        "source": payload.get(
                            "source",
                            ""
                        ),

                        "score": float(score)
                    }
                )

            if next_offset is None:
                break

            offset = next_offset

        # ----------------------------------------------------
        # Sort best matches first.
        # ----------------------------------------------------

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return candidates[:limit]

    except Exception as e:

        print()
        print(
            "========== SEARCH ERROR =========="
        )
        print(
            repr(e)
        )
        print(
            "=================================="
        )
        print()

        return []