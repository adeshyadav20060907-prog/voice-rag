import json
import os
import re
from functools import lru_cache

from sarvamai import SarvamAI


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "data.json"
)


# ============================================================
# SARVAM
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
# LANGUAGE MAP
# ============================================================

LANGUAGE_CODES = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
}


# ============================================================
# LOAD DATA
# ============================================================

@lru_cache(maxsize=1)
def load_documents():

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Data file not found: {DATA_FILE}"
        )

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "data.json must contain a list of documents."
        )

    print(
        f"Loaded {len(data)} documents from data.json"
    )

    return data


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

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


def tokenize(text):

    text = normalize_text(text)

    if not text:
        return set()

    return set(text.split())


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):

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
# SCORE DOCUMENT
# ============================================================

def score_document(
    query_tokens,
    question,
    answer,
    paragraph,
    title
):

    if not query_tokens:
        return 0.0

    question_tokens = tokenize(question)
    answer_tokens = tokenize(answer)
    paragraph_tokens = tokenize(paragraph)
    title_tokens = tokenize(title)

    question_matches = len(
        query_tokens & question_tokens
    )

    answer_matches = len(
        query_tokens & answer_tokens
    )

    paragraph_matches = len(
        query_tokens & paragraph_tokens
    )

    title_matches = len(
        query_tokens & title_tokens
    )

    total = len(query_tokens)

    score = 0.0

    # Question gets highest importance.
    score += (
        question_matches / total
    ) * 0.60

    # Answer.
    score += (
        answer_matches / total
    ) * 0.15

    # Paragraph.
    score += (
        paragraph_matches / total
    ) * 0.15

    # Title.
    score += (
        title_matches / total
    ) * 0.10

    # Exact phrase bonus.
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
        score += 0.20

    return min(score, 1.0)


# ============================================================
# DOCUMENT SEARCH
# ============================================================

def search_documents(
    query: str,
    limit: int = 5
):

    query = (query or "").strip()

    if not query:
        return []

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    try:

        documents = load_documents()

    except Exception as e:

        print(
            "DATA LOAD ERROR:",
            repr(e)
        )

        return []

    results = []

    for document in documents:

        question = document.get(
            "question",
            ""
        )

        answer = document.get(
            "answer",
            ""
        )

        paragraph = document.get(
            "paragraph",
            ""
        )

        title = document.get(
            "title",
            ""
        )

        score = score_document(
            query_tokens,
            question,
            answer,
            paragraph,
            title
        )

        if score <= 0:
            continue

        results.append(
            {
                "text": document.get(
                    "text",
                    ""
                ),

                "question": question,

                "answer": answer,

                "paragraph": paragraph,

                "title": title,

                "language": document.get(
                    "language",
                    ""
                ),

                "source": document.get(
                    "source",
                    ""
                ),

                "score": float(score)
            }
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:limit]