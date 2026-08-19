import uuid
import re
import duckdb

from huggingface_hub import hf_hub_download
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

COLLECTION_NAME = "multilingual_rag"
QDRANT_PATH = "qdrant_data"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

RECORDS_PER_LANGUAGE = 3000
SHORT_CHUNK_CHARS = 800
LONG_CHUNK_CHARS = 500
OVERLAP_SENTENCES = 1

LANG_FILES = {
    "Hindi": "train/hintrain.parquet",
    "Marathi": "train/martrain.parquet",
    "Gujarati": "train/gujtrain.parquet",
}

model = SentenceTransformer(MODEL_NAME)

client = QdrantClient(
    path=QDRANT_PATH
)


def clean_text(text):
    text = str(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text):
    text = clean_text(text)

    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?।॥])\s+",
        text
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def chunk_text(text):
    text = clean_text(text)

    if not text:
        return []

    if len(text) <= SHORT_CHUNK_CHARS:
        return [
            {
                "text": text,
                "chunk_type": "paragraph",
                "chunk_index": 0
            }
        ]

    sentences = split_sentences(text)

    if len(sentences) <= 1:
        chunks = []

        start = 0
        index = 0

        while start < len(text):
            end = min(
                start + SHORT_CHUNK_CHARS,
                len(text)
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "text": chunk,
                    "chunk_type": "fixed_overlap",
                    "chunk_index": index
                })

            if end >= len(text):
                break

            start = max(
                end - 100,
                start + 1
            )

            index += 1

        return chunks

    chunks = []
    current = []
    current_length = 0
    index = 0

    for sentence in sentences:

        sentence_length = len(sentence)

        if (
            current
            and current_length + sentence_length + 1
            > LONG_CHUNK_CHARS
        ):
            chunk = " ".join(current).strip()

            chunks.append({
                "text": chunk,
                "chunk_type": "semantic_sentence",
                "chunk_index": index
            })

            overlap = current[
                -OVERLAP_SENTENCES:
            ]

            current = overlap + [sentence]

            current_length = sum(
                len(x)
                for x in current
            ) + max(
                len(current) - 1,
                0
            )

            index += 1

        else:
            current.append(sentence)

            current_length += (
                sentence_length
                + (1 if current_length else 0)
            )

    if current:
        chunk = " ".join(current).strip()

        chunks.append({
            "text": chunk,
            "chunk_type": "semantic_sentence",
            "chunk_index": index
        })

    return chunks


def load_msmarco_language(remote_path):

    local_path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        filename=remote_path,
        repo_type="dataset"
    )

    con = duckdb.connect()

    path = local_path.replace("\\", "/")

    query = f"""
        SELECT *
        FROM read_parquet('{path}')
        LIMIT {RECORDS_PER_LANGUAGE}
    """

    result = con.execute(query)

    columns = [
        x[0]
        for x in result.description
    ]

    rows = result.fetchall()

    con.close()

    return [
        dict(zip(columns, row))
        for row in rows
    ]


def extract_passage(record):

    passages = record.get("passages")

    if not passages:
        return clean_text(
            record.get("passage")
            or record.get("passage_text")
            or ""
        )

    selected = []
    others = []

    for passage in passages:

        if isinstance(passage, dict):

            text = clean_text(
                passage.get("passage_text")
                or passage.get("text")
                or ""
            )

            if not text:
                continue

            if passage.get("is_selected"):
                selected.append(text)
            else:
                others.append(text)

    chosen = (
        selected
        if selected
        else others[:2]
    )

    return clean_text(
        " ".join(chosen)
    )


def build_record_chunks(
    record,
    language,
    record_index
):

    question = clean_text(
        record.get("query")
        or record.get("question")
        or ""
    )

    answer = clean_text(
        record.get("Answer")
        or record.get("answer")
        or ""
    )

    passage = extract_passage(record)

    title = clean_text(
        record.get("title")
        or ""
    )

    if not passage:
        passage = answer

    if not passage:
        passage = question

    chunks = chunk_text(
        passage
    )

    output = []

    for chunk in chunks:

        text_parts = []

        if title:
            text_parts.append(
                f"Title: {title}"
            )

        if question:
            text_parts.append(
                f"Question: {question}"
            )

        if answer:
            text_parts.append(
                f"Answer: {answer}"
            )

        text_parts.append(
            f"Context: {chunk['text']}"
        )

        final_text = "\n".join(
            text_parts
        )

        output.append({
            "text": final_text,
            "question": question,
            "answer": answer,
            "paragraph": chunk["text"],
            "title": title,
            "language": language,
            "chunk_type": chunk["chunk_type"],
            "chunk_index": chunk["chunk_index"],
            "record_index": record_index,
            "source": "MSMARCO-XI"
        })

    return output


def load_indic_language(
    language,
    remote_path
):

    print()
    print(
        f"========== {language} =========="
    )

    records = load_msmarco_language(
        remote_path
    )

    print(
        f"Loaded {len(records)} records"
    )

    chunks = []

    for index, record in enumerate(
        records
    ):

        record_chunks = build_record_chunks(
            record,
            language,
            index
        )

        chunks.extend(
            record_chunks
        )

    print(
        f"{language}: "
        f"{len(chunks)} chunks created"
    )

    return chunks


def load_english():

    print()
    print(
        "========== English =========="
    )

    try:

        dataset = load_dataset(
            "ai4bharat/Indic-Rag-Suite",
            "en",
            split="train",
            streaming=True
        )

        chunks = []

        for index, item in enumerate(
            dataset
        ):

            question = clean_text(
                item.get("question")
            )

            answer = clean_text(
                item.get("answer")
            )

            paragraph = clean_text(
                item.get("paragraph")
            )

            title = clean_text(
                item.get("title")
            )

            if not question:
                continue

            if not answer:
                continue

            if not paragraph:
                paragraph = answer

            record = {
                "question": question,
                "answer": answer,
                "paragraph": paragraph,
                "title": title
            }

            record_chunks = build_record_chunks(
                record,
                "English",
                index
            )

            chunks.extend(
                record_chunks
            )

            if len(chunks) >= RECORDS_PER_LANGUAGE * 2:
                break

        print(
            f"English: "
            f"{len(chunks)} chunks created"
        )

        return chunks

    except Exception as e:

        print(
            "English dataset error:",
            repr(e)
        )

        return []


def create_collection():

    if client.collection_exists(
        COLLECTION_NAME
    ):

        client.delete_collection(
            COLLECTION_NAME
        )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    print(
        "Collection created."
    )


def upload_chunks(
    chunks,
    language
):

    if not chunks:
        return

    print()
    print(
        f"Embedding {language}: "
        f"{len(chunks)} chunks"
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    vectors = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    points = []

    for chunk, vector in zip(
        chunks,
        vectors
    ):

        payload = {
            "text": chunk["text"],
            "question": chunk["question"],
            "answer": chunk["answer"],
            "paragraph": chunk["paragraph"],
            "title": chunk["title"],
            "language": chunk["language"],
            "chunk_type": chunk["chunk_type"],
            "chunk_index": chunk["chunk_index"],
            "record_index": chunk["record_index"],
            "source": chunk["source"]
        }

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload=payload
            )
        )

    batch_size = 500

    for start in range(
        0,
        len(points),
        batch_size
    ):

        batch = points[
            start:start + batch_size
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )

        uploaded = min(
            start + batch_size,
            len(points)
        )

        print(
            f"{language}: "
            f"{uploaded}/{len(points)} uploaded"
        )


def main():

    print(
        "Creating multilingual RAG database..."
    )

    create_collection()

    english = load_english()

    upload_chunks(
        english,
        "English"
    )

    for language in [
        "Hindi",
        "Marathi",
        "Gujarati"
    ]:

        chunks = load_indic_language(
            language,
            LANG_FILES[language]
        )

        upload_chunks(
            chunks,
            language
        )

    print()
    print(
        "INGESTION COMPLETED"
    )
    print(
        "Collection:",
        COLLECTION_NAME
    )


if __name__ == "__main__":
    main()