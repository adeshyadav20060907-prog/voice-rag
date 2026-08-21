import json
from qdrant_client import QdrantClient

COLLECTION_NAME = "multilingual_rag"
QDRANT_PATH = "qdrant_data"
OUTPUT_FILE = "app/data.json"

client = QdrantClient(path=QDRANT_PATH)

all_data = []
offset = None

while True:
    points, next_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=256,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )

    for point in points:
        all_data.append(point.payload or {})

    print("Loaded:", len(all_data))

    if next_offset is None:
        break

    offset = next_offset

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        all_data,
        f,
        ensure_ascii=False,
        indent=2
    )

print()
print("==============================")
print("Export completed!")
print("Records:", len(all_data))
print("Saved to:", OUTPUT_FILE)
print("==============================")