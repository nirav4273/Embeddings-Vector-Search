"""Minimal embeddings + vector search example.

Flow:
  1. Turn text into embeddings with the NVIDIA embeddings API (raw HTTP).
  2. Store those vectors in a Pinecone index (upsert).
  3. Embed a question and ask Pinecone for the most similar stored texts.

Run it once with PINECONE_INDEX unset to discover the embedding dimension,
create a Pinecone index with that dimension + the "cosine" metric, then run
again to see search results.
"""

import os
import sys

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL")
NVIDIA_EMBED_URL = os.getenv(
    "NVIDIA_EMBED_URL", "https://integrate.api.nvidia.com/v1/embeddings"
)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

for name, value in {
    "NVIDIA_API_KEY": NVIDIA_API_KEY,
    "NVIDIA_EMBED_MODEL": NVIDIA_EMBED_MODEL,
    "PINECONE_API_KEY": PINECONE_API_KEY,
}.items():
    if not value:
        sys.exit(f"Missing required environment variable: {name} (see .env.example)")


# ---------------------------------------------------------------------------
# 2. Embedding helper
# ---------------------------------------------------------------------------
def embed(texts, input_type):
    """Return a list of embedding vectors for `texts`.

    input_type is "passage" when embedding documents to store, and "query"
    when embedding a search query. NVIDIA retrieval-QA models require it.
    """
    response = requests.post(
        NVIDIA_EMBED_URL,
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "input": texts,
            "model": NVIDIA_EMBED_MODEL,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        },
        timeout=30,
    )
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["data"]]


# ---------------------------------------------------------------------------
# 3. Sample corpus (kept inline so the example is self-contained)
# ---------------------------------------------------------------------------
CORPUS = [
    "To brew a good cup of coffee, use freshly ground beans and water just off the boil.",
    "Photosynthesis lets plants convert sunlight, water, and carbon dioxide into sugar.",
    "The Eiffel Tower in Paris was completed in 1889 and stands about 330 metres tall.",
    "TCP/IP is the set of protocols that lets computers exchange data over the internet.",
    "Sourdough bread rises using a live culture of wild yeast and lactic acid bacteria.",
    "A black hole is a region of spacetime where gravity is so strong that light cannot escape.",
]


# ---------------------------------------------------------------------------
# 4. Embed the corpus and report the vector dimension
# ---------------------------------------------------------------------------
print(f"Embedding {len(CORPUS)} documents with {NVIDIA_EMBED_MODEL} ...")
doc_vectors = embed(CORPUS, input_type="passage")
dimension = len(doc_vectors[0])
print(f"Embedding dimension: {dimension}")

if not PINECONE_INDEX:
    print(
        "\nPINECONE_INDEX is not set. Create a Pinecone index with:\n"
        f"  dimension = {dimension}\n"
        "  metric    = cosine\n"
        "then set PINECONE_INDEX in your .env and run again."
    )
    sys.exit(0)


# ---------------------------------------------------------------------------
# 5. Upsert vectors into Pinecone
# ---------------------------------------------------------------------------
from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

index.upsert(
    vectors=[
        {"id": f"doc-{i}", "values": vector, "metadata": {"text": text}}
        for i, (text, vector) in enumerate(zip(CORPUS, doc_vectors))
    ]
)
print(f"Upserted {len(CORPUS)} vectors into index '{PINECONE_INDEX}'.")


# ---------------------------------------------------------------------------
# 6. Search
# ---------------------------------------------------------------------------
QUESTION = "How do I brew a good cup of coffee?"
query_vector = embed([QUESTION], input_type="query")[0]

result = index.query(vector=query_vector, top_k=3, include_metadata=True)

print(f"\nQuery: {QUESTION}\nTop matches:")
for match in result["matches"]:
    print(f"  {match['score']:.3f}  {match['metadata']['text']}")
