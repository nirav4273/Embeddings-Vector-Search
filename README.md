# Embeddings & Vector Search

A minimal example that:

1. Converts text into embeddings using the **NVIDIA embeddings API**
   (`https://integrate.api.nvidia.com/v1/embeddings`), called with plain HTTP.
2. Stores those vectors in a **Pinecone** index.
3. Embeds a question and returns the most similar stored texts (vector search).

Managed with [`uv`](https://docs.astral.sh/uv/).

## What's in here

| File           | Purpose                                                        |
| -------------- | ------------------------------------------------------------- |
| `main.py`      | The whole example: `embed()` helper, sample corpus, upsert, query. |
| `.env.example` | Template for the API keys / settings you need.                |
| `pyproject.toml` | Dependencies: `requests`, `pinecone`, `python-dotenv`.      |

## Setup

```bash
uv sync                     # create the venv and install dependencies
cp .env.example .env        # then edit .env with your real values
```

Fill in `.env`:

- `NVIDIA_API_KEY` – your NVIDIA API key.
- `NVIDIA_EMBED_MODEL` – e.g. `nvidia/nv-embedqa-e5-v5` (1024 dims) or
  `nvidia/llama-3.2-nv-embedqa-1b-v2` (2048 dims). The script auto-detects the
  dimension from the API response, so any retrieval model works.
- `NVIDIA_EMBED_URL` – already set to the default; override only if needed.
- `PINECONE_API_KEY` – your Pinecone API key.
- `PINECONE_INDEX` – **leave blank for now** (see below).

## Step 1 – find the embedding dimension

```bash
uv run main.py
```

With `PINECONE_INDEX` blank, the script embeds the sample texts and prints, e.g.:

```
Embedding dimension: 1024
```

## Step 2 – create the Pinecone index

In the [Pinecone console](https://app.pinecone.io/), create an index:

- **Dimension**: the number printed in step 1 (must match the model).
- **Metric**: `cosine`.
- **Type**: serverless is fine.

Put the index name in `PINECONE_INDEX` in your `.env`.

## Step 3 – run the search

```bash
uv run main.py
```

Now the script upserts the 6 sample documents and runs a query:

```
Upserted 6 vectors into index 'text-search'.

Query: How do I brew a good cup of coffee?
Top matches:
  0.78  To brew a good cup of coffee, use freshly ground beans and water just off the boil.
  0.41  Sourdough bread rises using a live culture of wild yeast and lactic acid bacteria.
  0.32  Photosynthesis lets plants convert sunlight, water, and carbon dioxide into sugar.
```

(Scores will vary by model; the coffee sentence should rank first.)

## How it works

- **`input_type`** – NVIDIA retrieval-QA embedding models need to know whether text
  is a stored document (`"passage"`) or a search query (`"query"`). `main.py` passes
  the right one in each call.
- **Upsert** – each vector is stored with an `id` and `metadata.text` so the original
  sentence comes back with the search result.
- **Query** – `index.query(vector=..., top_k=3, include_metadata=True)` returns the
  nearest vectors by cosine similarity.

## Troubleshooting

| Symptom                              | Cause                                             |
| ------------------------------------ | ------------------------------------------------ |
| `401` from NVIDIA                    | Bad or missing `NVIDIA_API_KEY`.                 |
| Pinecone `400` about vector dimension | Index dimension ≠ model dimension — recreate the index. |
| Empty / poor matches                 | Index still filling, or a non-retrieval model.  |
