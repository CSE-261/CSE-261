# TAT-RAG

A production-ready Retrieval-Augmented Generation stack that pairs **Qdrant** vector search with configurable **local or cloud LLMs**. The project ships with a Streamlit UI, CLI utilities, and RAGAS-based evaluation to help you move from ingestion to quality monitoring quickly.

## Highlights
- **Flexible LLM backends** - run locally through Ollama or swap to AWS Bedrock/Gemini via `.env`.
- **Table-aware ingestion** - pre-chunked JSONL + CSV archives hydrate tabular context before embedding.
- **Context-enriched embeddings** - metadata headers (document, section, page, chunk type) are prefixed before encoding, while payload content stays clean for display.
- **Local GPU tuned ingestion** - FP16 batching, length-aware ordering, BM25 keyword indexes, and table normalization keep ingestion fast and retrieval precise.
- **One-click Streamlit UI** - chat with your data, inspect retrieved sources, and open originals.
- **CLI workflow** - ingest, query, and evaluate from `main.py`.
- **RAGAS evaluation** - measure answer quality with reproducible cached responses.
- **Docker-first setup** - all services orchestrated via `docker-compose`, GPU passthrough supported.

## Prerequisites
- Docker & Docker Compose
- Ollama running on the host (for local inference) or credentials for your chosen cloud provider
- Optional: NVIDIA GPU + drivers for accelerated embeddings/LLM

## Quick Start
```bash
# 1. Configure environment
cp .env.example .env
# edit .env to point at your Ollama/GPU settings as needed
# optional: enable hybrid retrieval by setting HYBRID_SEARCH_ENABLED=true

# 2. Launch services
docker-compose up -d

# (Optional) Reset Qdrant collection before re-ingesting
docker-compose exec rag-app python -c "from qdrant_client import QdrantClient; QdrantClient(host='qdrant', port=6333).delete_collection('documents')"

# 3. (If you need to create chunks from PDFs) Run the chunker
# Make sure docker-compose mounts your PDFs: ./tat_docs_test -> /app/tat_docs_test
docker-compose run --rm -w /app/scripts rag-app python chunking.py
# Output: /app/data/chunks_all.jsonl and /app/data/csvs/

# 4. Ingest pre-chunked data (JSONL + CSVs on disk)
docker-compose exec rag-app python main.py ingest /app/data/chunks_all.jsonl

# 5. Run a query from the CLI
docker-compose exec rag-app python main.py query \
  --top-k 5 \
  "What is the A10 Networks' total cost of revenue in 2019?"

# 6. (Optional) Open the Streamlit chat UI
docker-compose exec rag-app streamlit run app.py --server.port 8501 --server.address 0.0.0.0
# visit http://localhost:8501
```

## Eval
```bash
# Evaluate with a golden set
# docker-compose exec rag-app python main.py evaluate \
#   --json-path /app/golden_set/golden_set.json \
#   --json-question-key query \
#   --json-answer-key answer \
#   --output /app/output/report.txt

# Evaluate retriever
docker-compose exec rag-app python scripts/evaluate_retrieval.py \
  --golden-path requests/requests.json \
  --top-k 10 --k-values 1 3 5 10 \
  --save-details /app/output/retrieval_details.json


# Evaluate with request.json
docker-compose exec rag-app python main.py evaluate \
  --json-path /app/requests/requests.json \
  --json-question-key query \
  --json-answer-key answer \
  --output /app/output/report.txt

docker-compose exec rag-app python scripts/latency_eval.py \
  --input //app/requests/requests.json \
  --limit 10 \
  --top-k 20 \
  --price-in 0.001 \
  --price-out 0.005 \
  --output //app/output/latency_eval.json
```




## Configuration
Update `.env` to control the pipeline:

- `LLM_PROVIDER`, `LLM_MODEL`, `OLLAMA_BASE_URL`: select the generator model. Increase `LLM_MAX_TOKENS` when using thinking-capable models.
- `GEMINI_API_KEY`, `LLM_REQUEST_INTERVAL`: required when `LLM_PROVIDER=gemini`. Set a small interval (e.g., 6s) to avoid API throttling during evaluation.
- `DENSE_BATCH_SIZE`, `USE_FP16`, `SORT_BY_LENGTH`: tune GPU-friendly ingestion; large batches + FP16 keep the GPU busy while length sorting reduces padding waste.
- `RAG_TOP_K`, `RAG_SCORE_THRESHOLD`, `RAG_FINAL_CONTEXT_LIMIT`: retrieval fan-out, similarity gating, and how many reranked contexts actually reach the LLM prompt.
- `RERANK_ENABLED`, `RERANK_MODEL`, `RERANK_TOP_N`, `RERANK_REQUEST_INTERVAL`: enable optional Bedrock reranking (e.g., `amazon.rerank-v1:0`) and tune how many docs are scored + how long to pause between calls to avoid throttling.
- `HYBRID_SEARCH_ENABLED`: set to `true` to fuse vector search with BM25 keyword search via Qdrant's Fusion API (requires re-ingestion after toggling).
- `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`, `EMBEDDING_CACHE_DIR`: embedding backend configuration.
- `QDRANT_HOST`, `QDRANT_COLLECTION`: vector store parameters (overridden inside Docker to `qdrant`).

Restart the `rag-app` service after changing environment variables:
```bash
docker-compose restart rag-app
```

## Project Structure
```
TAT-RAG/
|-- app.py                 # Streamlit UI for chat + source inspection
|-- docker-compose.yml     # Orchestrates Qdrant and rag-app containers
|-- Dockerfile             # Builds the Python runtime used by rag-app
|-- main.py                # CLI entry point (ingest / query / evaluate)
|-- requirements.txt       # Python dependencies
|-- .env                   # Runtime configuration (gitignored)
|-- data/                  # Sample ingestion archives (mounted into container)
|-- golden_set/            # Evaluation datasets
|-- models/                # Local embedding/LLM cache (mounted)
|-- output/                # Generated evaluation and cache artifacts
|-- scripts/
|   `-- precompute_answers.py  # Cache LLM outputs for RAGAS runs
`-- src/
    |-- __init__.py
    |-- config.py              # Dataclass-driven configuration loader
    |-- embedding_providers.py # Embedding abstraction (local BGE)
    |-- evaluation.py          # RAGAS evaluator and reporting helpers
    |-- ingestion.py           # Loads archives, prefixes metadata headers, FP16 batches, hybrid indexes, table normalization
    |-- llm_providers.py       # Ollama / Bedrock / Gemini adapters
    |-- retrieval.py           # Core RAG pipeline (retrieve + generate)
    `-- parsers/               # Legacy PDF parsers kept for reference
```

## Ingestion Optimizations at a Glance
- **Local GPU tuned batches** – `_optimize_embedding_provider()` reads `.env`’s `DENSE_BATCH_SIZE`, flips the encoder to FP16 when `USE_FP16=true`, and feeds large batches to maximize throughput while lowering VRAM usage.
- **Length-aware packing** – `_sort_by_length` groups chunks by similar length before encoding, which trims padding overhead and keeps the GPU busy.
- **Context headers** – `_build_context_header` + `_add_metadata_context` prepend document/section/page/type metadata so embeddings remain distinguishable while payload `content` stays raw for display.
- **Hybrid-ready indexes** – `_ensure_content_index` and `_ensure_metadata_indexes` automatically create BM25 text and keyword indexes; hybrid dense/BM25 search works without extra setup.
- **Pipelined ingestion** – queues plus a `ThreadPoolExecutor` balance I/O and GPU compute, preventing stalls when streaming large archives.
- **Table normalization** – CSV references are parsed into `|`-separated rows so tabular data becomes searchable chunks aligned with `FILE` references in golden sets.

Together, these features make ingestion faster than stock SentenceTransformers and yield cleaner, more retrievable vectors—as long as `.env` keeps `USE_FP16`, `DENSE_BATCH_SIZE`, `SORT_BY_LENGTH`, and hybrid options enabled where applicable.

## Evaluating with RAGAS

The resulting report summarises faithfulness, context precision/recall, answer relevancy, and correctness.

## Docker Services
| Service   | Purpose                          | Ports        |
|-----------|----------------------------------|--------------|
| `qdrant`  | Vector database + web UI         | 6333, 6334   |
| `rag-app` | Python RAG pipeline + Streamlit  | 8501 (UI)    |

GPU passthrough is pre-configured in `docker-compose.yml`. Comment the `deploy.resources` block if you do not have an NVIDIA GPU.

## Tech Stack
- **Vector DB:** Qdrant (cosine similarity)
- **Embeddings:** BAAI BGE-M3 (SentenceTransformers)
- **LLMs:** Ollama (qwen3 8b) with optional Bedrock/Gemini
- **Evaluation:** RAGAS (gemini-2.5-flash-lite)
- **Frontend:** Streamlit

## Acknowledgements
- [Ollama](https://ollama.com/)
- [Qdrant](https://qdrant.tech/)
- [BAAI BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [RAGAS](https://github.com/explodinggradients/ragas)
