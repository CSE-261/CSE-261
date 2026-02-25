#!/usr/bin/env bash
set -euo pipefail

# Runs chunking + ingest + retrieval evaluation
# Usage: bash ./run_full_pipeline.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

echo "--- Build embedding samples from requests.json ---"
docker-compose exec -T rag-app python scripts/build_embedding_samples_from_requests.py \
  --requests /app/requests/requests.json \
  --nq /app/original_text/v1.0-simplified_simplified-nq-train.jsonl \
  --out /app/original_text/embedding_samples_from_requests.jsonl \
  --missing-out /app/output/requests_doc_missing.json

echo "--- Chunk JSONL with chunking_jsonl_naive.py ---"
docker-compose exec -T rag-app python scripts/chunking_jsonl_naive.py \
  --input-file /app/original_text/embedding_samples_from_requests.jsonl \
  --output-file /app/data/chunks_all.jsonl

echo "--- Reset collection ---"
docker-compose exec -T rag-app python - <<'PY'
from qdrant_client import QdrantClient
c = QdrantClient(host="qdrant", port=6333)
try:
    c.delete_collection("documents")
    print("deleted collection")
except Exception as e:
    print("delete warn:", e)
PY

echo "--- Ingest generated chunks ---"
docker-compose exec -T rag-app python main.py ingest /app/data/chunks_all.jsonl

echo "--- Evaluate retrieval ---"
docker-compose exec -T rag-app python scripts/evaluate_retrieval.py \
  --golden-path /app/requests/requests.json \
  --top-k 20 --k-values 1 3 5 10 \
  --save-details /app/output/retrieval_details.json

echo "Pipeline completed successfully."
