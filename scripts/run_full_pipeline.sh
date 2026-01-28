#!/usr/bin/env bash
set -euo pipefail

# Runs chunking + ingest + retrieval evaluation
# Usage: bash ./run_full_pipeline.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

echo "--- Chunk PDFs with chunk_revise_.py ---"
docker-compose run --rm -w //app/scripts rag-app python chunking_baseline.py

echo "--- Ingest generated chunks ---"
docker-compose exec rag-app python main.py ingest //app/data/chunks_all.jsonl

echo "--- Evaluate retrieval ---"
docker-compose exec rag-app python scripts/evaluate_retrieval.py   --golden-path requests/requests.json   --top-k 20 --k-values 1 3 5 10   --save-details //app/output/retrieval_details.json

echo "Pipeline completed successfully."
