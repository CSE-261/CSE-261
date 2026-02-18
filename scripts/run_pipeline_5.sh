#!/usr/bin/env bash
set -euo pipefail

# Mini pipeline for first 5 samples.
# 1) Chunk JSONL -> data/chunks_all.jsonl
# 2) Filter前5条 -> data/chunks_first5.jsonl
# 3) Ingest前5条
# 4) Evaluate检索（仅5条）

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

echo "--- Chunk JSONL (all 100) ---"
docker-compose run --rm -w //app/scripts rag-app python chunking_jsonl_naive.py

echo "--- Filter first 5 doc_ids -> chunks_first5.jsonl ---"
docker-compose exec -T rag-app python - <<'PY'
"""
Use requests_first5.json to derive the exact doc_ids (example_id) to keep,
so filtering in build_requests_from_nq.py won't desync with chunking.
Fallback: if mapping fails, fall back to first 5 unique doc_ids in chunks_all.jsonl.
"""
import json, pathlib

requests_path = pathlib.Path("/app/requests/requests_first5.json")
embedding_samples = pathlib.Path("/app/original_text/embedding_samples.jsonl")
chunks_all = pathlib.Path("/app/data/chunks_all.jsonl")
dst = pathlib.Path("/app/data/chunks_first5.jsonl")

# Build mapping from query -> example_id (doc_id) using embedding_samples
query_to_id = {}
with embedding_samples.open() as f:
    for line in f:
        obj = json.loads(line)
        q = obj.get("query", "").strip()
        doc_id = obj.get("example_id")
        if q and doc_id is not None:
            query_to_id[q] = str(doc_id)

# Collect requested doc_ids
keep_ids = []
with requests_path.open() as f:
    for req in json.load(f):
        q = (req.get("query") or "").strip()
        did = query_to_id.get(q)
        if did and did not in keep_ids:
            keep_ids.append(did)

if len(keep_ids) < 5:
    # Fallback: first 5 unique doc_ids in chunks_all
    with chunks_all.open() as fin:
        for line in fin:
            obj = json.loads(line)
            did = obj.get("metadata", {}).get("doc_id")
            if did and did not in keep_ids:
                keep_ids.append(did)
            if len(keep_ids) >= 5:
                break

keep_ids_set = set(keep_ids)
count = 0
with chunks_all.open() as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        if obj.get("metadata", {}).get("doc_id") in keep_ids_set:
            fout.write(line)
            count += 1

print(f"Collected doc_ids (from requests): {keep_ids}")
print(f"Wrote {count} chunks to {dst}")
PY

echo "--- Reset collection & ingest first 5 ---"
docker-compose exec -T rag-app python - <<'PY'
from qdrant_client import QdrantClient
c = QdrantClient(host="qdrant", port=6333)
try:
    c.delete_collection("documents")
    print("deleted collection")
except Exception as e:
    print("delete warn:", e)
PY

docker-compose exec -T rag-app python main.py ingest //app/data/chunks_first5.jsonl

echo "--- Evaluate retrieval on 5 samples ---"
docker-compose exec -T rag-app python scripts/evaluate_retrieval.py \
  --golden-path /app/requests/requests_first5.json \
  --top-k 10 --k-values 1 3 5 10 \
  --save-details /app/output/retrieval_details.json

echo "Done. Details -> output/retrieval_details.json"
