Retrieval Playbook
==================

Overview
- Single-pass retrieval (no query rewrite): `RAGPipeline.query` calls `retrieve` once, applies adaptive filters, then trims to the final context limit before generation.
- Providers: embeddings use local `BAAI/bge-m3` cached in `./models`.
- Modes: dense-only or hybrid (dense + BM25) based on `HYBRID_SEARCH_ENABLED` in `.env`.

Key settings (src/retrieval.py)
- Threshold: `RAG_SCORE_THRESHOLD=0.4` gates low-scoring hits early.
- Fan-out: `fetch_k = max(50, top_k * 10)`; with `RAG_TOP_K=20` this yields a 200-candidate pool before filters/rerank.
- Hybrid search: dense prefetch + BM25 prefetch fused via RRF when enabled; otherwise dense-only.
- Hybrid prefetch: `HYBRID_PREFETCH_LIMIT=100` caps dense/text prefetch size when hybrid is on.
- Section routing: light keyword routing (risk, MD&A, financials, controls, business, directors) to keep matching sections when possible.
- Boosting: tables get 1.5x, company/source/doc_id match 1.4x, section match 1.1x, table title/row/column/year match 1.2x; keyword hits can also boost scores.
- Tables: if too few tables are in the top results, extra table chunks from the same doc_ids are pulled to improve recall.
- Adaptive filters: optional trimming of low-score tails and per-doc caps (controlled by config flags).
- Rerank: optional Bedrock reranker (e.g., `amazon.rerank-v1:0`) controlled by `RERANK_ENABLED` and `RERANK_TOP_N=20` to define how many candidates the reranker scores.

Inputs and ingestion
- Ingestion consumes that JSONL, attaches metadata headers (doc/section/page/type), can run FP16 + length sorting for GPU efficiency, and builds BM25 indexes when hybrid search is enabled.
