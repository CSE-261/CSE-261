# Retrieval Evaluation Metrics (TAT-RAG)

This document explains the retrieval metrics emitted by `scripts/evaluate_retrieval.py` after the coverage-focused changes.

## Matching rule
- Normalize text: lowercase and collapse whitespace.
- A retrieved chunk covers a gold evidence span if either:
  - Exact/substring match after normalization, or
  - `SequenceMatcher` similarity >= `--fuzzy-threshold` (default `0.7`).
- Coverage is computed per query using only the top-k retrieved chunks for each metric's k.

## Common parameters (alignment with retrieval config)
- `RAG_TOP_K` (e.g., 20): how many hits the retriever returns; metrics use subsets of this list at each k.
- `RAG_SCORE_THRESHOLD` (e.g., 0.4): low-scoring hits below this are pruned before evaluation.
- `HYBRID_PREFETCH_LIMIT` (e.g., 100): caps dense/BM25 prefetch size when hybrid search is enabled.
- `RERANK_TOP_N` (e.g., 20): how many candidates the reranker re-scores before the final top-K slice.
- `--fuzzy-threshold` (default 0.7): similarity cutoff for counting a retrieved chunk as covering a gold evidence span; higher = stricter.

## Metrics
- Precision@5: Relevant chunks in top 5 / 5.
- EvidenceRecall@K (K=3,10): Unique gold evidences covered by top-K chunks / total gold evidences (micro).
- PerQueryCoverage@K (K=3,10): For each query, (covered gold evidences in top-K) / (total gold evidences); then averaged across queries (macro).
- FullCoverageRate@K (K=3,10): Share of queries where all gold evidences are covered by at least one chunk in the top-K.
- MAP: Mean Average Precision across queries (uses all retrieved ranks).
- MRR: Mean Reciprocal Rank of the first relevant chunk.
- HitRate@10: Fraction of queries with at least one relevant chunk in the top 10.

Adjust fuzzy matching if needed:
```bash
--fuzzy-threshold 0.7 # stricter overlap requirement
```

## Speed & cost (reporting guidance)
- Speed: log per-query latency at the entry point (start/end timestamps) and report avg/P50/P90/P99; optional subspans (retrieval, rerank, LLM generation) help pinpoint bottlenecks.
- Cost (priced at $/1K tokens): for each LLM call, `cost = ((prompt_tokens + completion_tokens) / 1000) * price_per_1k`; sum across calls and divide by query count for avg cost/query. If a paid reranker/embedding is used, add its per-call cost the same way.
