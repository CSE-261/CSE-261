Chunking scripts overview
========================

Paths: `scripts/chunking_baseline.py` and `scripts/chunking_naive.py`.

Baseline chunker (`scripts/chunking_baseline.py`)
- Token budget: ~320 tokens per chunk, ~15% overlap (48 tokens), minimum length 80 chars.
- Text parsing: page-level block parsing with multi-column detection; heading heuristics to tag `section_path`.
- Table handling: extracts tables via Tabula to CSVs and emits table chunks with metadata.
- Metadata: per-chunk hashes, page numbers, section names, doc ids; preserves ordering; parallel per-PDF.
- I/O: reads PDFs from `/app/tat_docs_test/`, writes chunks to `/app/data/chunks_all.jsonl`, tables to `/app/data/csvs/`; skips files listed in `not_included.txt`.

Naive chunker (`scripts/chunking_naive.py`)
- Token budget: 512 tokens per chunk, 30% overlap; no minimum length guard.
- Text parsing: per-page plain text (`page.get_text("text")`), no column detection or paragraph splitting.
- Table handling: none (no CSV export, no table chunks).
- Metadata: basic `doc_id`, page, tokens, order, hash; `section_path` fixed to `"General"`.
- I/O: same input/output paths and process pool structure as baseline; no exclude list handling beyond config defaults.


Ingestion technique (downstream)
- Generate chunks: run one of the chunkers to produce `/app/data/chunks_all.jsonl` (and CSVs for baseline).
- Ingest: `docker-compose exec rag-app python main.py ingest /app/data/chunks_all.jsonl`.
- Embeddings: local `BAAI/bge-m3` (configurable via `.env`), with optional FP16 and length sorting for GPU efficiency.
- Hybrid-ready indexes: ingestion prepends metadata headers to content for embeddings and builds BM25 indexes for hybrid search (controlled by `HYBRID_SEARCH_ENABLED`).
- Table normalization: baseline chunker CSVs are flattened to `|`-separated rows during ingest so tables remain searchable and align with golden references.
