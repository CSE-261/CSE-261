"""
Naive chunker for pre-tokenized JSONL (e.g., embedding_samples.jsonl).
Splits the `text` field into ~512-token chunks with 30% overlap, mirroring chunking_naive.py.
"""
from __future__ import annotations
import json
import hashlib
import pathlib
from typing import List, Dict, Any

import tiktoken

CONFIG = {
    "input_file": "/app/original_text/embedding_samples.jsonl",
    "output_file": "/app/data/chunks_all.jsonl",
    "max_chunk_tokens": 512,
    "chunk_overlap": int(512 * 0.3),  # 30% overlap
}

_enc = tiktoken.get_encoding("cl100k_base")


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    tokens = _enc.encode(text or "")
    if not tokens:
        return []
    step = max(1, size - overlap)
    chunks: List[str] = []
    idx = 0
    while idx < len(tokens):
        window = tokens[idx : idx + size]
        if not window:
            break
        chunks.append(_enc.decode(window))
        idx += step
    return chunks


def process_record(obj: Dict[str, Any], order_base: int) -> List[Dict[str, Any]]:
    doc_id = str(obj.get("example_id") or obj.get("source") or obj.get("query") or order_base)
    source = obj.get("source") or doc_id
    text = obj.get("text") or obj.get("document_text") or ""

    results: List[Dict[str, Any]] = []
    for offset, chunk in enumerate(
        chunk_text(text, CONFIG["max_chunk_tokens"], CONFIG["chunk_overlap"])
    ):
        token_count = len(_enc.encode(chunk))
        md = {
            "doc_id": doc_id,
            "source": source,
            "type": "text",
            "section_path": "General",
            "page": 1,
            "pages": [1],
            "order": order_base + offset,
            "tokens": token_count,
            "hash": md5(f"{doc_id}|{order_base + offset}|{chunk[:50]}"),
            "parser": "jsonl_naive",
            "query": obj.get("query", ""),
        }
        results.append({"text": chunk, "metadata": md})
    return results


def main():
    src = pathlib.Path(CONFIG["input_file"])
    dst = pathlib.Path(CONFIG["output_file"])
    dst.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in, 1):
            obj = json.loads(line)
            chunks = process_record(obj, order_base=idx * 100000)  # keep per-record ordering separated
            for ch in chunks:
                f_out.write(json.dumps(ch, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)
            if idx % 10 == 0:
                print(f"[{idx}] records processed, chunks so far: {total_chunks}")

    print(f"Done. Records: {idx}, Total chunks: {total_chunks}")
    print(f"Output -> {dst}")


if __name__ == "__main__":
    main()
