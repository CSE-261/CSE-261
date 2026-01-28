from __future__ import annotations
import json
import hashlib
import pathlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any

import fitz
import tiktoken

CONFIG = {
    "max_chunk_tokens": 512,
    "chunk_overlap": int(512 * 0.3),  # 30% overlap
    "input_dir": "/app/tat_docs_test/",
    "output_file": "/app/data/chunks_all.jsonl",
    "exclude_file": "./not_included.txt",
    "num_workers": None,
}

_enc = tiktoken.get_encoding("cl100k_base")

def estimate_tokens(s: str) -> int:
    return len(_enc.encode(s))


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


def build_chunks_for_pdf(pdf_path: str, doc_id: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    order = 0
    results: List[Dict[str, Any]] = []

    for page_no, page in enumerate(doc, 1):
        text = page.get_text("text") or ""
        for chunk in chunk_text(text, CONFIG["max_chunk_tokens"], CONFIG["chunk_overlap"]):
            token_count = estimate_tokens(chunk)
            md = {
                "doc_id": doc_id,
                "type": "text",
                "section_path": "General",
                "page": page_no,
                "pages": [page_no],
                "order": order,
                "tokens": token_count,
                "hash": md5(f"{doc_id}|text|{order}|{chunk[:50]}")
            }
            results.append({"text": chunk, "metadata": md})
            order += 1

    doc.close()
    return results


def _process_single_pdf(pdf_path_str: str) -> Tuple[str, List[Dict[str, Any]]]:
    pdf_path = pathlib.Path(pdf_path_str)
    doc_id = pdf_path.stem
    pathlib.Path(CONFIG["output_file"]).parent.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks_for_pdf(str(pdf_path), doc_id)
    return pdf_path.name, chunks


def main():
    data_folder = pathlib.Path(CONFIG["input_dir"])
    pdf_files = sorted(data_folder.glob("*.pdf"))

    exclude_files = set()
    if pathlib.Path(CONFIG["exclude_file"]).exists():
        with open(CONFIG["exclude_file"], "r", encoding="utf-8") as f:
            exclude_files = {line.strip() for line in f if line.strip()}

    filtered_pdf_files = [f for f in pdf_files if f.name not in exclude_files]

    print(f"Found {len(filtered_pdf_files)} PDFs. Mode: Naive 512/30%.")
    print(f"Config: MaxTokens={CONFIG['max_chunk_tokens']}, Overlap={CONFIG['chunk_overlap']}")

    total_count = 0
    pathlib.Path(CONFIG["output_file"]).parent.mkdir(parents=True, exist_ok=True)

    worker_count = CONFIG["num_workers"] or os.cpu_count() or 1

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_pdf = {
            executor.submit(_process_single_pdf, str(pdf_path)): pdf_path.name
            for pdf_path in filtered_pdf_files
        }
        with open(CONFIG["output_file"], "w", encoding="utf-8") as f_out:
            for idx, future in enumerate(as_completed(future_to_pdf), 1):
                pdf_name = future_to_pdf[future]
                try:
                    pdf_name, chunks = future.result()
                except Exception as e:
                    print(f"[Error] {pdf_name}: {e}")
                    continue

                print(f"[{idx}/{len(filtered_pdf_files)}] {pdf_name}...")
                text_count = sum(1 for c in chunks if c["metadata"]["type"] == "text")
                print(f"  -> Text Chunks: {text_count}")

                for chunk in chunks:
                    f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_count += len(chunks)

    print(f"\nDone! Total chunks: {total_count}")
    print(f"Output saved to: {CONFIG['output_file']}")


if __name__ == "__main__":
    main()
