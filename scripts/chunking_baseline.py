from __future__ import annotations
import json
import re
import hashlib
import pathlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import fitz # PyMuPDF
import tabula
import tiktoken
import pandas as pd


CONFIG = {
    # Chunk sizing tuned for higher recall/precision tradeoff
    "min_chunk_length": 80,
    "max_chunk_tokens": 320,
    "chunk_overlap": 48, # ~15% overlap for ~320-token chunks
    
    # Docker paths so output lives in /app/data for ingest
    "input_dir": "/app/tat_docs_test/",
    "output_file": "/app/data/chunks_all.jsonl",
    "exclude_file": "./not_included.txt",
    "csv_output_dir": "/app/data/csvs/",
    
    "TABLES_ENABLED": True,

    # Parallelism (per-PDF); set to None to use os.cpu_count()
    "num_workers": None,
}

_enc = tiktoken.get_encoding("cl100k_base")

def estimate_tokens(s: str) -> int: 
    return len(_enc.encode(s))

def tok_split(s: str, n: int) -> List[str]:
    ids = _enc.encode(s)
    return [_enc.decode(ids[i:i+n]) for i in range(0, len(ids), n)]

def md5(s: str) -> str: 
    return hashlib.md5(s.encode("utf-8")).hexdigest()

@dataclass
class ParagraphBlock:
    text: str
    page: int
    section_path: str

@dataclass
class TableBlock:
    csv_file_name: str
    page: int
    table_id_for_page: int

HEADING_PATTERNS = [
    r"^\s*Table\s+of\s+Contents\b.*",
    r"^\s*Item\s+\d+[A-Z]?\b.*",
    r"^\s*Management[’']?s\s+Discussion\b.*",
    r"^\s*Consolidated\s+(Statements?|Statement)\b.*",
    r"^\s*Notes?\s+to\s+Consolidated\s+Financial\s+Statements\b.*",
    r"^\s*Report\s+of\s+Independent\b.*",
    r"^\s*Exhibit\s+Index\b.*",
    r"^\s*Signatures\b.*",
]

HEADING_RE = re.compile("|".join(HEADING_PATTERNS), re.I)

def _normalize_block_text(text: str) -> str:
    if not text: return ""
    t = text.replace("\r", "\n")
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _split_into_paragraphs(text: str) -> List[str]:
    if not text: return []
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p and p.strip()]

def _detect_columns(blocks: List[Tuple[float,float,float,float,str]], page_w: float) -> Optional[float]:
    if not blocks or len(blocks) < 8: return None
    mids = sorted([((b[0]+b[2])/2.0) for b in blocks])
    gaps = [(mids[i]-mids[i-1], (mids[i-1]+mids[i])/2.0) for i in range(1, len(mids))]
    if not gaps: return None
    gaps.sort(reverse=True, key=lambda x: x[0])
    best_gap, gutter_x = gaps[0]
    if best_gap < 0.08 * page_w: return None
    left_ct = sum(1 for b in blocks if (b[0]+b[2])/2.0 < gutter_x)
    right_ct = len(blocks) - left_ct
    if left_ct >= 3 and right_ct >= 3:
        return gutter_x
    return None

def _assign_blocks_to_columns(blocks: List[Tuple], gutter_x: float) -> Tuple[List, List]:
    left, right = [], []
    for b in blocks:
        midx = (b[0]+b[2])/2.0
        (left if midx < gutter_x else right).append(b)
    left.sort(key=lambda b: (round(b[1],2), round(b[0],2)))
    right.sort(key=lambda b: (round(b[1],2), round(b[0],2)))
    return left, right

def _blocks_to_paragraphs(blocks: List[Tuple]) -> List[str]:
    paras: List[str] = []
    for b in blocks:
        if len(b) < 5: continue
        raw = _normalize_block_text(b[4])
        if not raw: continue
        paras.extend(_split_into_paragraphs(raw))
    return paras

def extract_paragraphs_with_sections(pdf_path: str) -> List[ParagraphBlock]:
    blocks: List[ParagraphBlock] = []
    doc = fitz.open(pdf_path)
    current_section = "General"
    for pno, page in enumerate(doc, 1):
        try:
            bks = page.get_text("blocks")
        except Exception:
            text = page.get_text("text")
            chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
            for c in chunks:
                head = c.splitlines()[0].strip() if c else ""
                if HEADING_RE.match(head): current_section = head
                blocks.append(ParagraphBlock(text=c, page=pno, section_path=current_section))
            continue

        bks = [b for b in bks if len(b) >= 5 and (b[4] or "").strip()]
        gutter_x = _detect_columns(bks, page.rect.width)
        ordered_columns = _assign_blocks_to_columns(bks, gutter_x) if gutter_x else [sorted(bks, key=lambda b: (round(b[1],2), round(b[0],2)))]

        for col_blocks in ordered_columns:
            for c in _blocks_to_paragraphs(col_blocks):
                head = c.splitlines()[0].strip() if c else ""
                if HEADING_RE.match(head): current_section = head
                blocks.append(ParagraphBlock(text=c, page=pno, section_path=current_section))
    doc.close()
    return blocks

def extract_tables(pdf_path: str, doc_id: str) -> List[TableBlock]:
    if not CONFIG["TABLES_ENABLED"]:
        return []
    
    tbls: List[TableBlock] = []
    print(f"  -> Extracting tables with Tabula...")
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        for pno in range(total_pages):
            dfs = tabula.read_pdf(pdf_path, pages=pno+1, multiple_tables=True, stream=True, silent=True)
            if not dfs: continue
            
            for i, df in enumerate(dfs):
                if len(df) < 2: continue
                df = df.dropna(axis=1, how='all')
                df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False, na=False)]
                df = df.dropna(axis=0, how='all')
                df = df.fillna("")
                if df.empty: continue

                fname = f"{doc_id}_page_{pno+1}_table_{i}.csv"
                path = pathlib.Path(CONFIG["csv_output_dir"]) / fname
                df.to_csv(path, index=False)
                
                tbls.append(TableBlock(csv_file_name=str(path), page=pno+1, table_id_for_page=i))
    except Exception as e:
        print(f"  [Table Error] Failed to extract tables: {e}")
    return tbls

def chunk_paragraph(text: str, size: int, overlap: int) -> List[str]:
    if estimate_tokens(text) <= size:
        return [text]
    
    step = max(1, size - overlap)
    parts = tok_split(text, step)
    res = []
    prev_tail = ""
    for i, p in enumerate(parts):
        cur = (prev_tail + p) if i > 0 else p
        if estimate_tokens(cur) > size:
            cur = tok_split(cur, size)[0]
        res.append(cur)
        
        # Re-tokenize to get a clean overlap boundary
        tail_tokens = _enc.encode(cur)
        overlap_start_idx = max(0, len(tail_tokens) - overlap)
        prev_tail = _enc.decode(tail_tokens[overlap_start_idx:])

    return [r for r in res if r]

def build_chunks_for_report(pdf_path: str, doc_id: str) -> List[Dict]:
    text_chunks = []
    paras = extract_paragraphs_with_sections(pdf_path)
    order = 0
    for pb in paras:
        splits = chunk_paragraph(pb.text, CONFIG["max_chunk_tokens"], CONFIG["chunk_overlap"])
        for s in splits:
            if len(s) < CONFIG["min_chunk_length"]:
                continue
            md = {
                "doc_id": doc_id,
                "type": "text",
                "section_path": pb.section_path,
                "page": pb.page,
                "pages": [pb.page],
                "order": order,
                "tokens": estimate_tokens(s),
                "hash": md5(f"{doc_id}|text|{order}|{s[:50]}")
            }
            text_chunks.append({"text": s, "metadata": md})
            order += 1

    table_chunks = []
    if CONFIG["TABLES_ENABLED"]:
        tables = extract_tables(pdf_path, doc_id)
        for t_idx, tb in enumerate(tables):
            md = {
                "doc_id": doc_id,
                "type": "table",
                "page": tb.page,
                "pages": [tb.page],
                "order": order,
                "table_id": f"T{t_idx}",
                "section_path": "None", # Tables don't have sections from this method
                "tokens": 0,
                "hash": md5(f"{doc_id}|table|{tb.page}|{tb.table_id_for_page}")
            }
            table_chunks.append({"text": tb.csv_file_name, "metadata": md})
            order += 1
            
    all_chunks = text_chunks + table_chunks
    all_chunks.sort(key=lambda x: (x['metadata'].get('page', 0), x['metadata']['order']))
    return all_chunks

def _process_single_pdf(pdf_path_str: str) -> Tuple[str, List[Dict[str, Any]]]:
    pdf_path = pathlib.Path(pdf_path_str)
    doc_id = pdf_path.stem
    pathlib.Path(CONFIG["csv_output_dir"]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(CONFIG["output_file"]).parent.mkdir(parents=True, exist_ok=True)
    
    chunks = build_chunks_for_report(str(pdf_path), doc_id)
    return pdf_path.name, chunks

def main():
    data_folder = pathlib.Path(CONFIG["input_dir"])
    pdf_files = sorted(data_folder.glob("*.pdf"))
    
    exclude_files = set()
    if pathlib.Path(CONFIG["exclude_file"]).exists():
        with open(CONFIG["exclude_file"], 'r') as f:
            exclude_files = {line.strip() for line in f if line.strip()}
    
    filtered_pdf_files = [f for f in pdf_files if f.name not in exclude_files]
    
    print(f"Found {len(filtered_pdf_files)} PDFs. Mode: Baseline Integrated.")
    print(f"Config: MaxTokens={CONFIG['max_chunk_tokens']}, Overlap={CONFIG['chunk_overlap']}")
    
    total_count = 0
    pathlib.Path(CONFIG["output_file"]).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(CONFIG["csv_output_dir"]).mkdir(parents=True, exist_ok=True)

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
                table_count = sum(1 for c in chunks if c["metadata"]["type"] == "table")
                print(f"  -> Text Chunks: {text_count}")
                if CONFIG["TABLES_ENABLED"]:
                    print(f"  -> Table Chunks: {table_count}")

                for chunk in chunks:
                    f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_count += len(chunks)

    print(f"\nDone! Total chunks: {total_count}")
    print(f"Output saved to: {CONFIG['output_file']}")

if __name__ == "__main__":
    main()
