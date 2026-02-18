"""
Build requests/requests.json from NQ simplified train set with contextual gold passages.

Rules:
- Take first 100 records from /app/original_text/v1.0-simplified_simplified-nq-train.jsonl
- Query: question_text
- Answer: short answer text if present, else long answer span, else empty
- Gold: one TEXT item with a context window around the annotated answer:
    * Prefer long_answer span; fallback to first short_answer span; fallback to first 200 tokens of document_text
    * Window = [start-80, end+80] tokens (clipped to doc bounds)
"""
from __future__ import annotations
import json
from pathlib import Path
from bs4 import BeautifulSoup

SRC = Path("/app/original_text/v1.0-simplified_simplified-nq-train.jsonl")
DST = Path("/app/requests/requests.json")
LIMIT = 100
# Keep gold snippets closer to chunked paragraph length; original chunker uses ~512-token max.
# A 80-token window (before/after combined) yields ~160-token spans, closer to paragraph chunks.
WINDOW = 80   # tokens to extend before/after the annotated span for evidence
ANSWER_MAX_WORDS = 20  # cap answer length

def strip_html(text: str) -> str:
    """HTML 清洗：与 chunking_jsonl_naive 的策略对齐，保留空格分隔。"""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(" ", strip=True)
    return " ".join(cleaned.split())


def extract_span_tokens(tokens, start, end, window=WINDOW):
    """Return a context window around start:end token span."""
    n = len(tokens)
    if start is None or end is None or start < 0 or end <= start or start >= n:
        return None
    a = max(0, start - window)
    b = min(n, end + window)
    return " ".join(tokens[a:b])


def build_entry(obj):
    tokens = obj.get("document_text", "").split()
    ann = (obj.get("annotations") or [{}])[0]
    short_answers = ann.get("short_answers") or []
    long_answer = ann.get("long_answer") or {}

    gold_items = []
    seen = set()

    def add_ctx(ctx: str):
        ctx_clean = strip_html(ctx)
        if ctx_clean and ctx_clean not in seen:
            seen.add(ctx_clean)
            gold_items.append({"content": ctx_clean, "type": "TEXT"})

    # pick spans for answer (use first short span) and evidence (all spans)
    answer_span = None
    if short_answers:
        sa0 = short_answers[0]
        answer_span = (sa0["start_token"], sa0["end_token"])
    elif long_answer.get("start_token") is not None:
        answer_span = (long_answer["start_token"], long_answer["end_token"])

    # answer text (short)
    answer_text = ""
    if answer_span:
        answer_text = " ".join(tokens[answer_span[0]: answer_span[1]])
    if answer_text:
        words = answer_text.split()
        if len(words) > ANSWER_MAX_WORDS:
            answer_text = " ".join(words[:ANSWER_MAX_WORDS])

    # long answer context (preferred evidence)
    if long_answer.get("start_token") is not None:
        ctx = extract_span_tokens(tokens, long_answer["start_token"], long_answer["end_token"])
        if ctx:
            add_ctx(ctx)

    # all short answers as evidence windows
    for sa in short_answers:
        ctx = extract_span_tokens(tokens, sa["start_token"], sa["end_token"])
        if ctx:
            add_ctx(ctx)

    # fallback evidence
    if not gold_items:
        add_ctx(" ".join(tokens[:200]))

    # Fallback: if answer empty, borrow first gold snippet
    if not answer_text and gold_items:
        answer_text = " ".join(gold_items[0]["content"].split()[:ANSWER_MAX_WORDS])

    return {
        "query": obj.get("question_text", "").strip(),
        "answer": answer_text,
        "source": obj.get("document_url", ""),
        "gold": gold_items,
    }


def main():
    DST.parent.mkdir(parents=True, exist_ok=True)
    out = []
    with SRC.open() as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            out.append(build_entry(obj))
            if i >= LIMIT:
                break
    DST.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} entries to {DST}")


if __name__ == "__main__":
    main()
