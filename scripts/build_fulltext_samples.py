"""
Build a 100-record JSONL from NQ train set, keeping full document_text for retrieval.
Input: /app/original_text/v1.0-simplified_simplified-nq-train.jsonl
Output: /app/original_text/embedding_samples.jsonl
"""
from __future__ import annotations
import json
import pathlib

SRC = pathlib.Path("/app/original_text/v1.0-simplified_simplified-nq-train.jsonl")
DST = pathlib.Path("/app/original_text/embedding_samples.jsonl")
LIMIT = 100


def main():
    DST.parent.mkdir(parents=True, exist_ok=True)
    with SRC.open() as f_in, DST.open("w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in, 1):
            obj = json.loads(line)
            rec = {
                "query": obj.get("question_text", "").strip(),
                "example_id": obj.get("example_id"),
                "source": obj.get("document_url", ""),
                "text": obj.get("document_text", ""),
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i >= LIMIT:
                break
    print(f"Wrote {LIMIT} records to {DST}")


if __name__ == "__main__":
    main()
