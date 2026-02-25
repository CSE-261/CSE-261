"""
Build embedding samples aligned to requests.json.

Input:
- /app/requests/requests.json
- /app/original_text/v1.0-simplified_simplified-nq-train.jsonl

Output:
- /app/original_text/embedding_samples_from_requests.jsonl
- /app/output/requests_doc_missing.json
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlsplit


def normalize_query(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def source_key(url: str) -> str:
    raw = html.unescape((url or "").strip())
    if not raw:
        return ""
    parts = urlsplit(raw)
    path = "/" + "/".join([p for p in (parts.path or "").split("/") if p])
    qs = parse_qs(parts.query or "", keep_blank_values=True)
    title = (qs.get("title") or [""])[0].strip().lower()
    oldid = (qs.get("oldid") or [""])[0].strip()
    if title and oldid:
        return f"{parts.netloc.lower()}|{path.lower()}|title={title}|oldid={oldid}"
    return f"{parts.netloc.lower()}|{path.lower()}|{(parts.query or '').strip().lower()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build embedding samples from requests.")
    parser.add_argument("--requests", type=Path, default=Path("/app/requests/requests.json"))
    parser.add_argument(
        "--nq",
        type=Path,
        default=Path("/app/original_text/v1.0-simplified_simplified-nq-train.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/app/original_text/embedding_samples_from_requests.jsonl"),
    )
    parser.add_argument(
        "--missing-out",
        type=Path,
        default=Path("/app/output/requests_doc_missing.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requests = json.loads(args.requests.read_text(encoding="utf-8"))
    if not isinstance(requests, list):
        raise ValueError(f"requests must be a JSON list: {args.requests}")

    request_rows: List[Dict[str, Any]] = []
    needed_source_keys: set[str] = set()
    needed_query_keys: set[str] = set()
    for req in requests:
        rq = (req.get("query") or "").strip()
        rs = req.get("source") or ""
        sk = source_key(rs)
        qk = normalize_query(rq)
        request_rows.append(
            {
                "query": rq,
                "source": rs,
                "source_key": sk,
                "query_key": qk,
            }
        )
        if sk:
            needed_source_keys.add(sk)
        if qk:
            needed_query_keys.add(qk)

    # Memory-safe: only keep first matching candidate for keys used by requests.
    by_source: Dict[str, Dict[str, Any]] = {}
    by_query: Dict[str, Dict[str, Any]] = {}

    scanned = 0
    with args.nq.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            scanned += 1
            if scanned % 200000 == 0:
                print(
                    f"scanned={scanned} source_hits={len(by_source)}/{len(needed_source_keys)} "
                    f"query_hits={len(by_query)}/{len(needed_query_keys)}"
                )

            obj = json.loads(line)
            query = (obj.get("question_text") or "").strip()
            source = obj.get("document_url") or ""
            text = obj.get("document_text") or ""
            example_id = obj.get("example_id")
            if not query or example_id is None or not text:
                continue

            row = {
                "query": query,
                "example_id": str(example_id),
                "source": source,
                "text": text,
            }

            sk = source_key(source)
            if sk and sk in needed_source_keys and sk not in by_source:
                by_source[sk] = row

            qk = normalize_query(query)
            if qk and qk in needed_query_keys and qk not in by_query:
                by_query[qk] = row

            if len(by_source) == len(needed_source_keys) and len(by_query) == len(needed_query_keys):
                # Best case: every key used in requests already matched.
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.missing_out.parent.mkdir(parents=True, exist_ok=True)

    missing: List[Dict[str, Any]] = []
    matched = 0
    source_hits = 0
    query_hits = 0

    with args.out.open("w", encoding="utf-8") as out:
        for req in request_rows:
            rq = req["query"]
            rs = req["source"]
            if not rq:
                missing.append({"reason": "empty_query", "query": rq, "source": rs})
                continue

            selected: Optional[Dict[str, Any]] = None
            matched_by = ""

            sk = req["source_key"]
            if sk and sk in by_source:
                selected = by_source[sk]
                matched_by = "source"
                source_hits += 1

            if selected is None:
                qk = req["query_key"]
                if qk in by_query:
                    selected = by_query[qk]
                    matched_by = "query"
                    query_hits += 1

            if selected is None:
                missing.append({"reason": "no_match_in_nq", "query": rq, "source": rs})
                continue

            out.write(
                json.dumps(
                    {
                        "query": rq,
                        "example_id": selected["example_id"],
                        "source": selected["source"],
                        "text": selected["text"],
                        "matched_by": matched_by,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            matched += 1

    args.missing_out.write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"requests_total={len(requests)}")
    print(f"matched={matched}")
    print(f"source_hits={source_hits}")
    print(f"query_hits={query_hits}")
    print(f"missing={len(missing)}")
    print(f"output={args.out}")
    print(f"missing_output={args.missing_out}")


if __name__ == "__main__":
    main()
