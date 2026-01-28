#!/usr/bin/env python
"""Run a small query set and report latency stats and token cost estimates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAGConfig
from src.retrieval import RAGPipeline


def _percentile(values: List[float], pct: float) -> float:
    """Compute percentile (0-100) with linear interpolation."""
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    d0 = ordered[f] * (c - k)
    d1 = ordered[c] * (k - f)
    return d0 + d1


def load_queries(path: Path, limit: int) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = []
    for row in data:
        query = row.get("query")
        if not query:
            continue
        items.append({"query": query, "raw": row})
        if 0 < limit <= len(items):
            break
    return items


def compute_cost(
    prompt_tokens: int,
    completion_tokens: int,
    price_in_per_1k: float,
    price_out_per_1k: float,
) -> float:
    return (prompt_tokens / 1000.0) * price_in_per_1k + (completion_tokens / 1000.0) * price_out_per_1k


def main() -> None:
    parser = argparse.ArgumentParser(description="Run latency/cost probe over a small query set.")
    parser.add_argument("--input", type=Path, default=Path("/app/requests/requests.json"), help="Path to requests.json.")
    parser.add_argument("--limit", type=int, default=10, help="Number of queries to run (default: 10).")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top_k.")
    parser.add_argument("--score-threshold", type=float, default=None, help="Optional score threshold override.")
    parser.add_argument("--price-in", type=float, default=0.0008, help="USD per 1K prompt tokens.")
    parser.add_argument("--price-out", type=float, default=0.004, help="USD per 1K completion tokens.")
    parser.add_argument("--output", type=Path, default=Path("/app/output/latency_eval.json"), help="Where to write JSON results.")
    args = parser.parse_args()

    queries = load_queries(args.input, args.limit)
    if not queries:
        raise ValueError("No queries found in input file.")

    config = RAGConfig()
    pipeline = RAGPipeline(config)
    retrieval_k = args.top_k or config.top_k
    score_threshold = args.score_threshold if args.score_threshold is not None else config.score_threshold

    per_query: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []
    costs: List[float] = []

    for idx, item in enumerate(queries, 1):
        q = item["query"]
        t0 = time.perf_counter()
        result = pipeline.query(q, top_k=retrieval_k, score_threshold=score_threshold)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        usage = result.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        cost = compute_cost(prompt_tokens, completion_tokens, args.price_in, args.price_out)

        latencies_ms.append(latency_ms)
        costs.append(cost)

        per_query.append(
            {
                "query": q,
                "answer": result.get("answer"),
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost,
                "model": result.get("model"),
            }
        )

        print(f"[{idx}/{len(queries)}] {q[:60]}...  latency={latency_ms:.1f} ms  cost=${cost:.6f}")

    total_queries = len(per_query)
    avg_latency = sum(latencies_ms) / total_queries
    p50_latency = _percentile(latencies_ms, 50)
    p90_latency = _percentile(latencies_ms, 90)
    p99_latency = _percentile(latencies_ms, 99)
    total_cost = sum(costs)
    avg_cost = total_cost / total_queries

    summary = {
        "total_queries": total_queries,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p90_latency_ms": p90_latency,
        "p99_latency_ms": p99_latency,
        "avg_cost_per_query_usd": avg_cost,
        "total_cost_usd": total_cost,
        "price_in_per_1k": args.price_in,
        "price_out_per_1k": args.price_out,
        "retrieval_top_k": retrieval_k,
        "score_threshold": score_threshold,
    }

    output_payload = {"summary": summary, "per_query": per_query}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nLatency/Cost Summary")
    print("=" * 24)
    print(f"Avg latency: {avg_latency:.1f} ms | P50: {p50_latency:.1f} | P90: {p90_latency:.1f} | P99: {p99_latency:.1f}")
    print(f"Avg cost/query: ${avg_cost:.6f} | Total: ${total_cost:.6f}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
