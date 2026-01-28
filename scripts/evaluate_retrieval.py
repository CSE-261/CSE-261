#!/usr/bin/env python
"""Evaluate retriever quality against the golden set without involving the LLM."""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAGConfig

logger = logging.getLogger(__name__)

_MISSING_REFERENCE_LOG: set[str] = set()


def normalize_text(value: str) -> str:
    """Lowercase text and collapse whitespace for fuzzy containment checks."""
    if not value:
        return ""
    return " ".join(value.split()).lower()


def fuzzy_overlap_ratio(text_a: str, text_b: str) -> float:
    """Compute a len-weighted similarity ratio for fuzzy span alignment."""
    from difflib import SequenceMatcher

    a = normalize_text(text_a)
    b = normalize_text(text_b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def normalize_path(value: str) -> str:
    if not value:
        return ""
    return value.replace("\\", "/").lower()


def _read_csv_text_from_string(data: str) -> str:
    reader = csv.reader(io.StringIO(data))
    rows: List[str] = []
    for row in reader:
        joined = " | ".join(cell.strip() for cell in row)
        if joined:
            rows.append(joined)
    return "\n".join(rows)


def _read_csv_from_path(path: Path) -> str:
    rows: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            joined = " | ".join(cell.strip() for cell in row)
            if joined:
                rows.append(joined)
    return "\n".join(rows)


def load_reference_text(path_str: str, repo_root: Path) -> Optional[str]:
    """Load reference content for FILE entries from disk or bundled archives."""
    normalized = path_str.replace("\\", "/").strip()
    if not normalized:
        return None

    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return None

    normalized_variants = ["/".join(parts[i:]) for i in range(len(parts))]

    lower_repo_name = normalize_path(repo_root.name)
    first_lower = normalize_path(parts[0])
    if first_lower == lower_repo_name or first_lower == "tat-rag":
        normalized_variants.append("/".join(parts[1:]))

    normalized_variants = [variant for variant in normalized_variants if variant]
    normalized_variants = list(dict.fromkeys(normalized_variants))

    raw_path = Path(path_str)

    candidates: List[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    for variant in normalized_variants:
        variant_path = Path(variant)
        candidates.append(repo_root / variant_path)
        candidates.append(repo_root / "data" / variant_path)

    for candidate in candidates:
        if candidate.exists():
            if candidate.suffix.lower() == ".csv":
                return _read_csv_from_path(candidate)
            return candidate.read_text(encoding="utf-8")

    return None


@dataclass
class GoldItem:
    """Lightweight structure for a single golden reference."""

    index: int
    type: str
    normalized_text: str
    file_name: Optional[str]
    raw: Dict[str, Any]


@dataclass
class QuerySample:
    """Golden set entry augmented with normalized references."""

    query: str
    gold_items: List[GoldItem]


def load_golden_set(golden_path: Path) -> List[QuerySample]:
    repo_root = PROJECT_ROOT
    if not golden_path.is_absolute():
        golden_path = repo_root / golden_path

    with golden_path.open("r", encoding="utf-8") as handle:
        raw_entries = json.load(handle)

    samples: List[QuerySample] = []
    for entry in raw_entries:
        query = entry.get("query")
        if not query:
            logger.warning("Skipping entry without query: %s", entry)
            continue

        processed_items: List[GoldItem] = []
        for idx, gold in enumerate(entry.get("gold", [])):
            gold_type = (gold.get("type") or "TEXT").upper()
            file_name: Optional[str] = None
            normalized = ""

            if gold_type == "TEXT":
                normalized = normalize_text(gold.get("content", ""))
            elif gold_type == "FILE":
                reference_path = gold.get("content", "")
                if isinstance(reference_path, str) and reference_path:
                    file_name = Path(reference_path).name.lower()
                    loaded_text = load_reference_text(reference_path, repo_root)
                    if loaded_text:
                        normalized = normalize_text(loaded_text)
                    else:
                        canonical = reference_path.replace("\\", "/")
                        if canonical not in _MISSING_REFERENCE_LOG:
                            logger.warning("Referenced file not found for golden entry: %s", reference_path)
                            _MISSING_REFERENCE_LOG.add(canonical)
            else:
                logger.warning("Unsupported gold type '%s' in entry '%s'", gold_type, query)

            processed_items.append(
                GoldItem(
                    index=idx,
                    type=gold_type,
                    normalized_text=normalized,
                    file_name=file_name,
                    raw=gold,
                )
            )

        samples.append(QuerySample(query=query, gold_items=processed_items))

    return samples


def match_doc_to_gold(
    doc: Dict[str, Any],
    gold_items: Sequence[GoldItem],
    fuzzy_threshold: float,
) -> Optional[int]:
    doc_text = normalize_text(doc.get("content", ""))
    metadata = doc.get("metadata") or {}

    for item in gold_items:
        if item.normalized_text and doc_text:
            if item.normalized_text in doc_text or doc_text in item.normalized_text:
                return item.index
            if fuzzy_overlap_ratio(item.normalized_text, doc_text) >= fuzzy_threshold:
                return item.index

    meta_strings: List[str] = []
    for key in ("table_path", "table_title", "source", "doc_id", "section_path"):
        value = metadata.get(key)
        if isinstance(value, str):
            meta_strings.append(normalize_path(value))
    tags = metadata.get("tags")
    if isinstance(tags, list):
        meta_strings.extend(normalize_path(str(tag)) for tag in tags if isinstance(tag, (str, Path)))

    for item in gold_items:
        if not item.file_name:
            continue
        for meta_value in meta_strings:
            if item.file_name in meta_value:
                return item.index

    return None


def evaluate_query(
    retrieved_docs: List[Dict[str, Any]],
    gold_items: Sequence[GoldItem],
    k_values: Sequence[int],
    fuzzy_threshold: float,
) -> Dict[str, Any]:
    max_k = max(k_values) if k_values else 0
    total_gold = sum(1 for item in gold_items if item.normalized_text or item.file_name)
    if total_gold == 0:
        return {
            "precision_at_k": {k: 0.0 for k in k_values},
            "recall_at_k": {k: 0.0 for k in k_values},
            "evidence_recall_at_k": {k: 0.0 for k in k_values},
            "covered_at_k": {k: 0 for k in k_values},
            "total_gold": 0,
            "per_query_evidence_coverage": 0.0,
            "full_coverage_flags": {k: 0.0 for k in k_values},
            "average_precision": 0.0,
            "mrr": 0.0,
            "hit": 0.0,
            "matched": 0,
            "retrieved": len(retrieved_docs),
            "hit_at_k": {k: 0.0 for k in k_values},
        }

    matches: List[Optional[int]] = []
    for doc in retrieved_docs:
        matches.append(match_doc_to_gold(doc, gold_items, fuzzy_threshold))

    precision_at_k: Dict[int, float] = {}
    recall_at_k: Dict[int, float] = {}
    covered_at_k: Dict[int, int] = {}
    full_coverage_flags: Dict[int, float] = {}
    hit_flags: Dict[int, float] = {}
    for k in k_values:
        top_matches = matches[:k]
        denom = min(k, len(matches))
        relevant = sum(1 for match in top_matches if match is not None)
        precision_at_k[k] = relevant / denom if denom else 0.0
        unique_hits = {match for match in top_matches if match is not None}
        covered_at_k[k] = len(unique_hits)
        recall_at_k[k] = covered_at_k[k] / total_gold if total_gold else 0.0
        full_coverage_flags[k] = 1.0 if covered_at_k[k] >= total_gold and total_gold > 0 else 0.0
        hit_flags[k] = 1.0 if relevant > 0 else 0.0

    ap = 0.0
    seen_for_ap: set[int] = set()
    hits = 0
    for rank, match in enumerate(matches, start=1):
        if match is None or match in seen_for_ap:
            continue
        seen_for_ap.add(match)
        hits += 1
        ap += hits / rank
    average_precision = ap / total_gold if total_gold else 0.0

    reciprocal_rank = 0.0
    for rank, match in enumerate(matches, start=1):
        if match is not None:
            reciprocal_rank = 1.0 / rank
            break

    per_query_coverage = covered_at_k.get(max_k, 0) / total_gold if total_gold else 0.0

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "evidence_recall_at_k": recall_at_k,
        "covered_at_k": covered_at_k,
        "total_gold": total_gold,
        "average_precision": average_precision,
        "mrr": reciprocal_rank,
        "hit": 1.0 if reciprocal_rank > 0 else 0.0,
        "matched": covered_at_k.get(max_k, 0),
        "retrieved": len(retrieved_docs),
        "match_sequence": matches,
        "gold_coverage": per_query_coverage,
        "per_query_evidence_coverage": per_query_coverage,
        "full_coverage_flags": full_coverage_flags,
        "hit_at_k": hit_flags,
    }


def aggregate_metrics(per_query: List[Dict[str, Any]], k_values: Sequence[int]) -> Dict[str, Any]:
    total_queries = len(per_query)
    if total_queries == 0:
        return {}

    precision_totals = {k: 0.0 for k in k_values}
    recall_totals = {k: 0.0 for k in k_values}
    covered_totals = {k: 0.0 for k in k_values}
    full_coverage_totals = {k: 0.0 for k in k_values}
    hit_at_k_totals = {k: 0.0 for k in k_values}
    map_total = 0.0
    mrr_total = 0.0
    hit_total = 0.0
    coverage_total = 0.0
    total_gold = 0.0

    for result in per_query:
        for k in k_values:
            precision_totals[k] += result["precision_at_k"][k]
            recall_totals[k] += result["recall_at_k"][k]
            covered_totals[k] += result.get("covered_at_k", {}).get(k, 0)
            full_coverage_totals[k] += result.get("full_coverage_flags", {}).get(k, 0.0)
            hit_at_k_totals[k] += result.get("hit_at_k", {}).get(k, 0.0)
        map_total += result["average_precision"]
        mrr_total += result["mrr"]
        hit_total += result["hit"]
        coverage_total += result.get("gold_coverage", 0.0)
        total_gold += result.get("total_gold", 0.0)

    summary = {
        "precision_at_k": {k: precision_totals[k] / total_queries for k in k_values},
        "recall_at_k": {k: recall_totals[k] / total_queries for k in k_values},
        "evidence_recall_at_k": {
            k: (covered_totals[k] / total_gold) if total_gold else 0.0 for k in k_values
        },
        "mean_average_precision": map_total / total_queries,
        "mean_reciprocal_rank": mrr_total / total_queries,
        "hit_rate": hit_total / total_queries,
        "hit_rate_at_k": {k: hit_at_k_totals[k] / total_queries for k in k_values},
        "avg_gold_coverage": coverage_total / total_queries,
        "per_query_evidence_coverage": {
            k: recall_totals[k] / total_queries for k in k_values
        },
        "full_coverage_rate": {
            k: full_coverage_totals[k] / total_queries for k in k_values
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retriever using the annotated golden set.")
    parser.add_argument(
        "--golden-path",
        type=Path,
        default=Path("golden_set/golden_set.json"),
        help="Path to the golden set JSON file.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="Cutoffs for Precision@K/Recall@K.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Override number of documents to retrieve per query.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        help="Optional score threshold override for retrieval.",
    )
    parser.add_argument(
        "--save-details",
        type=Path,
        help="Optional path to store per-query retrieval details as JSON.",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.7,
        help="Similarity ratio (0-1) required to count a retrieved chunk as covering a gold evidence span.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    k_values = sorted({k for k in args.k_values if k > 0})
    if not k_values:
        raise ValueError("At least one positive k value is required.")

    samples = load_golden_set(Path(args.golden_path))
    if not samples:
        raise ValueError("Golden set is empty or invalid.")

    config = RAGConfig()
    from src.retrieval import RAGPipeline  # noqa: WPS433 (local import keeps optional dependency lazy)
    pipeline = RAGPipeline(config)

    retrieval_k = args.top_k or max(config.top_k, max(k_values))
    if retrieval_k < max(k_values):
        raise ValueError("Retrieval top_k must be >= max requested k.")

    score_threshold = args.score_threshold if args.score_threshold is not None else config.score_threshold

    per_query_results: List[Dict[str, Any]] = []
    detailed_rows: List[Dict[str, Any]] = []
    for sample in samples:
        retrieved = pipeline.retrieve(
            sample.query, top_k=retrieval_k, score_threshold=score_threshold
        )

        evaluation = evaluate_query(retrieved, sample.gold_items, k_values, args.fuzzy_threshold)
        per_query_results.append(evaluation)

        if args.save_details:
            match_sequence = evaluation.get("match_sequence", [])
            metrics_copy = {k: v for k, v in evaluation.items() if k != "match_sequence"}
            detailed_rows.append(
                {
                    "query": sample.query,
                    "metrics": metrics_copy,
                    "retrieved": [
                        {
                            "rank": idx + 1,
                            "id": doc.get("id"),
                            "score": doc.get("score"),
                            "metadata": doc.get("metadata"),
                            "content_preview": (doc.get("content") or "")[:200],
                            "is_relevant": idx < len(match_sequence) and match_sequence[idx] is not None,
                            "matched_gold_index": match_sequence[idx] if idx < len(match_sequence) else None,
                        }
                        for idx, doc in enumerate(retrieved)
                    ],
                }
            )

    summary = aggregate_metrics(per_query_results, k_values)

    precision_at_5 = summary["precision_at_k"].get(5)
    evidence_recall_at_3 = summary["evidence_recall_at_k"].get(3, 0.0)
    evidence_recall_at_10 = summary["evidence_recall_at_k"].get(10, 0.0)
    per_query_coverage_at_3 = summary["per_query_evidence_coverage"].get(3, 0.0)
    per_query_coverage_at_10 = summary["per_query_evidence_coverage"].get(10, 0.0)
    full_coverage_rate_at_3 = summary["full_coverage_rate"].get(3, 0.0)
    full_coverage_rate_at_10 = summary["full_coverage_rate"].get(10, 0.0)
    hit_rate_at_10 = summary.get("hit_rate_at_k", {}).get(10, 0.0)

    filtered_summary: Dict[str, Any] = {
        "precision_at_5": precision_at_5,
        "evidence_recall_at_3": evidence_recall_at_3,
        "evidence_recall_at_10": evidence_recall_at_10,
        "per_query_coverage_at_3": per_query_coverage_at_3,
        "per_query_coverage_at_10": per_query_coverage_at_10,
        "full_coverage_rate_at_3": full_coverage_rate_at_3,
        "full_coverage_rate_at_10": full_coverage_rate_at_10,
        "mean_average_precision": summary["mean_average_precision"],
        "mean_reciprocal_rank": summary["mean_reciprocal_rank"],
        "hit_rate_at_10": hit_rate_at_10,
    }

    print("\nRetrieval Evaluation Summary")
    print("=" * 32)
    if precision_at_5 is not None:
        print(f"Precision@5:    {precision_at_5:.4f}")
    print(f"EvidenceRecall@3:     {evidence_recall_at_3:.4f}")
    print(f"EvidenceRecall@10:    {evidence_recall_at_10:.4f}")
    print(f"PerQueryCoverage@3:   {per_query_coverage_at_3:.4f}")
    print(f"PerQueryCoverage@10:  {per_query_coverage_at_10:.4f}")
    print(f"FullCoverageRate@3:   {full_coverage_rate_at_3:.4f}")
    print(f"FullCoverageRate@10:  {full_coverage_rate_at_10:.4f}")
    print(f"MAP:            {filtered_summary['mean_average_precision']:.4f}")
    print(f"MRR:            {filtered_summary['mean_reciprocal_rank']:.4f}")
    print(f"HitRate@10:     {hit_rate_at_10:.4f}")

    if args.save_details:
        args.save_details.parent.mkdir(parents=True, exist_ok=True)
        with args.save_details.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "summary": filtered_summary,
                    "details": detailed_rows,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("Per-query details written to %s", args.save_details)


if __name__ == "__main__":
    main()
