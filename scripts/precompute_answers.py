"""Utility to precompute answers/contexts for later RAGAS evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config as default_config
from src.evaluation import RAGEvaluator
from src.retrieval import RAGPipeline


def _load_from_csv(
    csv_path: Path,
    question_col: str,
    ground_truth_col: Optional[str],
    limit: Optional[int],
) -> Tuple[List[str], Optional[List[str]]]:
    df = pd.read_csv(csv_path)
    if question_col not in df.columns:
        raise KeyError(f"Column '{question_col}' not found in {csv_path}")

    if limit is not None:
        df = df.head(limit)

    questions = df[question_col].astype(str).tolist()

    ground_truths: Optional[List[str]] = None
    if ground_truth_col and ground_truth_col in df.columns:
        ground_truths = df[ground_truth_col].astype(str).tolist()

    return questions, ground_truths


def _load_from_json(
    json_path: Path,
    question_key: str,
    ground_truth_key: Optional[str],
    limit: Optional[int],
) -> Tuple[List[str], Optional[List[str]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of evaluation samples")

    if limit is not None:
        data = data[:limit]

    questions: List[str] = []
    ground_truths: Optional[List[str]] = [] if ground_truth_key else None

    for item in data:
        if question_key not in item:
            raise KeyError(f"Missing '{question_key}' field in item: {item}")
        questions.append(str(item[question_key]))

        if ground_truth_key:
            ground_truth = item.get(ground_truth_key)
            ground_truths.append(str(ground_truth) if ground_truth is not None else "")

    if ground_truths is not None and all(gt == "" for gt in ground_truths):
        ground_truths = None

    return questions, ground_truths


def load_questions(
    path: Path,
    question_field: str,
    ground_truth_field: Optional[str],
    limit: Optional[int],
) -> Tuple[List[str], Optional[List[str]]]:
    if path.suffix.lower() in {".csv"}:
        return _load_from_csv(path, question_field, ground_truth_field, limit)

    if path.suffix.lower() in {".json", ".jsonl"}:
        return _load_from_json(path, question_field, ground_truth_field, limit)

    raise ValueError(f"Unsupported evaluation file format: {path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute answers/contexts for RAG evaluation.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to CSV or JSON evaluation file containing questions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "cached_answers.json",
        help="Where to write the cached results JSON.",
    )
    parser.add_argument(
        "--question-field",
        default="question",
        help="Column/key name that stores the question text.",
    )
    parser.add_argument(
        "--ground-truth-field",
        default="ground_truth",
        help="Column/key name for ground truth answers (optional).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top_k used for retrieval during caching.",
    )
    parser.add_argument(
        "--include-docs",
        action="store_true",
        help="Persist retrieved_docs alongside answers and contexts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions to cache (useful for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ground_truth_field = args.ground_truth_field or None

    questions, ground_truths = load_questions(
        input_path,
        question_field=args.question_field,
        ground_truth_field=ground_truth_field,
        limit=args.limit,
    )

    rag_pipeline = RAGPipeline(default_config)
    evaluator = RAGEvaluator(rag_pipeline)

    evaluator.precompute_answers(
        questions=questions,
        ground_truths=ground_truths,
        top_k=args.top_k,
        output_path=str(output_path),
        include_retrieved_docs=args.include_docs,
    )


if __name__ == "__main__":
    main()
