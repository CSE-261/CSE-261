"""Smoke test for Cohere ReRank via AWS Bedrock."""

from __future__ import annotations

import json
from typing import List, Dict, Any

import boto3

# ---------------------------------------------------------------------------
# Configuration: fill these with your credentials or rely on your AWS config.
# ---------------------------------------------------------------------------
AWS_REGION = "us-west-2"
AWS_PROFILE = ""  # optional named profile
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_SESSION_TOKEN = ""  # optional temporary credential

MODEL_ID = "amazon.rerank-v1:0"
QUERY = "what are the biggest benefits of retrieval augmented generation?"
DOCUMENTS: List[str] = [
    "RAG combines retrieval with generation so models ground answers in source data.",
    "Reinforcement learning from human feedback is used to align model outputs.",
    "Vector databases store dense embeddings for similarity search.",
    "Bedrock provides managed access to foundation models with enterprise controls.",
]
TOP_N = 3


def _create_session() -> boto3.Session:
    session_kwargs: Dict[str, Any] = {"region_name": AWS_REGION}
    if AWS_PROFILE:
        session_kwargs["profile_name"] = AWS_PROFILE
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        session_kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        session_kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
        if AWS_SESSION_TOKEN:
            session_kwargs["aws_session_token"] = AWS_SESSION_TOKEN
    return boto3.Session(**session_kwargs)


def _invoke_rerank(client, query: str, docs: List[str], top_n: int) -> Dict[str, Any]:
    limit = min(top_n, len(docs))

    if MODEL_ID.startswith("cohere.rerank"):
        payload: Dict[str, Any] = {
            "query": query,
            "documents": docs,
            "top_n": limit,
            "api_version": "1",
        }
    elif MODEL_ID.startswith("amazon.rerank"):
        payload = {
            "query": query,
            "documents": docs,
        }
    else:
        raise ValueError(f"Unsupported rerank model id: {MODEL_ID}")

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())


def main() -> int:
    if not DOCUMENTS:
        raise SystemExit("Add at least one document to rerank.")

    session = _create_session()
    client = session.client("bedrock-runtime")
    result = _invoke_rerank(client, QUERY, DOCUMENTS, TOP_N)

    results = result.get("results") or []
    if not results:
        print("No rerank results:", result)
        return 1

    print(f"Query: {QUERY}\n")
    for rank, item in enumerate(results, start=1):
        idx = int(item.get("index", -1))
        score = (
            item.get("relevanceScore")
            or item.get("relevance_score")
            or item.get("score")
        )
        text = DOCUMENTS[idx] if 0 <= idx < len(DOCUMENTS) else ""
        preview = (text or "")[:120].replace("\n", " ")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        print(f"- rank={rank} score={score_str} doc_index={idx} text={preview}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
