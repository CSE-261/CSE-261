"""Reranking utilities for retrieved documents."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence
import time

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None  # type: ignore
    BotoCoreError = ClientError = Exception  # type: ignore

from .config import RerankConfig

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract interface for reranking strategies."""

    @abstractmethod
    def rerank(self, query: str, documents: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return documents ordered by relevance for the query."""


class BedrockReranker(BaseReranker):
    """AWS Bedrock wrapper supporting Cohere and Amazon rerank models."""

    def __init__(
        self,
        model_id: str,
        region_name: Optional[str],
        top_n: Optional[int],
        request_interval: Optional[float] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
        max_attempts: int = 3,
    ) -> None:
        if boto3 is None:
            raise ImportError("boto3 is required for Bedrock reranker support.")

        session_kwargs: Dict[str, Any] = {}
        if region_name:
            session_kwargs["region_name"] = region_name
        if aws_profile_name:
            session_kwargs["profile_name"] = aws_profile_name
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
            if aws_session_token:
                session_kwargs["aws_session_token"] = aws_session_token

        self.session = boto3.Session(**session_kwargs)  # type: ignore[arg-type]
        self.client = self.session.client("bedrock-runtime")
        self.model_id = model_id
        self.max_docs = top_n
        self.request_interval = request_interval or 0.0
        self.max_attempts = max(1, max_attempts)

    def rerank(self, query: str, documents: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return list(documents)

        limit = min(len(documents), self.max_docs or len(documents))
        selected_docs = list(documents[:limit])
        doc_texts = [doc.get("content", "") or "" for doc in selected_docs]

        request_payload: Dict[str, Any]
        model_id_lower = self.model_id.lower()

        if model_id_lower.startswith("cohere.rerank"):
            request_payload = {
                "query": query,
                "documents": doc_texts,
                "top_n": limit,
                "api_version": "1",
            }
        elif model_id_lower.startswith("amazon.rerank"):
            request_payload = {
                "query": query,
                "documents": doc_texts,
            }
        else:
            logger.warning("Unknown Bedrock rerank model prefix for '%s'", self.model_id)
            return list(documents)

        data: Dict[str, Any] | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_payload),
                    contentType="application/json",
                    accept="application/json",
                )
                data = json.loads(response["body"].read())
                break
            except (BotoCoreError, ClientError) as exc:
                error_code = getattr(exc, "response", {}).get("Error", {}).get("Code")  # type: ignore[attr-defined]
                if error_code not in {"ThrottlingException", "TooManyRequestsException"} or attempt == self.max_attempts:
                    logger.warning("Bedrock rerank request failed: %s", exc, exc_info=True)
                    return list(documents)
                sleep_time = max(self.request_interval, 1.0) * attempt
                logger.info("Rerank throttled (attempt %s/%s). Sleeping %.1fs before retry.", attempt, self.max_attempts, sleep_time)
                time.sleep(sleep_time)
            except (RuntimeError, ValueError, KeyError) as exc:
                logger.warning("Bedrock rerank response parsing failed: %s", exc, exc_info=True)
                return list(documents)

        if data is None:
            return list(documents)

        results = data.get("results") or []
        if not isinstance(results, list):
            logger.warning("Unexpected rerank response shape: %s", data)
            return list(documents)

        score_map: Dict[int, float] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if index is None:
                continue
            try:
                index_int = int(index)
            except (TypeError, ValueError):
                continue

            score_value = (
                item.get("relevanceScore")
                or item.get("relevance_score")
                or item.get("score")
            )
            try:
                score_float = float(score_value)
            except (TypeError, ValueError):
                continue
            score_map[index_int] = score_float

        if not score_map:
            logger.debug("Bedrock rerank returned empty scores; keeping original order.")
            return list(documents)

        ranked_indices = sorted(
            range(len(selected_docs)),
            key=lambda idx: score_map.get(idx, float("-inf")),
            reverse=True,
        )
        reordered = [selected_docs[idx] for idx in ranked_indices]

        if len(documents) > limit:
            reordered.extend(list(documents[limit:]))

        return reordered


def create_reranker(config: Optional[RerankConfig]) -> Optional[BaseReranker]:
    """Instantiate reranker based on configuration."""
    if not config or not config.enabled:
        return None

    provider = (config.provider or "").lower()
    if provider != "bedrock":
        logger.warning("Unsupported rerank provider '%s'; reranking disabled.", config.provider)
        return None

    if not config.model_name:
        logger.warning("Rerank is enabled but no model is configured; reranking disabled.")
        return None

    try:
        reranker = BedrockReranker(
            model_id=config.model_name,
            region_name=config.aws_region,
            top_n=config.top_n,
            request_interval=config.request_interval,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            aws_session_token=config.aws_session_token,
            aws_profile_name=config.aws_profile_name,
        )
        logger.info("Initialized Bedrock reranker with model '%s'.", config.model_name)
        return reranker
    except Exception as exc:  # pragma: no cover - configuration/credential issues
        logger.warning("Failed to initialize Bedrock reranker: %s", exc, exc_info=True)
        return None
