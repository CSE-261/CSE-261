"""Configuration settings for the RAG pipeline - optimized for RTX 4090."""

import os
from dataclasses import dataclass
from typing import Optional


def _get_bool_env(var_name: str, default: bool = False) -> bool:
    """Read boolean flag from environment."""
    value = os.getenv(var_name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _get_optional_int_env(var_name: str) -> Optional[int]:
    """Read optional integer from environment."""
    value = os.getenv(var_name)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _get_optional_float_env(var_name: str) -> Optional[float]:
    """Read optional float from environment."""
    value = os.getenv(var_name)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration."""
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name: str = os.getenv("QDRANT_COLLECTION", "documents")
    vector_size: Optional[int] = None  # Will be set dynamically based on embedding model
    distance_metric: str = "Cosine"
    timeout: int = int(os.getenv("QDRANT_TIMEOUT", "60"))
    dense_vector_name: str = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")


@dataclass
class LLMConfig:
    """Large Language Model configuration."""
    # Provider: 'ollama', 'bedrock', or 'gemini'
    provider: str = os.getenv("LLM_PROVIDER", "bedrock")
    model_name: str = os.getenv("LLM_MODEL", "anthropic.claude-haiku-4-5-20251001-v1:0")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    request_interval: float = float(os.getenv("LLM_REQUEST_INTERVAL", "0"))

    # Ollama specific
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # AWS Bedrock specific
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token: Optional[str] = os.getenv("AWS_SESSION_TOKEN")
    aws_profile_name: Optional[str] = os.getenv("AWS_PROFILE_NAME")

    # Gemini specific
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")


@dataclass
class EmbeddingConfig:
    """Embedding model configuration - optimized for RTX 4090."""
    # Provider: 'local' only
    provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    # ========== 4090 优化配置 ==========
    # Dense embedding batch size - 4090有24GB VRAM，可以激进
    dense_batch_size: int = int(os.getenv("DENSE_BATCH_SIZE", "128"))

    # FP16加速 - 4090支持，可以2倍加速
    use_fp16: bool = _get_bool_env("USE_FP16", True)

    # 按长度排序以减少padding - 10-15%提升
    sort_by_length: bool = _get_bool_env("SORT_BY_LENGTH", True)

    # Local model specific
    device: Optional[str] = os.getenv("EMBEDDING_DEVICE", None)  # 'cuda', 'cpu', or None for auto
    cache_dir: Optional[str] = os.getenv("EMBEDDING_CACHE_DIR", None)  # Local model cache directory

@dataclass
class RAGASEvaluatorConfig:
    """RAGAS evaluation settings."""
    timeout: int = int(os.getenv("RAGAS_TIMEOUT", "180"))
    max_workers: int = int(os.getenv("RAGAS_MAX_WORKERS", "2"))
    max_retries: int = int(os.getenv("RAGAS_MAX_RETRIES", "3"))
    max_wait: int = int(os.getenv("RAGAS_MAX_WAIT", "30"))


@dataclass
class RerankConfig:
    """Optional document reranker settings."""
    enabled: bool = _get_bool_env("RERANK_ENABLED", False)
    provider: str = os.getenv("RERANK_PROVIDER", "bedrock")
    model_name: Optional[str] = os.getenv("RERANK_MODEL", "amazon.rerank-v1:0")
    top_n: Optional[int] = _get_optional_int_env("RERANK_TOP_N")
    request_interval: Optional[float] = _get_optional_float_env("RERANK_REQUEST_INTERVAL")
    aws_region: Optional[str] = os.getenv("RERANK_AWS_REGION") or os.getenv("AWS_REGION")
    aws_access_key_id: Optional[str] = os.getenv("RERANK_AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = os.getenv("RERANK_AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token: Optional[str] = os.getenv("RERANK_AWS_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN")
    aws_profile_name: Optional[str] = os.getenv("RERANK_AWS_PROFILE") or os.getenv("AWS_PROFILE_NAME")


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""
    top_k: int = int(os.getenv("RAG_TOP_K", "20"))
    score_threshold: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.1"))  # Minimum similarity score
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    hybrid_search: bool = _get_bool_env("HYBRID_SEARCH_ENABLED", False)
    hybrid_prefetch: int = int(os.getenv("HYBRID_PREFETCH_LIMIT", "50"))
    final_context_limit: int = int(os.getenv("RAG_FINAL_CONTEXT_LIMIT", "5"))
    max_chunks_per_doc: Optional[int] = None

    qdrant: QdrantConfig = None
    llm: LLMConfig = None
    embedding: EmbeddingConfig = None
    ragas: RAGASEvaluatorConfig = None
    rerank: RerankConfig = None

    def __post_init__(self):
        if self.qdrant is None:
            self.qdrant = QdrantConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.ragas is None:
            self.ragas = RAGASEvaluatorConfig()
        if self.rerank is None:
            self.rerank = RerankConfig()

        if self.max_chunks_per_doc is not None and self.max_chunks_per_doc <= 0:
            self.max_chunks_per_doc = None


# Default configuration instance
config = RAGConfig()
