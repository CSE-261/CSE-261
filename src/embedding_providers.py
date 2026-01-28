"""Embedding provider abstraction layer for dense local models."""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional

# Suppress FutureWarnings from sentence-transformers
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for dense embedding providers."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query string."""

    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimensionality."""


class LocalEmbeddingProvider(EmbeddingProvider):
    """Dense embedding provider backed by sentence-transformers models."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

        if cache_dir:
            logger.info("Using local model cache directory: %s", cache_dir)
        else:
            logger.info("Using default HuggingFace cache directory")

        logger.info("Loading dense embedding model: %s", model_name)
        self.model = SentenceTransformer(
            model_name, device=device, cache_folder=cache_dir
        )
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            "Initialized dense embedding provider (model=%s, dim=%s, device=%s)",
            model_name,
            self.dimension,
            self.model.device,
        )

    def embed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        # Extract encode-specific kwargs, filtering out any that aren't valid
        encode_kwargs = {
            'show_progress_bar': kwargs.get('show_progress_bar', False),
            'convert_to_numpy': True,
        }

        # Add optional parameters if provided
        if 'batch_size' in kwargs:
            encode_kwargs['batch_size'] = kwargs['batch_size']
        if 'convert_to_tensor' in kwargs:
            encode_kwargs['convert_to_tensor'] = kwargs['convert_to_tensor']
            # If converting to tensor, don't convert to numpy
            if kwargs['convert_to_tensor']:
                encode_kwargs.pop('convert_to_numpy', None)
        if 'normalize_embeddings' in kwargs:
            encode_kwargs['normalize_embeddings'] = kwargs['normalize_embeddings']

        embeddings = self.model.encode(texts, **encode_kwargs)

        # Convert to list format expected by Qdrant
        if hasattr(embeddings, 'cpu'):
            # PyTorch tensor
            embeddings = embeddings.cpu().numpy()
        if hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        return embeddings

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Compatibility alias."""
        return self.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode(
            query, show_progress_bar=False, convert_to_numpy=True
        )
        return embedding.tolist()

    def get_dimension(self) -> int:
        return self.dimension


def create_embedding_provider(
    provider_type: str,
    model_name: str,
    api_key: Optional[str] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> EmbeddingProvider:
    """Factory for dense embedding providers."""
    del api_key  # not used for local models
    provider_type = provider_type.lower()

    if provider_type == "local":
        return LocalEmbeddingProvider(
            model_name=model_name, device=device, cache_dir=cache_dir
        )

    raise ValueError(
        f"Unknown embedding provider type: {provider_type}. Supported values: 'local'."
    )
