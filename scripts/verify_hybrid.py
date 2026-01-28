"""Verify Qdrant collection is configured for Dense + BM25 hybrid search."""

import os
from qdrant_client import QdrantClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_hybrid_search():
    """
    Verifies that the Qdrant collection is properly configured for Dense + BM25 hybrid search.

    Checks:
    1. Dense vector configuration exists
    2. BM25 text index is configured on 'content' field
    3. Sample points contain both dense vectors and text content
    """
    # --- Configuration ---
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "documents")
    dense_vector_name = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
    # --- End Configuration ---

    print("\n" + "="*70)
    print("  Dense + BM25 Hybrid Search Verification")
    print("="*70)

    try:
        # 1. Initialize the Qdrant client
        client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=10)
        logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")

        # Check if the server is reachable
        client.get_collections()
        logger.info("Successfully connected to Qdrant.")

        # 2. Get the collection information
        logger.info(f"Fetching details for collection: '{collection_name}'")
        collection_info = client.get_collection(collection_name=collection_name)

        # 3. Verify dense vector configuration
        print("\n--- Dense Vector Configuration ---")
        vector_config = collection_info.config.params.vectors

        dense_config = None
        if isinstance(vector_config, dict):
            dense_config = vector_config.get(dense_vector_name)
        else:
            # Single vector configuration (not named)
            dense_config = vector_config

        if dense_config is not None:
            size = getattr(dense_config, 'size', 'unknown')
            distance = getattr(dense_config, 'distance', 'unknown')
            print(f"✅ Dense vector '{dense_vector_name}' configured")
            print(f"   - Dimension: {size}")
            print(f"   - Distance metric: {distance}")
        else:
            print(f"❌ Dense vector configuration '{dense_vector_name}' not found!")
            return

        # 4. Verify BM25 text index on 'content' field
        print("\n--- BM25 Text Index Configuration ---")
        payload_schema = collection_info.payload_schema
        content_field_info = payload_schema.get("content")

        if content_field_info:
            params = getattr(content_field_info, "params", None)

            # params itself is the TextIndexParams object
            if params:
                tokenizer = getattr(params, "tokenizer", "unknown")
                min_token_len = getattr(params, "min_token_len", "unknown")
                max_token_len = getattr(params, "max_token_len", "unknown")
                lowercase = getattr(params, "lowercase", "unknown")

                print(f"✅ BM25 text index configured on 'content' field")
                print(f"   - Tokenizer: {tokenizer}")
                print(f"   - Token length: {min_token_len}-{max_token_len}")
                print(f"   - Lowercase: {lowercase}")
            else:
                print("❌ No BM25 text index found on 'content' field!")
                print("   Hybrid search will not work properly.")
                return
        else:
            print("❌ No payload schema found for 'content' field!")
            return

        # 5. Verify sample points contain required data
        print("\n--- Sample Point Verification ---")
        sample_points, _ = client.scroll(
            collection_name=collection_name,
            limit=3,
            with_vectors=True,
            with_payload=True,
        )

        if not sample_points:
            print("⚠️  No points found in the collection.")
            print("   Please ingest data first: docker-compose exec rag-app python main.py ingest /app/data/chunks_all.jsonl")
            return

        print(f"Checking {len(sample_points)} sample point(s)...\n")

        all_valid = True
        for idx, sample in enumerate(sample_points, 1):
            print(f"Point {idx} (ID: {sample.id}):")

            # Check dense vector
            vectors = sample.vector or {}
            has_dense_vector = False

            if isinstance(vectors, dict):
                has_dense_vector = dense_vector_name in vectors
                if has_dense_vector:
                    dense_dim = len(vectors[dense_vector_name])
                    print(f"  ✅ Dense vector present (dim={dense_dim})")
                else:
                    print(f"  ❌ Dense vector missing!")
                    all_valid = False
            else:
                # Single vector (not named)
                has_dense_vector = bool(vectors)
                if has_dense_vector:
                    dense_dim = len(vectors)
                    print(f"  ✅ Dense vector present (dim={dense_dim})")
                else:
                    print(f"  ❌ Dense vector missing!")
                    all_valid = False

            # Check content field for BM25
            payload = sample.payload or {}
            content = payload.get("content", "")

            if content:
                content_preview = content[:80] + "..." if len(content) > 80 else content
                print(f"  ✅ Content field present ({len(content)} chars)")
                print(f"     Preview: {content_preview}")
            else:
                print(f"  ❌ Content field missing or empty!")
                all_valid = False

            print()

        # 6. Summary
        print("="*70)
        if all_valid:
            print("✅ SUCCESS: Collection is properly configured for Dense + BM25 hybrid search!")
            print("\nHybrid search architecture:")
            print("  • Dense embeddings: Vector similarity search")
            print("  • BM25 text index: Keyword-based search")
            print("  • Qdrant automatically fuses both results when using hybrid search")
            print("\nTo enable hybrid search, set in your .env:")
            print("  HYBRID_SEARCH_ENABLED=true")
        else:
            print("❌ ISSUES FOUND: Some sample points are missing required data.")
            print("   Please re-ingest your data to fix this.")
        print("="*70 + "\n")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"\n❌ [Error] Verification failed!")
        print(f"   Error details: {e}")
        print(f"\nPlease check:")
        print(f"   1. Is Qdrant running at {qdrant_host}:{qdrant_port}?")
        print(f"      Try: docker-compose ps")
        print(f"   2. Does the collection '{collection_name}' exist?")
        print(f"   3. Are environment variables set correctly?")
        print(f"      QDRANT_HOST={qdrant_host}")
        print(f"      QDRANT_PORT={qdrant_port}")
        print(f"      QDRANT_COLLECTION={collection_name}\n")

if __name__ == "__main__":
    verify_hybrid_search()
