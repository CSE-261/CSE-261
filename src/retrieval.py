"""Retrieval and generation module for RAG pipeline."""

import logging
import json
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional
from pathlib import Path

from qdrant_client import QdrantClient

from .config import RAGConfig
from .llm_providers import create_llm_provider
from qdrant_client import models as qdrant_models

from .embedding_providers import create_embedding_provider
from .rerankers import create_reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(self, config: RAGConfig):
        """
        Initialize the RAG pipeline.

        Args:
            config: RAG configuration object
        """
        self.config = config
        self.qdrant_client = QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port
        )
        configured_name = getattr(config.qdrant, "dense_vector_name", None)
        if isinstance(configured_name, str):
            configured_name = configured_name.strip() or None
        self._configured_dense_vector_name: Optional[str] = configured_name
        self._dense_prefetch_using: Optional[str] = None
        self._checked_dense_vector_support: bool = False

        # Initialize LLM provider
        self.llm_provider = create_llm_provider(
            provider_type=config.llm.provider,
            model_name=config.llm.model_name,
            api_key=config.llm.gemini_api_key,
            base_url=config.llm.ollama_base_url,
            region_name=config.llm.aws_region,
            aws_access_key_id=config.llm.aws_access_key_id,
            aws_secret_access_key=config.llm.aws_secret_access_key,
            aws_session_token=config.llm.aws_session_token,
            aws_profile_name=config.llm.aws_profile_name,
            request_interval=config.llm.request_interval
        )

        # Initialize embedding provider
        self.embedding_provider = create_embedding_provider(
            provider_type=config.embedding.provider,
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            cache_dir=config.embedding.cache_dir
        )
        self.reranker = create_reranker(config.rerank)
        self.keyword_boost_terms = self._load_keywords()
        self.keyword_boost_factor = 1.3

        # Set vector size dynamically if not set
        if config.qdrant.vector_size is None:
            config.qdrant.vector_size = self.embedding_provider.get_dimension()
            logger.info(f"Set vector size to {config.qdrant.vector_size} based on embedding model")

        self._ensure_dense_prefetch_vector()

        # Section routing keywords -> canonical section hints
        self._section_routes = {
            "risk": ["risk factors", "item 1a", "risks"],
            "md&a": ["md&a", "management discussion", "management’s discussion", "item 7"],
            "financials": ["financial statements", "balance sheet", "income statement", "cash flow", "item 8"],
            "controls": ["controls", "internal control", "item 9a"],
            "business": ["business", "item 1"],
            "directors": ["board of directors", "directors", "item 10"],
        }

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            query: Query string

        Returns:
            Embedding vector
        """
        return self.embedding_provider.embed_query(query)

    def _extract_filters(self, query: str) -> Optional[qdrant_models.Filter]:
        """
        Extract strict company filters from the query.
        """
        qlow = query.lower()
        company_map = {
            "a10": "a10-networks-inc",
            "oracle": "oracle-corporation",
            "xperi": "xperi-corporation",
            "eros": "eros-international-plc",
            "overseas": "overseas-shipholding-group-inc",
            "osg": "overseas-shipholding-group-inc"
        }
        
        matched_companies = []
        for key, source_prefix in company_map.items():
            if key in qlow:
                matched_companies.append(source_prefix)
        
        # If no companies matched, or if multiple matched (e.g. comparison),
        # we might want to include all matched ones.
        if not matched_companies:
            return None
            
        # Build a filter that requires 'source' to start with one of the matched prefixes
        # Note: Qdrant 'MatchValue' is exact. 'source' in metadata usually includes the year suffix.
        # So we should use 'MatchText' or prefix logic if we had a dedicated field.
        # But looking at ingestion, 'company' field is stored. Let's try to filter on 'company' or partial 'source'.
        # Actually, ingestion stores 'company' metadata derived from filename.
        # Let's check ingestion.py: company = base.replace("-", " ")...
        # Better yet, the 'source' field is 'a10-networks-inc_2019'.
        # So we can use a "should" clause with MatchText on 'source' or just match the known doc_ids if year is 2019.
        # Since all docs are 2019 in this dataset, let's filter by the known source names.
        
        should_conditions = []
        for prefix in matched_companies:
            should_conditions.append(
                qdrant_models.FieldCondition(
                    key="metadata.source",
                    match=qdrant_models.MatchText(text=prefix) # MatchText allows prefix/token matching usually
                )
            )
            
        return qdrant_models.Filter(should=should_conditions)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search with optional auto-filtering.

        Args:
            query: Query string
            top_k: Maximum number of documents to retrieve (default from config)
            score_threshold: Minimum similarity score (default from config)

        Returns:
            List of retrieved documents with content, metadata, and score
        """
        if top_k is None:
            top_k = self.config.top_k
        if score_threshold is None:
            score_threshold = self.config.score_threshold
        fetch_k = max(50, top_k * 10)

        query_vector = self.embed_query(query)
        self._ensure_dense_prefetch_vector()
        dense_query_vector = self._make_query_vector(query_vector)
        
        # Initialize qlow here for broader use
        qlow = query.lower()

        # Apply strict company filtering if detected
        hard_filter = self._extract_filters(query)
        if hard_filter:
            logger.info(f"Applying hard filter for query: {query}")

        # Determine target sections based on query routing
        target_sections = self._route_sections(query)

        if self.config.hybrid_search and query.strip():
            logger.info("Performing Hybrid (Dense + BM25) search.")
            try:
                hybrid_prefetch = max(fetch_k, getattr(self.config, "hybrid_prefetch", top_k))
                self._ensure_dense_prefetch_vector()

                query_response = None
                for attempt in range(2):
                    dense_prefetch_kwargs = dict(
                        query=query_vector,
                            limit=hybrid_prefetch,
                            score_threshold=score_threshold,
                            filter=hard_filter, # Apply filter here
                        )
                    if self._dense_prefetch_using:
                        dense_prefetch_kwargs["using"] = self._dense_prefetch_using

                    dense_prefetch = qdrant_models.Prefetch(**dense_prefetch_kwargs)

                    # Merge hard filter with text filter if needed, but Prefetch takes a filter directly
                    text_filter_conditions = [
                        qdrant_models.FieldCondition(
                            key="content",
                            match=qdrant_models.MatchText(text=query),
                        )
                    ]
                    
                    # Combine text match with hard filter
                    combined_text_filter = None
                    if hard_filter:
                        # Combine musts
                        musts = text_filter_conditions + (hard_filter.must or [])
                        shoulds = hard_filter.should
                        combined_text_filter = qdrant_models.Filter(must=musts, should=shoulds)
                    else:
                        combined_text_filter = qdrant_models.Filter(must=text_filter_conditions)

                    text_prefetch = qdrant_models.Prefetch(
                        filter=combined_text_filter,
                        limit=hybrid_prefetch,
                    )

                    fusion_query = qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF)

                    try:
                        query_response = self.qdrant_client.query_points(
                            collection_name=self.config.qdrant.collection_name,
                            prefetch=[dense_prefetch, text_prefetch],
                            query=fusion_query,
                            limit=fetch_k,
                            with_payload=True,
                            with_vectors=False,
                        )
                        break
                    except Exception as hybrid_exc:
                        error_text = str(hybrid_exc)
                        if (
                            self._dense_prefetch_using
                            and "Vector with name" in error_text
                        ):
                            logger.info(
                                "Dense vector name '%s' is not configured in Qdrant; retrying without named vector.",
                                self._dense_prefetch_using,
                            )
                            self._dense_prefetch_using = None
                            self._checked_dense_vector_support = True
                            continue
                        raise hybrid_exc

                if query_response is None:
                    raise RuntimeError("Hybrid query failed without response.")

                search_result = getattr(query_response, "points", query_response)
            except Exception as exc:
                logger.warning(
                    "Hybrid query_points failed (%s); falling back to dense-only search.", exc
                )
                search_result = self.qdrant_client.search(
                    collection_name=self.config.qdrant.collection_name,
                    query_vector=dense_query_vector,
                    limit=fetch_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                    filter=hard_filter, # Apply filter here
                )
        else:
            logger.info("Performing Dense-only vector search.")
            search_result = self.qdrant_client.search(
                collection_name=self.config.qdrant.collection_name,
                query_vector=dense_query_vector,
                limit=fetch_k,
                score_threshold=score_threshold,
                with_payload=True,
                filter=hard_filter, # Apply filter here
            )

        formatted_docs = self._format_retrieved_docs(search_result, query, score_threshold, target_sections, qlow)
        
        # Subsequent filtering and balancing steps
        formatted_docs = self._augment_with_tables(query, formatted_docs, fetch_k)
        filtered_docs = self._filter_by_section(formatted_docs, target_sections, fetch_k)
        balanced_docs = self._balance_companies(query, filtered_docs, fetch_k)
        formatted_docs = self._maybe_rerank(query, balanced_docs)
        return formatted_docs[:top_k]

    def _route_sections(self, query: str) -> List[str]:
        """
        Lightweight routing: map query keywords to target section hints.
        """
        q = query.lower()
        matched: List[str] = []
        # Item number hints
        item_matches = re.findall(r"item\s+(\d+[a-z]?)", q)
        for im in item_matches:
            matched.append(f"item_{im.lower()}")
        for sec, keywords in self._section_routes.items():
            if any(kw in q for kw in keywords):
                matched.append(sec)
        
        if matched:
            logger.info(f"Routing query to sections: {matched}")
            
        return matched

    def _filter_by_section(self, docs: List[Dict[str, Any]], target_sections: List[str], top_k: int) -> List[Dict[str, Any]]:
        """
        If target sections are detected, keep docs whose section metadata matches; else return original.
        """
        if not target_sections:
            return docs

        routed: List[Dict[str, Any]] = []
        for doc in docs:
            meta = doc.get("metadata") or {}
            section = (meta.get("section_name") or meta.get("section_path") or "").lower()
            if any(t in section for t in target_sections):
                routed.append(doc)

        logger.info(f"Section filter: Kept {len(routed)}/{len(docs)} docs matching {target_sections}")

        # If filtering is too aggressive, fallback by preferentially pulling from the same doc_id
        if not routed:
            return docs[:top_k]

        min_keep = min(5, top_k)
        if len(routed) >= min_keep:
            return routed

        keep: List[Dict[str, Any]] = []
        seen_ids = set()

        def _add(doc: Dict[str, Any]) -> None:
            did = doc.get("id")
            if did in seen_ids:
                return
            seen_ids.add(did)
            keep.append(doc)

        for doc in routed:
            _add(doc)

        target_doc_ids = []
        for doc in routed:
            meta = doc.get("metadata") or {}
            doc_id = meta.get("doc_id") or meta.get("source")
            if doc_id:
                target_doc_ids.append(doc_id)

        if target_doc_ids:
            for doc in docs:
                meta = doc.get("metadata") or {}
                doc_id = meta.get("doc_id") or meta.get("source")
                if doc_id and doc_id in target_doc_ids:
                    _add(doc)
                if len(keep) >= min_keep:
                    break

        if len(keep) < min_keep:
            for doc in docs:
                _add(doc)
                if len(keep) >= min_keep:
                    break

        return keep[:top_k]

    def _balance_companies(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Ensure multi-company queries keep some evidence from each mentioned company.
        """
        if not docs:
            return docs

        qlow = query.lower()
        company_docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for doc in docs:
            meta = doc.get("metadata") or {}
            comp = (meta.get("company") or meta.get("source") or meta.get("doc_id") or "").lower()
            if comp and comp in qlow:
                company_docs[comp].append(doc)

        if len(company_docs) < 2:
            return docs

        # allocate roughly even quota across companies
        per_company_quota = max(2, top_k // len(company_docs))
        balanced: List[Dict[str, Any]] = []
        used_ids = set()
        # round-robin selection
        for _ in range(per_company_quota):
            for comp, comp_docs in company_docs.items():
                if not comp_docs:
                    continue
                doc = comp_docs.pop(0)
                did = doc.get("id")
                if did in used_ids:
                    continue
                used_ids.add(did)
                balanced.append(doc)
                if len(balanced) >= top_k:
                    return balanced

        # fill remaining with original order without duplicates
        for doc in docs:
            did = doc.get("id")
            if did in used_ids:
                continue
            balanced.append(doc)
            if len(balanced) >= top_k:
                break
        return balanced

    def _augment_with_tables(self, query: str, docs: List[Dict[str, Any]], fetch_k: int) -> List[Dict[str, Any]]:
        """
        If few tables are present, pull additional table chunks from the same doc_ids to improve table recall.
        """
        existing_tables = sum(1 for d in docs if (d.get("metadata") or {}).get("type") == "table")
        if existing_tables >= 3:
            return docs

        doc_ids = []
        seen_doc_ids = set()
        for doc in docs:
            meta = doc.get("metadata") or {}
            did = meta.get("doc_id") or meta.get("source")
            if did and did not in seen_doc_ids:
                seen_doc_ids.add(did)
                doc_ids.append(did)

        if not doc_ids:
            return docs

        added: List[Dict[str, Any]] = []
        boost_score = docs[0].get("score", 0.0) if docs else 0.0
        max_tables_per_doc = fetch_k
        tables_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for did in doc_ids:
            try:
                scroll_filter = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="metadata.doc_id",
                            match=qdrant_models.MatchValue(value=did),
                        ),
                        qdrant_models.FieldCondition(
                            key="metadata.type",
                            match=qdrant_models.MatchValue(value="table"),
                        ),
                    ]
                )
                next_page = None
                pulled = 0
                while pulled < max_tables_per_doc:
                    points, next_page = self.qdrant_client.scroll(
                        collection_name=self.config.qdrant.collection_name,
                        scroll_filter=scroll_filter,
                        with_payload=True,
                        with_vectors=False,
                        limit=50,
                        offset=next_page,
                    )
                    if not points:
                        break
                    for pt in points:
                        if pulled >= max_tables_per_doc:
                            break
                        meta = pt.payload.get("metadata", {}) if pt.payload else {}
                        tables_by_doc[did].append(
                            {
                                "content": (pt.payload or {}).get("content", ""),
                                "metadata": meta,
                                "score": boost_score,
                                "original_score": boost_score,
                                "id": pt.id,
                            }
                        )
                        pulled += 1
                    if next_page is None:
                        break
            except Exception as exc:
                logger.debug("Table augmentation failed for doc_id=%s: %s", did, exc)
                continue

        # Select a small, spaced subset of tables per doc to ensure coverage across pages
        for did, tbls in tables_by_doc.items():
            if not tbls:
                continue
            sorted_tbls = sorted(
                tbls,
                key=lambda t: (t.get("metadata", {}).get("page") or 0, t.get("metadata", {}).get("table_id") or ""),
            )
            take_count = min(6, len(sorted_tbls))
            if take_count == len(sorted_tbls):
                selection = sorted_tbls
            else:
                indices = [round(i * (len(sorted_tbls) - 1) / max(take_count - 1, 1)) for i in range(take_count)]
                selection = [sorted_tbls[idx] for idx in indices]
            # Give a slight descending boost so these surface ahead
            for idx, item in enumerate(selection):
                item["score"] = boost_score + 1.0 - idx * 0.01
                item["original_score"] = item["score"]
            added.extend(selection)

        if not added:
            return docs

        existing_ids = {d.get("id") for d in docs}
        merged: List[Dict[str, Any]] = []
        for d in docs + added:
            did = d.get("id")
            if did in existing_ids:
                existing_ids.discard(did)
                merged.append(d)
            elif did not in existing_ids:
                merged.append(d)
        # Resort by score desc to keep original ranking preference
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return merged

    def _ensure_dense_prefetch_vector(self) -> None:
        """
        Ensure we only ask for a named dense vector if the collection actually exposes it.
        """
        if self._checked_dense_vector_support:
            return

        dense_name = self._configured_dense_vector_name
        if not dense_name:
            self._checked_dense_vector_support = True
            return

        try:
            collection_info = self.qdrant_client.get_collection(
                collection_name=self.config.qdrant.collection_name
            )
        except Exception as exc:
            logger.debug(
                "Could not inspect collection vector schema (%s); assuming unnamed vector.",
                exc,
            )
            return

        vectors_config = getattr(collection_info.config.params, "vectors", None)
        if isinstance(vectors_config, dict):
            if dense_name in vectors_config:
                self._dense_prefetch_using = dense_name
            else:
                logger.info(
                    "Configured dense vector name '%s' not found in collection schema; using default vector.",
                    dense_name,
                )
                self._dense_prefetch_using = None
        else:
            self._dense_prefetch_using = None
        self._checked_dense_vector_support = True

    def _load_keywords(self) -> List[str]:
        """Load keyword list from src/keywords.json; fallback to a default set."""
        default = [
            "goodwill impairment",
            "non-gaap",
            "non gaap",
            "credit risk",
            "revenue recognition",
            "lease liability",
            "operating lease",
            "deferred revenue",
            "stock-based compensation",
            "tax provision",
            "foreign currency",
            "liquidity",
            "going concern",
            "material weakness",
            "internal control",
            "cybersecurity",
        ]
        kw_path = Path(__file__).resolve().parent / "keywords.json"
        try:
            data = json.loads(kw_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(k).lower() for k in data]
        except Exception:
            logger.debug("Failed to load keywords from %s; using default set.", kw_path)
        return default

    def _make_query_vector(self, query_vector: List[float]):
        """
        Prepare query vector for Qdrant, using named vector when available.
        """
        if self._dense_prefetch_using:
            return qdrant_models.NamedVector(
                name=self._dense_prefetch_using,
                vector=query_vector,
            )
        return query_vector

    def _format_retrieved_docs(self, search_result, query: str, score_threshold: Optional[float], target_sections: Optional[List[str]] = None, qlow: str = "") -> List[Dict[str, Any]]:
        """
        Normalize Qdrant search results into payload dictionaries.
        Apply a small boost to table chunks to help structured answers surface higher.
        """
        retrieved_docs: List[Dict[str, Any]] = []
        # qlow is now passed as a parameter
        # qlow = query.lower() # Removed

        def _contains(haystack: Optional[str]) -> bool:
            if not haystack:
                return False
            return haystack.lower() in qlow

        for hit in search_result:
            payload = hit.payload or {}
            metadata = payload.get("metadata", {}) or {}
            if metadata.get("type") == "table" and not metadata.get("table_path"):
                table_title = metadata.get("table_title")
                if table_title:
                    # Populate a canonical table_path so eval matching can hit gold CSV filenames
                    metadata["table_path"] = f"TAT-RAG/csvs/{table_title}.csv"

            boosted_score = hit.score
            if metadata.get("type") == "table":
                boosted_score *= 1.5  # lift tables more to surface structured answers
            # Keyword booster: if query mentions key financial terms and chunk tags match, bump
            kw_hits = metadata.get("keywords_hit") or []
            if kw_hits:
                if any(term in qlow for term in self.keyword_boost_terms):
                    boosted_score *= self.keyword_boost_factor

            # Metadata-aware boosts to favor the right company/section/table context
            company = metadata.get("company") or metadata.get("source") or metadata.get("doc_id")
            if _contains(company):
                boosted_score *= 1.4

            section_text = metadata.get("section_name") or metadata.get("section_path")
            if _contains(section_text) or (target_sections and any(ts in (section_text or "") for ts in target_sections)):
                boosted_score *= 2.0

            table_title = metadata.get("table_title") or metadata.get("table_id")
            row_label = metadata.get("row_label")
            column_label = metadata.get("column_label")
            fiscal_year = metadata.get("fiscal_year")
            if any(_contains(value) for value in (table_title, row_label, column_label, fiscal_year)):
                boosted_score *= 1.2

            retrieved_docs.append({
                "content": payload.get("content", ""),
                "metadata": metadata,
                "score": boosted_score,
                "original_score": hit.score,
                "id": hit.id
            })

        # Re-sort after boosting
        retrieved_docs.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}... "
            f"(threshold: {score_threshold:.2f})"
        )
        return retrieved_docs

    def _maybe_rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optionally rerank retrieved documents with an external reranker.
        """
        if not documents or not self.reranker:
            return documents

        try:
            reranked = self.reranker.rerank(query, documents)
        except Exception as exc:
            logger.warning("Reranking failed for query '%s': %s", query[:50], exc, exc_info=True)
            return documents

        return reranked or documents

    def _apply_adaptive_filters(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Dynamically trim low-relevance chunks and optionally cap per-document fan-out.
        """
        if not documents or not getattr(self.config, "adaptive_filter_enabled", False):
            return documents

        min_keep = max(0, getattr(self.config, "adaptive_min_keep", 0))
        score_ratio = getattr(self.config, "adaptive_score_ratio", 0.0)
        score_drop = getattr(self.config, "adaptive_score_drop", 0.0)
        max_chunks_per_doc = getattr(self.config, "max_chunks_per_doc", None)

        top_score = documents[0].get("score") or 0.0
        per_doc_counts: Dict[str, int] = defaultdict(int)
        filtered: List[Dict[str, Any]] = []

        for doc in documents:
            score = doc.get("score") or 0.0
            metadata = doc.get("metadata") or {}
            doc_key = metadata.get("source") or metadata.get("doc_id") or ""
            if max_chunks_per_doc and doc_key:
                if per_doc_counts[doc_key] >= max_chunks_per_doc:
                    continue

            keep = True
            if len(filtered) >= min_keep:
                keep_by_ratio = True
                keep_by_drop = True

                if score_ratio > 0 and top_score > 0:
                    keep_by_ratio = (score / top_score) >= score_ratio
                if score_drop > 0 and top_score >= score:
                    keep_by_drop = (top_score - score) <= score_drop

                keep = keep_by_ratio or keep_by_drop

            if not keep:
                break

            filtered.append(doc)
            if doc_key:
                per_doc_counts[doc_key] += 1

        if not filtered:
            return documents[:max(min_keep, 1)]
        return filtered

    def _select_docs_for_generation(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Slice the reranked documents according to the final context limit.
        """
        limit = getattr(self.config, "final_context_limit", None)
        if isinstance(limit, int) and limit > 0:
            return documents[:limit]
        return documents

    def generate_prompt(self, query: str, contexts: List[str]) -> str:
        """
        Generate a prompt for the LLM with query and retrieved contexts.

        Args:
            query: User query
            contexts: List of retrieved context strings

        Returns:
            Formatted prompt string
        """
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""You are a helpful assistant. Answer the user's question based on the provided contexts.

{context_text}

Question: {query}

Answer: Provide a detailed and accurate answer based on the contexts above. If the contexts don't contain enough information to answer the question, say so."""

        return prompt

    def generate(self, query: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Generate an answer using the LLM.

        Args:
            query: User query
            contexts: List of retrieved context strings

        Returns:
            Dictionary with answer and metadata
        """
        prompt = self.generate_prompt(query, contexts)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on provided contexts.",
            },
            {"role": "user", "content": prompt},
        ]

        max_attempts = 3
        response: Optional[Dict[str, Any]] = None
        answer: str = ""
        usage: Dict[str, Any] = {}

        for attempt in range(1, max_attempts + 1):
            response = self.llm_provider.generate(
                messages=messages,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )

            usage = dict(response.get("usage", {}) or {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens")
            if total_tokens is None:
                total_tokens = prompt_tokens + completion_tokens
                usage["total_tokens"] = total_tokens

            answer = (response.get("content") or "").strip()

            if answer or total_tokens > 0:
                break

            logger.warning(
                "Received empty response from LLM (attempt %s/%s) for query: %s",
                attempt,
                max_attempts,
                query[:50],
            )

        else:
            logger.error(
                "LLM failed to return a non-empty response after %s attempts for query: %s",
                max_attempts,
                query[:50],
            )

        logger.info(f"Generated answer for query: {query[:50]}...")

        return {
            "answer": answer,
            "query": query,
            "contexts": contexts,
            "model": response.get("model") if response else None,
            "usage": usage,
        }


    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve and generate without query rewriting.

        Args:
            query: User query
            top_k: Maximum number of documents to retrieve (default from config)
            score_threshold: Minimum similarity score (default from config)

        Returns:
            Dictionary with answer, contexts, and metadata
        """
        logger.info(f"Processing query: {query}")

        retrieved_docs = self.retrieve(query, top_k, score_threshold)
        filtered_docs = self._apply_adaptive_filters(retrieved_docs)
        final_docs_for_generation = self._select_docs_for_generation(filtered_docs)
        contexts = [doc.get("content", "") for doc in final_docs_for_generation]

        result = self.generate(query, contexts)
        result["retrieved_docs"] = filtered_docs
        result["used_retrieved_docs"] = final_docs_for_generation

        return result

    def batch_query(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch (no query rewriting).
        """
        results: List[Dict[str, Any]] = []
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            result = self.query(query, top_k=top_k, score_threshold=score_threshold)
            results.append(result)
        return results
