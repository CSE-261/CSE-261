"""RAGAS evaluation module for RAG pipeline quality assessment."""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from datasets import Dataset
from ragas import evaluate

try:
    from ragas.executor import RunConfig  # type: ignore
except ImportError:
    RunConfig = None
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)

from langchain_ollama import ChatOllama
from langchain_aws import ChatBedrock
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from .retrieval import RAGPipeline
from .llm_providers import OllamaProvider, BedrockProvider, GeminiProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiChatAdapter(BaseChatModel):
    """Minimal LangChain chat model wrapper over GeminiProvider for RAGAS."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: GeminiProvider = Field(...)
    temperature: float = Field(...)
    max_tokens: int = Field(...)

    @staticmethod
    def _flatten_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return "" if content is None else str(content)

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        converted: List[Dict[str, str]] = []
        for msg in messages:
            role_type = getattr(msg, "type", "user")
            if role_type in ("ai", "assistant"):
                role = "assistant"
            elif role_type == "system":
                role = "system"
            else:
                role = "user"

            converted.append({
                "role": role,
                "content": self._flatten_content(msg.content).strip(),
            })
        return converted

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> ChatResult:
        converted_messages = self._convert_messages(messages)
        response = self.provider.generate(
            messages=converted_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = response.get("content", "") or ""
        ai_message = AIMessage(content=text)
        generation = ChatGeneration(
            message=ai_message,
            text=text,
            generation_info={"usage": response.get("usage")}
        )
        llm_output = {
            "usage": response.get("usage"),
            "model": response.get("model"),
        }
        return ChatResult(generations=[generation], llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "gemini-provider-adapter"


class RAGEvaluator:
    """Evaluate RAG pipeline using RAGAS metrics."""

    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize the evaluator.

        Args:
            rag_pipeline: RAG pipeline instance to evaluate
        """
        self.rag_pipeline = rag_pipeline

    def prepare_eval_dataset(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        contexts: Optional[List[List[str]]] = None,
    ) -> Dataset:
        """
        Prepare evaluation dataset in RAGAS format.

        Args:
            questions: List of questions
            ground_truths: List of ground truth answers (optional)
            answers: List of generated answers (optional, will generate if not provided)
            contexts: List of context lists (optional, will retrieve if not provided)

        Returns:
            Dataset object for RAGAS evaluation
        """
        if answers is None or contexts is None:
            logger.info("Generating answers and retrieving contexts...")
            results = self.rag_pipeline.batch_query(questions)
            if answers is None:
                answers = [r["answer"] for r in results]
            if contexts is None:
                contexts = [r["contexts"] for r in results]

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }

        if ground_truths is not None:
            data["ground_truth"] = ground_truths
            # Newer versions of ragas expect the alias `reference`
            data["reference"] = ground_truths

        dataset = Dataset.from_dict(data)
        logger.info(f"Prepared evaluation dataset with {len(questions)} samples")
        return dataset

    def evaluate(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        contexts: Optional[List[List[str]]] = None,
        metrics: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate RAG pipeline using RAGAS metrics.

        Args:
            questions: List of questions
            ground_truths: List of ground truth answers (required for some metrics)
            answers: List of generated answers (will generate if not provided)
            contexts: List of context lists (will retrieve if not provided)
            metrics: List of RAGAS metrics to use (default: all available metrics)

        Returns:
            Dictionary with evaluation results
        """
        # Prepare dataset
        dataset = self.prepare_eval_dataset(
            questions,
            ground_truths,
            answers,
            contexts,
        )

        # Select metrics based on available data
        if metrics is None:
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
            ]
            # Add metrics that require ground truth only if available
            if ground_truths is not None:
                metrics.extend([
                    context_recall,
                    # answer_correctness, # Disabled for smaller models that struggle with JSON output
                ])

        # Create a compatible LangChain chat model for ragas based on provider type
        llm_provider = self.rag_pipeline.llm_provider

        if isinstance(llm_provider, OllamaProvider):
            chat_model = ChatOllama(
                model=llm_provider.model_name,
                base_url=llm_provider.base_url,
            )
        elif isinstance(llm_provider, BedrockProvider):
            # Bedrock with Claude models (native ChatBedrock support)
            chat_model = ChatBedrock(
                model_id=llm_provider.model_name,
                region_name=llm_provider.region_name,
                client=llm_provider.client
            )
        elif isinstance(llm_provider, GeminiProvider):
            chat_model = GeminiChatAdapter(
                provider=llm_provider,
                temperature=self.rag_pipeline.config.llm.temperature,
                max_tokens=self.rag_pipeline.config.llm.max_tokens
            )
        else:
            raise ValueError(f"Unsupported LLM provider type: {type(llm_provider)}")

        # Run evaluation
        try:
            kwargs = {
                "metrics": metrics,
                "llm": chat_model,
                "embeddings": self.rag_pipeline.embedding_provider,
            }
            if RunConfig is not None:
                ragas_cfg = self.rag_pipeline.config.ragas
                kwargs["run_config"] = RunConfig(
                    timeout=ragas_cfg.timeout,
                    max_workers=ragas_cfg.max_workers,
                    max_retries=ragas_cfg.max_retries,
                    max_wait=ragas_cfg.max_wait,
                )

            result = evaluate(
                dataset,
                **kwargs
            )
            logger.info("Evaluation completed successfully")
            return result
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    def precompute_answers(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        output_path: Optional[str] = None,
        include_retrieved_docs: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate answers/contexts ahead of time so RAGAS only scores cached data.
        """
        results = self.rag_pipeline.batch_query(
            questions,
            top_k=top_k,
        )
        cached_results: List[Dict[str, Any]] = []

        for idx, result in enumerate(results):
            cache_entry: Dict[str, Any] = {
                "question": result.get("query", questions[idx]),
                "answer": result.get("answer"),
                "contexts": result.get("contexts", []),
                "model": result.get("model"),
                "usage": result.get("usage", {}),
            }
            if include_retrieved_docs:
                cache_entry["retrieved_docs"] = result.get("retrieved_docs", [])
            if ground_truths is not None:
                cache_entry["ground_truth"] = ground_truths[idx] if idx < len(ground_truths) else None
                cache_entry["reference"] = cache_entry["ground_truth"]
            cached_results.append(cache_entry)

        if output_path:
            import json

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cached_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached {len(cached_results)} answers to {output_path}")

        return cached_results

    def evaluate_from_cache(
        self,
        cached_results: List[Dict[str, Any]],
        metrics: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation using precomputed answers/contexts.
        """
        if not cached_results:
            raise ValueError("Cached results list is empty")

        questions = [item["question"] for item in cached_results]
        answers = [item.get("answer", "") for item in cached_results]
        contexts = [item.get("contexts", []) for item in cached_results]

        ground_truths = None
        if any("ground_truth" in item or "reference" in item for item in cached_results):
            ground_truths = []
            for item in cached_results:
                ground_truth = item.get("ground_truth", item.get("reference"))
                ground_truths.append(ground_truth if ground_truth is not None else "")
            if all(gt == "" for gt in ground_truths):
                ground_truths = None

        return self.evaluate(
            questions=questions,
            ground_truths=ground_truths,
            answers=answers,
            contexts=contexts,
            metrics=metrics
        )

    def evaluate_from_cache_file(
        self,
        cache_path: str,
        metrics: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Load cached answers/contexts from disk and evaluate.
        """
        import json

        with open(cache_path, "r", encoding="utf-8") as f:
            cached_results = json.load(f)

        if not isinstance(cached_results, list):
            raise ValueError("Cached results file must contain a list of entries")

        return self.evaluate_from_cache(cached_results, metrics=metrics)

    def evaluate_from_file(
        self,
        csv_path: str,
        question_col: str = "question",
        ground_truth_col: Optional[str] = "ground_truth",
        metrics: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG pipeline from a CSV file.

        Args:
            csv_path: Path to CSV file with questions and ground truths
            question_col: Name of question column
            ground_truth_col: Name of ground truth column (optional)
            metrics: List of RAGAS metrics to use

        Returns:
            Dictionary with evaluation results
        """
        df = pd.read_csv(csv_path)
        questions = df[question_col].tolist()

        ground_truths = None
        if ground_truth_col and ground_truth_col in df.columns:
            ground_truths = df[ground_truth_col].tolist()

        return self.evaluate(questions, ground_truths=ground_truths, metrics=metrics)

    def evaluate_from_json(
        self,
        json_path: str,
        question_key: str = "query",
        ground_truth_key: str = "answer",
        metrics: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG pipeline from a JSON list of evaluation samples.

        Args:
            json_path: Path to JSON file (list of dicts) containing evaluation data
            question_key: Field name that stores the question text
            ground_truth_key: Field name that stores the ground truth answer
            metrics: List of RAGAS metrics to use

        Returns:
            Dictionary with evaluation results
        """
        import json

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON evaluation file must contain a list of objects")

        questions = []
        ground_truths = []
        for item in data:
            if question_key not in item:
                raise KeyError(f"Missing '{question_key}' field in evaluation sample: {item}")
            questions.append(item[question_key])
            if ground_truth_key in item:
                ground_truths.append(item[ground_truth_key])
            else:
                ground_truths.append(None)

        # If every ground truth is None, treat as missing
        if all(gt is None for gt in ground_truths):
            ground_truths = None
        else:
            # Replace missing ones with empty string to keep length alignment
            ground_truths = [gt if gt is not None else "" for gt in ground_truths]

        return self.evaluate(
            questions,
            ground_truths=ground_truths,
            metrics=metrics,
        )

    def compare_configurations(
        self,
        questions: List[str],
        ground_truths: List[str],
        top_k_values: List[int] = [3, 5, 10],
    ) -> pd.DataFrame:
        """
        Compare RAG performance with different configurations.

        Args:
            questions: List of questions
            ground_truths: List of ground truth answers
            top_k_values: List of top_k values to test

        Returns:
            DataFrame with comparison results
        """
        results = []

        for top_k in top_k_values:
            logger.info(f"Evaluating with top_k={top_k}")

            # Generate answers with specific top_k
            rag_results = []
            for idx, question in enumerate(questions):
                result = self.rag_pipeline.query(
                    question,
                    top_k=top_k,
                )
                rag_results.append(result)

            answers = [r["answer"] for r in rag_results]
            contexts = [r["contexts"] for r in rag_results]

            # Evaluate
            eval_result = self.evaluate(
                questions=questions,
                ground_truths=ground_truths,
                answers=answers,
                contexts=contexts
            )

            # Store results
            result_dict = {"top_k": top_k}
            result_dict.update(eval_result)
            results.append(result_dict)

        df = pd.DataFrame(results)
        logger.info("Configuration comparison completed")
        return df

    def generate_report(
        self,
        eval_result: Any,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a human-readable evaluation report.

        Args:
            eval_result: Evaluation result (can be RAGAS EvaluationResult or dict)
            output_path: Path to save report (optional)

        Returns:
            Report string
        """
        report = "=" * 50 + "\n"
        report += "RAG Pipeline Evaluation Report\n"
        report += "=" * 50 + "\n\n"

        # Convert RAGAS EvaluationResult to dict if needed
        if hasattr(eval_result, 'to_pandas'):
            # RAGAS returns an EvaluationResult object
            # Convert to pandas and get mean scores
            df = eval_result.to_pandas()
            result_dict = {}
            # Get mean of each metric column (skip non-numeric columns)
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    result_dict[col] = df[col].mean()
        elif isinstance(eval_result, dict):
            result_dict = eval_result
        else:
            # Try to convert to dict
            result_dict = dict(eval_result)

        for metric, value in result_dict.items():
            if isinstance(value, (int, float)):
                report += f"{metric}: {value:.4f}\n"

        report += "\n" + "=" * 50 + "\n"

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report
