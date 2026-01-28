"""LLM provider abstraction layer supporting Ollama and AWS Bedrock."""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import copy
import json
import requests

try:
    import boto3
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with response and metadata
        """
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider (local)."""

    def __init__(self, model_name: str = "llama3:8b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama provider.

        Args:
            model_name: Ollama model name (e.g., llama3:8b, qwen2.5:7b)
            base_url: Ollama server URL
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/chat"
        logger.info(f"Initialized Ollama provider with model: {model_name} at {base_url}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate response using Ollama."""
        try:
            # Prepare request payload for Ollama API
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "thinking": {
                        "enabled": False
                    }
                }
            }

            # Make request to Ollama (streaming)
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=300,  # 5 minutes timeout
                stream=True
            )
            response.raise_for_status()

            content_parts = []
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            last_chunk: Dict[str, Any] = {}

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Failed to decode Ollama chunk: %s", line)
                    continue

                last_chunk = chunk

                if "error" in chunk:
                    raise RuntimeError(chunk["error"])

                message = chunk.get("message", {})
                text = message.get("content", "")
                if text:
                    content_parts.append(text)

                if "prompt_eval_count" in chunk:
                    usage["prompt_tokens"] = chunk.get("prompt_eval_count", usage["prompt_tokens"])
                if "eval_count" in chunk:
                    usage["completion_tokens"] = chunk.get("eval_count", usage["completion_tokens"])
                if chunk.get("done"):
                    break

            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

            answer = "".join(content_parts).strip()
            if not answer and last_chunk:
                answer = (last_chunk.get("response") or "").strip()

            return {
                "content": answer,
                "model": self.model_name,
                "usage": usage
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama generation failed: {str(e)}")
            raise RuntimeError(
                f"Failed to generate with Ollama. Make sure Ollama is running and "
                f"the model '{self.model_name}' is installed. Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Ollama generation failed: {str(e)}")
            raise RuntimeError(
                f"Failed to generate with Ollama. Error: {str(e)}"
            )


class BedrockProvider(LLMProvider):
    """AWS Bedrock LLM provider."""

    def __init__(
        self,
        model_name: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        profile_name: Optional[str] = None
    ):
        """
        Initialize AWS Bedrock provider.

        Args:
            model_name: Bedrock model ID (e.g., anthropic.claude-3-sonnet-20240229-v1:0)
            region_name: AWS region name
            aws_access_key_id: AWS access key (optional, can use IAM role or profile)
            aws_secret_access_key: AWS secret key (optional)
            aws_session_token: AWS session token (optional, for temporary credentials)
            profile_name: AWS profile name (optional)
        """
        if not BEDROCK_AVAILABLE:
            raise ImportError(
                "boto3 is required for AWS Bedrock. Install it with: pip install boto3"
            )

        # Build session kwargs
        session_kwargs = {}
        if profile_name:
            session_kwargs['profile_name'] = profile_name
        if aws_access_key_id:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key
        if aws_session_token:
            session_kwargs['aws_session_token'] = aws_session_token

        # Create session and client
        if session_kwargs:
            session = boto3.Session(**session_kwargs)
            self.client = session.client('bedrock-runtime', region_name=region_name)
        else:
            # Use default credentials (IAM role, environment variables, or default profile)
            self.client = boto3.client('bedrock-runtime', region_name=region_name)

        self.model_name = model_name
        self.region_name = region_name
        logger.info(f"Initialized AWS Bedrock provider with model: {model_name} in region: {region_name}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate response using AWS Bedrock with Claude models."""
        try:
            # Convert messages to Bedrock Claude format
            if "anthropic.claude" in self.model_name:
                # Format for Claude models
                system_message = None
                conversation = []

                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")

                    if role == "system":
                        system_message = content
                    else:
                        # Map 'assistant' and 'user' roles
                        conversation.append({
                            "role": role,
                            "content": content
                        })

                # Build request body for Claude
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": conversation,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

                if system_message:
                    body["system"] = system_message

            else:
                # Generic format for other models (adjust as needed)
                body = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

            # Call Bedrock API
            response = self.client.invoke_model(
                modelId=self.model_name,
                body=json.dumps(body)
            )

            # Parse response
            response_body = json.loads(response['body'].read())

            # Extract content based on model type
            if "anthropic.claude" in self.model_name:
                # Claude models format
                content = response_body.get("content", [{}])[0].get("text", "")
                usage = response_body.get("usage", {})

                return {
                    "content": content,
                    "model": self.model_name,
                    "usage": {
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    }
                }
            else:
                # Generic response parsing (fallback)
                return {
                    "content": response_body.get("completion", ""),
                    "model": self.model_name,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }

        except Exception as e:
            logger.error(f"AWS Bedrock generation failed: {str(e)}")
            raise RuntimeError(
                f"Failed to generate with AWS Bedrock. Error: {str(e)}"
            )


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider using the google-generativeai SDK."""

    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        request_interval: float = 0.0
    ):
        if not api_key:
            raise ValueError(
                "Gemini API key is missing. Set the GEMINI_API_KEY environment variable."
            )

        self.model_name = model_name
        self.model_name_raw = model_name
        self.api_key = api_key
        self.request_interval = max(request_interval or 0.0, 0.0)
        self._last_request_ts: Optional[float] = None
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        self.session = requests.Session()
        logger.info(f"Initialized Gemini provider with model: {model_name}")

    def _respect_rate_limit(self) -> None:
        if self.request_interval <= 0:
            return

        if self._last_request_ts is None:
            return

        elapsed = time.monotonic() - self._last_request_ts
        sleep_for = self.request_interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    @staticmethod
    def _convert_messages(messages: List[Dict[str, str]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        system_instruction_parts: List[str] = []
        contents: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = (msg.get("content") or "").strip()
            if not content:
                continue

            if role == "system":
                system_instruction_parts.append(content)
                continue

            gemini_role = "model" if role == "assistant" else "user"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}],
            })

        system_instruction = "\n".join(system_instruction_parts) if system_instruction_parts else None
        return system_instruction, contents

    @staticmethod
    def _collect_text_fragments(payload: Any) -> List[str]:
        fragments: List[str] = []

        if isinstance(payload, str):
            text = payload.strip()
            if text:
                fragments.append(text)
        elif isinstance(payload, (int, float)):
            fragments.append(str(payload))
        elif isinstance(payload, dict):
            text_field = payload.get("text")
            if isinstance(text_field, str) and text_field.strip():
                fragments.append(text_field.strip())

            preferred_keys = [
                "answer",
                "statement",
                "reason",
                "summary",
                "explanation",
                "content",
                "message",
            ]
            for key in preferred_keys:
                if key in payload:
                    fragments.extend(GeminiProvider._collect_text_fragments(payload[key]))

            for key, value in payload.items():
                if key in preferred_keys or key == "text":
                    continue
                fragments.extend(GeminiProvider._collect_text_fragments(value))
        elif isinstance(payload, list):
            for item in payload:
                fragments.extend(GeminiProvider._collect_text_fragments(item))

        return [frag for frag in fragments if frag]

    @classmethod
    def _normalize_json_text(cls, text: str) -> Optional[str]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None

        fragments = cls._collect_text_fragments(parsed)
        if not fragments:
            return None
        return " ".join(fragments).strip()

    @staticmethod
    def _extract_structured_json(text: str) -> Optional[str]:
        """
        Try to recover a valid JSON document from the model output.
        """
        if not text:
            return None

        candidate = text.strip()
        if not candidate:
            return None

        def _serialize(obj: Any) -> str:
            return json.dumps(obj, ensure_ascii=False)

        try:
            parsed = json.loads(candidate)
            return _serialize(parsed)
        except json.JSONDecodeError:
            pass

        bracket_pairs = [("{", "}"), ("[", "]")]
        for start, end in bracket_pairs:
            start_idx = candidate.find(start)
            end_idx = candidate.rfind(end)
            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                continue
            snippet = candidate[start_idx:end_idx + 1]
            try:
                parsed = json.loads(snippet)
            except json.JSONDecodeError:
                continue
            return _serialize(parsed)

        return None

    @staticmethod
    def _expects_structured_output(messages: List[Dict[str, str]]) -> bool:
        """
        Heuristic to detect prompts that explicitly require machine-readable JSON.
        """
        structured_keywords = [
            "json schema",
            "json format",
            "output json",
            "return json",
            "strict json",
            "please return the output in a json",
            "comply with the following schema",
            "application/json",
        ]

        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = (msg.get("content") or "").lower()
            if any(keyword in content for keyword in structured_keywords):
                return True
        return False

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        system_instruction, base_contents = self._convert_messages(messages)
        base_contents = copy.deepcopy(base_contents)
        expects_structured = self._expects_structured_output(messages)
        logger.debug(
            "Gemini prompt overview: %s",
            [(msg.get("role"), (msg.get("content") or "")[:400]) for msg in messages],
        )
        if not expects_structured:
            first_user = next((msg.get("content", "") for msg in messages if msg.get("role") == "user"), "")
            logger.debug("Gemini debug prompt snippet: %s", first_user[:600])

        if not base_contents:
            raise ValueError("Gemini provider received no user/assistant messages to process.")

        base_generation_config: Dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens:
            base_generation_config["maxOutputTokens"] = max_tokens
        if expects_structured:
            base_generation_config["responseMimeType"] = "application/json"

        system_parts = [{"text": system_instruction}] if system_instruction else []
        extra_system_instruction: Optional[str] = None
        extra_user_prompt: Optional[str] = None
        last_text_response: str = ""
        data: Dict[str, Any] = {}
        final_text: str = ""
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            self._respect_rate_limit()

            payload: Dict[str, Any] = {
                "contents": copy.deepcopy(base_contents),
                "generationConfig": copy.deepcopy(base_generation_config),
            }

            if extra_user_prompt:
                payload["contents"].append({
                    "role": "user",
                    "parts": [{"text": extra_user_prompt}],
                })

            system_parts_combined = list(system_parts)
            if extra_system_instruction:
                system_parts_combined.append({"text": extra_system_instruction})
            if system_parts_combined:
                payload["systemInstruction"] = {
                    "parts": system_parts_combined,
                }

            try:
                response = self.session.post(
                    self.api_url,
                    params={"key": self.api_key},
                    json=payload,
                    timeout=60,
                )
                self._last_request_ts = time.monotonic()
            except requests.RequestException as exc:
                logger.error(f"Gemini request failed: {exc}")
                raise RuntimeError(f"Failed to generate with Gemini. Error: {exc}") from exc

            if not response.ok:
                logger.error(
                    "Gemini API returned error %s: %s",
                    response.status_code,
                    response.text,
                )
                raise RuntimeError(
                    f"Gemini API request failed with status {response.status_code}: {response.text}"
                )

            data = response.json()
            prompt_feedback = data.get("promptFeedback") or {}
            block_reason = prompt_feedback.get("blockReason")
            if block_reason and block_reason != "BLOCK_REASON_UNSPECIFIED":
                logger.error(f"Gemini blocked the prompt: {block_reason}")
                raise RuntimeError(f"Gemini blocked the prompt. Reason: {block_reason}")

            candidates = data.get("candidates") or []
            text_content = ""
            for candidate in candidates:
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    part_text = part.get("text")
                    if part_text:
                        text_content += part_text
                if text_content:
                    break
            text_content = text_content.strip()
            last_text_response = text_content

            if expects_structured:
                extracted_json = self._extract_structured_json(text_content)
                if not extracted_json:
                    logger.warning(
                        "Gemini produced non-JSON output on attempt %s/%s. Retrying with stricter instructions.",
                        attempt,
                        max_attempts,
                    )
                    extra_system_instruction = (
                        "You MUST respond with strict JSON that adheres exactly to the schema provided. "
                        "Return only the JSON payload with no additional commentary."
                    )
                    extra_user_prompt = (
                        "The previous response was invalid. Reply again using ONLY the JSON that matches the schema."
                    )
                    continue
                text_content = extracted_json
            else:
                if text_content.startswith("{") or text_content.startswith("["):
                    normalized = self._normalize_json_text(text_content)
                    if normalized:
                        text_content = normalized

            final_text = text_content
            break
        else:
            logger.error(
                "Gemini failed to produce valid JSON after %s attempts. Last response: %s",
                max_attempts,
                last_text_response,
            )
            raise RuntimeError("Gemini failed to produce a valid JSON response for a structured prompt.")

        usage_meta = data.get("usageMetadata") or {}
        prompt_tokens = int(usage_meta.get("promptTokenCount") or 0)
        completion_tokens = int(usage_meta.get("candidatesTokenCount") or 0)
        total_tokens = usage_meta.get("totalTokenCount")
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(total_tokens) if total_tokens is not None else prompt_tokens + completion_tokens,
        }

        return {
            "content": final_text,
            "model": self.model_name,
            "usage": usage,
        }


def create_llm_provider(
    provider_type: str,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    region_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    aws_profile_name: Optional[str] = None,
    request_interval: Optional[float] = None
) -> LLMProvider:
    """
    Factory function to create LLM provider.

    Args:
        provider_type: Type of provider ('ollama', 'bedrock', or 'gemini')
        model_name: Model name
        api_key: API key (required for Gemini)
        base_url: Base URL (for Ollama)
        region_name: AWS region (for Bedrock)
        aws_access_key_id: AWS access key (for Bedrock)
        aws_secret_access_key: AWS secret key (for Bedrock)
        aws_session_token: AWS session token (for Bedrock)
        aws_profile_name: AWS profile name (for Bedrock)
        request_interval: Minimum delay between requests (optional, useful for Gemini rate limiting)

    Returns:
        LLMProvider instance
    """
    provider_type = provider_type.lower()

    if provider_type == "ollama":
        base_url = base_url or "http://localhost:11434"
        return OllamaProvider(model_name=model_name, base_url=base_url)

    elif provider_type == "bedrock":
        region_name = region_name or "us-east-1"
        return BedrockProvider(
            model_name=model_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            profile_name=aws_profile_name
        )

    elif provider_type == "gemini":
        return GeminiProvider(
            model_name=model_name,
            api_key=api_key,
            request_interval=request_interval or 0.0
        )

    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Use 'ollama' or 'bedrock'")
