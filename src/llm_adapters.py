#!/usr/bin/env python3
"""
LLM Adapters for Audio Benchmark
Unified interface for multiple LLM providers (OpenAI, Anthropic, Google Gemini).

Usage:
    from llm_adapters import create_llm_client
    
    # OpenAI / vLLM
    client = create_llm_client("openai", model="gpt-4o", api_key="...")
    
    # Anthropic
    client = create_llm_client("anthropic", model="claude-3-5-sonnet-20241022", api_key="...")
    
    # Google Gemini
    client = create_llm_client("gemini", model="gemini-1.5-pro", api_key="...")
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Unified message format."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Unified response format."""
    content: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None


class BaseLLMAdapter(ABC):
    """Base class for LLM adapters."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs
        # Provide .chat.completions.create() chain for OpenAI-compatible usage
        self.chat = self._ChatProxy(self)
    
    @abstractmethod
    def _do_chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Send messages and get response. Implement in subclasses."""
        pass
    
    class _ChatProxy:
        """Proxy so that adapter.chat.completions.create() works."""
        def __init__(self, adapter):
            self._adapter = adapter
            self.completions = BaseLLMAdapter._CompletionsProxy(adapter)
    
    class _CompletionsProxy:
        """Proxy so that adapter.chat.completions.create() works."""
        def __init__(self, adapter):
            self._adapter = adapter
        
        def create(self, messages: List[Dict[str, str]], **kwargs) -> Any:
            """OpenAI-compatible interface: returns obj with .choices[0].message.content."""
            # pop model if passed (adapter already knows its model)
            kwargs.pop("model", None)
            response = self._adapter._do_chat(messages, **kwargs)
            
            class Choice:
                class Message:
                    content = response.content
                message = Message()
            
            class Response:
                choices = [Choice()]
                usage = response.usage
            
            return Response()


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI and OpenAI-compatible APIs (vLLM, Ollama, etc.)."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, 
                 api_base: str = "https://api.openai.com/v1", **kwargs):
        super().__init__(model, api_key, **kwargs)
        self.api_base = api_base
        
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
                base_url=api_base
            )
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
    
    def _do_chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        token_limit = kwargs.get("max_tokens", 1024)
        request_kwargs = {
            "model": self.model,
            "messages": messages,
        }

        # GPT-5 family expects max_completion_tokens instead of max_tokens.
        if str(self.model).startswith("gpt-5"):
            request_kwargs["max_completion_tokens"] = token_limit
            # GPT-5 chat-completions supports only default temperature in this setup.
            # Omit temperature to use the API default.
            if "chat-latest" not in str(self.model):
                # Keep reasoning budget small so the model emits parseable text
                # (Thought/Action/Final Answer) within per-turn token limits.
                request_kwargs["reasoning_effort"] = kwargs.get("reasoning_effort", "low")
        else:
            request_kwargs["max_tokens"] = token_limit
            request_kwargs["temperature"] = kwargs.get("temperature", 0.1)

        # gpt-4o-*-audio-preview models require explicit audio modality headers
        if "audio-preview" in str(self.model):
            request_kwargs["modalities"] = ["text", "audio"]
            request_kwargs["audio"] = {"voice": "alloy", "format": "wav"}
            # temperature is not supported for audio-preview
            request_kwargs.pop("temperature", None)

        response = self._client.chat.completions.create(**request_kwargs)

        # audio-preview returns transcript in message.audio.transcript, not .content
        raw_content = response.choices[0].message.content
        if not raw_content and hasattr(response.choices[0].message, "audio") and response.choices[0].message.audio:
            raw_content = response.choices[0].message.audio.transcript or ""

        return LLMResponse(
            content=raw_content,
            usage=dict(response.usage) if response.usage else None,
            model=response.model
        )


class AnthropicAdapter(BaseLLMAdapter):
    """Adapter for Anthropic Claude API."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key, **kwargs)
        
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError("anthropic package required. Install: pip install anthropic")
    
    def _do_chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        # Extract system message (Anthropic handles it separately)
        system_content = None
        filtered_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                filtered_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Build request
        request_kwargs = {
            "model": self.model,
            "messages": filtered_messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
        }
        
        if system_content:
            request_kwargs["system"] = system_content
        
        response = self._client.messages.create(**request_kwargs)
        
        return LLMResponse(
            content=response.content[0].text,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            model=response.model
        )


class GeminiAdapter(BaseLLMAdapter):
    """Adapter for Google Gemini API using the new google-genai SDK (>= 1.0).

    Supports audio files embedded as paths in user messages.  Each unique
    audio file is uploaded once to the Gemini Files API and the resulting
    file reference is reused for the rest of the conversation.
    """

    # Regex to find audio file paths in message text
    _AUDIO_PATH_RE = re.compile(
        r'([A-Za-z0-9_./\\-]+\.(?:wav|mp3|flac|m4a|ogg|aac|opus))',
        re.IGNORECASE,
    )

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key, **kwargs)
        self._upload_cache: Dict[str, Any] = {}  # path -> uploaded File object

        try:
            from google import genai
            self._genai = genai
            resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not resolved_key:
                raise ValueError("No Gemini API key found. Set GEMINI_API_KEY env var.")
            self._client = genai.Client(api_key=resolved_key)
        except ImportError:
            raise ImportError(
                "google-genai package required. Install: pip install google-genai"
            )

    def _upload_audio(self, path: str) -> Any:
        """Upload audio file to Gemini Files API, caching the result."""
        if path not in self._upload_cache:
            if not os.path.exists(path):
                logger.warning("GeminiAdapter: audio path not found: %s", path)
                return None
            logger.info("GeminiAdapter: uploading audio %s", path)
            uploaded = self._client.files.upload(file=path)
            self._upload_cache[path] = uploaded
            logger.info("GeminiAdapter: uploaded as %s", uploaded.name)
        return self._upload_cache[path]

    def _build_parts(self, text: str) -> List[Any]:
        """Return a list of Gemini Part objects for a user message.

        Any recognised audio file paths in the text are extracted, uploaded,
        and added as separate file parts.  The remaining text (with path
        references stripped) is added as a text part.
        """
        from google.genai import types

        parts = []
        audio_paths = []

        for match in self._AUDIO_PATH_RE.finditer(text):
            candidate = match.group(1)
            if os.path.exists(candidate):
                audio_paths.append(candidate)

        # Strip audio paths from the text so the model isn't confused
        clean_text = text
        for p in audio_paths:
            clean_text = clean_text.replace(p, "").strip(", ")
        clean_text = clean_text.strip()

        if clean_text:
            parts.append(types.Part.from_text(text=clean_text))

        for path in audio_paths:
            uploaded = self._upload_audio(path)
            if uploaded is not None:
                parts.append(
                    types.Part.from_uri(
                        file_uri=uploaded.uri,
                        mime_type=uploaded.mime_type or "audio/wav",
                    )
                )

        return parts if parts else [types.Part.from_text(text=text)]

    def _do_chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        import re as _re
        from google.genai import types

        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""

            if role == "system":
                system_instruction = content
                continue

            gemini_role = "model" if role == "assistant" else "user"

            if gemini_role == "user":
                parts = self._build_parts(content)
            else:
                parts = [types.Part.from_text(text=content)]

            contents.append(types.Content(role=gemini_role, parts=parts))

        # Gemini requires the conversation to start with a user turn
        if contents and contents[0].role != "user":
            contents.insert(
                0,
                types.Content(role="user", parts=[types.Part.from_text(text="Begin.")])
            )

        gen_config = types.GenerateContentConfig(
            max_output_tokens=kwargs.get("max_tokens", 2048),
            temperature=kwargs.get("temperature", 0.1),
            system_instruction=system_instruction,
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=gen_config,
        )

        text_out = ""
        try:
            text_out = response.text or ""
        except Exception:
            for part in (response.candidates[0].content.parts if response.candidates else []):
                if hasattr(part, "text"):
                    text_out += part.text or ""

        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0),
                "completion_tokens": getattr(um, "candidates_token_count", 0),
                "total_tokens": getattr(um, "total_token_count", 0),
            }

        return LLMResponse(content=text_out, usage=usage, model=self.model)


class MockLLMAdapter(BaseLLMAdapter):
    """Mock adapter that follows ground truth for testing."""
    
    def __init__(self, dataset: Dict[str, Any], **kwargs):
        super().__init__(model="mock", **kwargs)
        self.dataset = dataset
        self._turn_counter = {}
    
    def _do_chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        # Find matching task based on question in messages
        task = None
        messages_str = str(messages)
        
        for tid, t in self.dataset.items():
            question = t.get("question") or t.get("user_query") or ""
            if question and question in messages_str:
                task = t
                task_id = tid
                break
        
        if not task:
            return LLMResponse(content="Final Answer: Could not identify task.")
        
        # Get or initialize turn counter
        turn_count = self._turn_counter.get(task_id, 0)
        self._turn_counter[task_id] = turn_count + 1
        
        # Get steps from dataset
        steps = task.get("steps", task.get("reference_tool_trace", []))
        
        if turn_count < len(steps):
            step = steps[turn_count]
            tool = step.get("tool", step.get("action", {}).get("tool_name"))
            args = step.get("args", step.get("action", {}).get("args", {}))
            thought = step.get("thought", f"I should use {tool}.")
            
            if tool:
                content = f"Thought: {thought}\nAction: {tool}\nAction Input: {json.dumps(args)}"
            else:
                content = f"Thought: {thought}\nFinal Answer: {task.get('answer', 'Mock answer')}"
        else:
            content = f"Final Answer: {task.get('answer', 'Mock final answer')}"
        
        return LLMResponse(content=content)


# Factory function
def create_llm_client(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    dataset: Optional[Dict] = None,  # For mock mode
    **kwargs
) -> BaseLLMAdapter:
    """
    Create an LLM client for the specified provider.
    
    Args:
        provider: One of "openai", "anthropic", "gemini", "mock"
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        api_key: API key (or set via environment variable)
        api_base: API base URL (for OpenAI-compatible servers)
        dataset: Dataset dict (only for mock provider)
    
    Returns:
        LLM adapter with unified interface
    """
    provider = provider.lower()
    
    if provider == "openai" or provider == "vllm":
        return OpenAIAdapter(
            model=model,
            api_key=api_key,
            api_base=api_base or "https://api.openai.com/v1",
            **kwargs
        )
    
    elif provider == "anthropic" or provider == "claude":
        return AnthropicAdapter(model=model, api_key=api_key, **kwargs)
    
    elif provider == "gemini" or provider == "google":
        return GeminiAdapter(model=model, api_key=api_key, **kwargs)
    
    elif provider == "mock":
        if not dataset:
            raise ValueError("Mock provider requires dataset argument")
        return MockLLMAdapter(dataset=dataset, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: openai, anthropic, gemini, mock"
        )


# Convenience function
def get_provider_from_model(model: str) -> str:
    """Infer provider from model name."""
    model_lower = model.lower()
    
    if "gpt" in model_lower or "o1" in model_lower:
        return "openai"
    elif "claude" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower:
        return "gemini"
    elif "qwen" in model_lower or "llama" in model_lower:
        return "openai"  # Assume vLLM
    else:
        return "openai"  # Default to OpenAI-compatible
