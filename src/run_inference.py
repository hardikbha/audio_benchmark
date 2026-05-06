#!/usr/bin/env python3
"""
Main Inference Script for Audio Benchmark
Runs ReAct agent on benchmark dataset with tool augmentation.

Usage:
    # End-to-end mode (live tool execution)
    python run_inference.py \
        --mode end_to_end \
        --model qwen2-audio \
        --tool_server http://localhost:16181 \
        --dataset data/audio_dataset/dataset.json \
        --output outputs/results.json

    # Step-by-step mode (with ground truth context)
    python run_inference.py \
        --mode step_by_step \
        --model qwen2-audio \
        --dataset data/audio_dataset/dataset.json \
        --output outputs/step_results.json
"""

import argparse
import contextlib
import json
import logging
import os
import re
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from react_agent import ReActAgent, StepByStepAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_inference")


# ============================================================================
# LLM Client Setup
# ============================================================================

def create_llm_client(model_name: str, api_base: str, api_key: str = "EMPTY"):
    """Create OpenAI-compatible client for LLM."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        # Wrap to include model name
        class LLMClientWrapper:
            def __init__(self, client, model):
                self._client = client
                self._model = model
                self.chat = self
                self.completions = self
            
            def create(self, messages=None, **kwargs):
                # react_agent passes model= as kwarg — strip it, we use self._model
                kwargs.pop("model", None)
                # gpt-4o-*-audio-preview models require modalities + audio params
                if "audio-preview" in self._model and "modalities" not in kwargs:
                    kwargs["modalities"] = ["text", "audio"]
                    kwargs["audio"] = {"voice": "alloy", "format": "wav"}
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    **kwargs
                )
                # audio-preview returns transcript in message.audio.transcript, not .content
                if "audio-preview" in self._model:
                    for choice in response.choices:
                        if not choice.message.content and hasattr(choice.message, "audio") and choice.message.audio:
                            choice.message.content = choice.message.audio.transcript or ""
                return response
        
        return LLMClientWrapper(client, model_name)
    except ImportError:
        logger.error("OpenAI package not installed. Run: pip install openai")
        raise


def create_local_llm_client(model_path: str, device: str = None):
    """Create local transformers-based LLM client."""
    try:
        def _try_native_endpoint_client(local_path: str):
            model_dir = Path(local_path).name.lower()
            endpoint_env_map = {
                "audio-flamingo-2": ("AUDIO_FLAMINGO_2_API_BASE", "AUDIO_FLAMINGO_2_MODEL_ID"),
                "freeze-omni": ("FREEZE_OMNI_API_BASE", "FREEZE_OMNI_MODEL_ID"),
                "osum": ("OSUM_API_BASE", "OSUM_MODEL_ID"),
                "speechgpt-7b-com": ("SPEECHGPT_7B_COM_API_BASE", "SPEECHGPT_7B_COM_MODEL_ID"),
                "mini-omni": ("MINI_OMNI_API_BASE", "MINI_OMNI_MODEL_ID"),
                "spiritlm-base": ("SPIRITLM_BASE_API_BASE", "SPIRITLM_BASE_MODEL_ID"),
                "audio-reasoner": ("AUDIO_REASONER_API_BASE", "AUDIO_REASONER_MODEL_ID"),
                "salmonn-7b": ("SALMONN_7B_API_BASE", "SALMONN_7B_MODEL_ID"),
            }

            if model_dir not in endpoint_env_map:
                return None

            api_env, model_env = endpoint_env_map[model_dir]
            api_base = os.getenv(api_env)
            if not api_base:
                return None

            model_id = os.getenv(model_env, Path(local_path).name)
            api_key = os.getenv("NATIVE_MODEL_API_KEY", "EMPTY")
            logger.info(
                "Using native endpoint adapter for %s via %s (model=%s)",
                Path(local_path).name,
                api_base,
                model_id,
            )
            return create_llm_client(model_name=model_id, api_base=api_base, api_key=api_key)

        # Non-HF repos can route through an external native endpoint without local torch/transformers.
        native_client = _try_native_endpoint_client(model_path)
        if native_client is not None:
            return native_client

        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import (
            AutoConfig,
            AutoProcessor,
            AutoTokenizer,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
        )

        logger.info(f"Loading local model: {model_path} on {device}")
        dtype = torch.float16 if device == "cuda" else torch.float32
        device_map = "auto" if device == "cuda" else None

        def _read_raw_config(local_path: str) -> Dict[str, Any]:
            cfg_path = Path(local_path) / "config.json"
            if not cfg_path.exists():
                return {}
            try:
                with open(cfg_path) as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Failed to parse config.json at %s: %s", cfg_path, exc)
                return {}

        def _resolve_effective_model_path(local_path: str, root_cfg: Dict[str, Any]) -> tuple[str, Optional[str]]:
            root = Path(local_path)
            if (root / "config.json").exists():
                model_type = str(root_cfg.get("model_type", "")).lower()
                llm_subdir = root / "llm"
                # Audio-Flamingo style checkpoints: use text backbone for tool-planning prompts.
                if model_type == "llava_llama" and (llm_subdir / "config.json").exists():
                    return str(llm_subdir), "llava_llama_text_backbone"
                return str(root), None

            llm_subdir = root / "llm"
            if (llm_subdir / "config.json").exists():
                return str(llm_subdir), "nested_llm_subdir"

            if (root / "adapter_config.json").exists():
                model_key = root.name.upper().replace("-", "_")
                base_override = (
                    os.getenv(f"{model_key}_BASE_MODEL_PATH")
                    or os.getenv("PEFT_BASE_MODEL_PATH")
                )
                if base_override:
                    return str(root), f"peft_adapter::{base_override}"
                raise RuntimeError(
                    f"Adapter-only checkpoint at {root} without base model config. "
                    f"Set {model_key}_BASE_MODEL_PATH (or PEFT_BASE_MODEL_PATH) to the base model."
                )

            raise RuntimeError(
                f"No config.json found in {root}. This appears to be a non-HF layout without "
                "an auto-detectable text backbone subdirectory."
            )

        def _make_openai_like_response(content: str, prompt_tokens: int, completion_tokens: int):
            total_tokens = prompt_tokens + completion_tokens

            class Choice:
                class Message:
                    def __init__(self, msg_content):
                        self.content = msg_content

                def __init__(self, msg_content):
                    self.message = Choice.Message(msg_content)

            class Response:
                def __init__(self, msg_content):
                    self.choices = [Choice(msg_content)]
                    self.usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }

            return Response(content)

        def _safe_count_tokens(tok, text_value: str) -> int:
            try:
                if hasattr(tok, "encode"):
                    return int(len(tok.encode(text_value, add_special_tokens=False)))
                enc = tok(text_value, return_tensors="pt", add_special_tokens=False)
                if isinstance(enc, dict) and "input_ids" in enc:
                    return int(enc["input_ids"].shape[-1])
            except Exception:
                return 0
            return 0

        def _load_text_processor(local_path: str):
            """Try processor first, then tokenizer."""
            first_error = None
            try:
                proc = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
                logger.info("Loaded processor via AutoProcessor")
                return proc
            except Exception as exc:
                first_error = exc
                logger.warning("AutoProcessor load failed, falling back to AutoTokenizer: %s", exc)
            try:
                tok = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
                logger.info("Loaded processor via AutoTokenizer fallback")
                return tok
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load both AutoProcessor and AutoTokenizer. "
                    f"AutoProcessor error: {first_error}; AutoTokenizer error: {exc}"
                )

        def _get_model_input_device(model_obj):
            """Resolve device to place input tensors for generation."""
            model_dev = getattr(model_obj, "device", None)
            if model_dev is not None and str(model_dev) != "meta":
                return model_dev
            try:
                return next(model_obj.parameters()).device
            except Exception:
                return torch.device("cpu")

        def _to_runtime_device(candidate):
            if device_map is None and hasattr(candidate, "to"):
                candidate = candidate.to(device)
            return candidate

        def _load_with_omni_backbone(local_path: str, raw_cfg: Dict[str, Any], family: str):
            cfg_dict = dict(raw_cfg)
            if family == "qwen2":
                from transformers import Qwen2Config

                cfg_dict["model_type"] = "qwen2"
                cfg_dict["architectures"] = ["Qwen2ForCausalLM"]
                cfg = Qwen2Config.from_dict(cfg_dict)
            elif family == "llama":
                from transformers import LlamaConfig

                cfg_dict["model_type"] = "llama"
                cfg_dict["architectures"] = ["LlamaForCausalLM"]
                cfg = LlamaConfig.from_dict(cfg_dict)
            else:
                raise ValueError(f"Unknown fallback backbone family: {family}")

            logger.info("Applying Omni text-backbone fallback: %s", family)
            candidate = AutoModelForCausalLM.from_pretrained(
                local_path,
                config=cfg,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            return _to_runtime_device(candidate)

        def _render_messages_plain(messages: List[Dict[str, Any]]) -> str:
            lines: List[str] = []
            for msg in messages:
                role = str(msg.get("role", "user")).strip().lower()
                content = msg.get("content", "")
                if isinstance(content, list):
                    pieces = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("text"):
                                pieces.append(str(item["text"]))
                            elif item.get("audio"):
                                pieces.append(str(item["audio"]))
                        else:
                            pieces.append(str(item))
                    content = " ".join(pieces)
                content = str(content).strip()
                if not content:
                    continue
                if role == "system":
                    lines.append(f"System: {content}")
                elif role == "assistant":
                    lines.append(f"Assistant: {content}")
                else:
                    lines.append(f"User: {content}")
            lines.append("Assistant:")
            return "\n\n".join(lines)

        def _extract_audio_path_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
            candidates: List[str] = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("audio"):
                            candidates.append(str(item["audio"]))
                elif isinstance(content, str):
                    # Capture both absolute and relative file paths in prompt text.
                    for match in re.findall(r'([A-Za-z0-9_./\\-]+\.(?:wav|mp3|flac|m4a|ogg))', content, flags=re.IGNORECASE):
                        candidates.append(match)

            for path_candidate in candidates:
                if os.path.exists(path_candidate):
                    return path_candidate
            return None

        def _try_audio_reasoner_client(local_path: str):
            model_dir = Path(local_path).name.lower()
            if model_dir not in {"audio-reasoner", "audio_reasoner"}:
                return None

            checkpoint = os.getenv("AUDIO_REASONER_CHECKPOINT")
            if not checkpoint:
                return None

            try:
                from swift.llm import InferRequest, PtEngine, RequestConfig
            except Exception as exc:
                raise RuntimeError(
                    "Audio-Reasoner native backend requested but swift.llm is unavailable: "
                    f"{exc}"
                ) from exc

            if not os.path.exists(checkpoint):
                raise RuntimeError(
                    f"AUDIO_REASONER_CHECKPOINT does not exist: {checkpoint}"
                )

            model_type = os.getenv("AUDIO_REASONER_MODEL_TYPE", "qwen2_audio")
            system_prompt = os.getenv(
                "AUDIO_REASONER_SYSTEM_PROMPT",
                "You are an audio deep-thinking model. Respond clearly and concisely.",
            )
            default_audio = os.getenv("AUDIO_REASONER_DEFAULT_AUDIO", "")
            logger.info(
                "Initializing Audio-Reasoner native backend with checkpoint=%s model_type=%s",
                checkpoint,
                model_type,
            )
            engine = PtEngine(checkpoint, max_batch_size=1, model_type=model_type)

            class AudioReasonerNativeClient:
                def __init__(self):
                    self.chat = self
                    self.completions = self

                def create(self, messages, **kwargs):
                    prompt_text = _render_messages_plain(messages)
                    audio_path = _extract_audio_path_from_messages(messages) or default_audio
                    if not audio_path or not os.path.exists(audio_path):
                        raise RuntimeError(
                            "Audio-Reasoner native backend needs an audio path. "
                            "Set AUDIO_REASONER_DEFAULT_AUDIO to a valid local file."
                        )

                    request_messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "audio": audio_path},
                                {"type": "text", "text": prompt_text},
                            ],
                        },
                    ]

                    request = InferRequest(messages=request_messages)
                    request_cfg = RequestConfig(
                        max_tokens=int(kwargs.get("max_tokens", 512)),
                        temperature=float(kwargs.get("temperature", 0.1)),
                        stream=False,
                    )
                    response_iter = engine.infer([request], request_cfg)

                    completion = ""
                    for resp_list in response_iter:
                        if not resp_list or resp_list[0] is None:
                            continue
                        choice0 = resp_list[0].choices[0]
                        msg_obj = getattr(choice0, "message", None)
                        delta_obj = getattr(choice0, "delta", None)
                        if msg_obj is not None and getattr(msg_obj, "content", None):
                            completion += str(msg_obj.content)
                        elif delta_obj is not None and getattr(delta_obj, "content", None):
                            completion += str(delta_obj.content)
                        elif getattr(choice0, "text", None):
                            completion += str(choice0.text)

                    if not completion:
                        raise RuntimeError("Audio-Reasoner backend returned empty output.")

                    prompt_tokens = max(1, len(prompt_text.split()))
                    completion_tokens = max(1, len(completion.split()))
                    return _make_openai_like_response(completion, prompt_tokens, completion_tokens)

            return AudioReasonerNativeClient()

        def _try_salmonn_client(local_path: str):
            model_dir = Path(local_path).name.lower()
            if model_dir not in {"salmonn-7b", "salmonn_7b"}:
                return None

            whisper_path = os.getenv("SALMONN_WHISPER_PATH")
            beats_path = os.getenv("SALMONN_BEATS_PATH")
            vicuna_path = os.getenv("SALMONN_VICUNA_PATH")
            ckpt_path = os.getenv(
                "SALMONN_CKPT_PATH",
                str(Path(local_path) / "salmonn_7b_v0.pth"),
            )
            default_audio = os.getenv("SALMONN_DEFAULT_AUDIO", "")
            lora_alpha = int(os.getenv("SALMONN_LORA_ALPHA", "32"))
            low_resource = os.getenv("SALMONN_LOW_RESOURCE", "0") == "1"

            if not (whisper_path and beats_path and vicuna_path):
                return None

            missing = [p for p in [whisper_path, beats_path, vicuna_path, ckpt_path] if not os.path.exists(p)]
            if missing:
                raise RuntimeError(
                    "SALMONN native backend is configured but required paths are missing: "
                    + ", ".join(missing)
                )

            import importlib.util

            model_py = Path(local_path) / "model.py"
            if not model_py.exists():
                raise RuntimeError(f"SALMONN native backend missing model.py at {model_py}")

            spec = importlib.util.spec_from_file_location("salmonn_local_model", str(model_py))
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            SALMONN = getattr(module, "SALMONN")

            logger.info("Initializing SALMONN native backend from %s", local_path)
            salmonn = SALMONN(
                ckpt=ckpt_path,
                whisper_path=whisper_path,
                beats_path=beats_path,
                vicuna_path=vicuna_path,
                lora_alpha=lora_alpha,
                low_resource=low_resource,
            )
            if hasattr(salmonn, "to"):
                salmonn = salmonn.to(device)
            if hasattr(salmonn, "eval"):
                salmonn.eval()

            class SalmonnNativeClient:
                def __init__(self):
                    self.chat = self
                    self.completions = self

                def create(self, messages, **kwargs):
                    prompt_text = _render_messages_plain(messages)
                    audio_path = _extract_audio_path_from_messages(messages) or default_audio
                    if not audio_path or not os.path.exists(audio_path):
                        raise RuntimeError(
                            "SALMONN native backend needs an audio path. "
                            "Set SALMONN_DEFAULT_AUDIO to a valid local file."
                        )
                    output = salmonn.generate(audio_path, prompt=prompt_text)
                    if isinstance(output, list) and output:
                        completion = str(output[0])
                    else:
                        completion = str(output)
                    prompt_tokens = max(1, len(prompt_text.split()))
                    completion_tokens = max(1, len(completion.split()))
                    return _make_openai_like_response(completion, prompt_tokens, completion_tokens)

            return SalmonnNativeClient()

        # Native/non-HF adapters (custom pipelines) are checked before HF loading.
        native_client = _try_audio_reasoner_client(model_path)
        if native_client is not None:
            return native_client

        native_client = _try_salmonn_client(model_path)
        if native_client is not None:
            return native_client

        root_raw_cfg = _read_raw_config(model_path)
        effective_model_path, path_note = _resolve_effective_model_path(model_path, root_raw_cfg)
        peft_base_model_path = None
        if path_note and path_note.startswith("peft_adapter::"):
            peft_base_model_path = path_note.split("::", 1)[1]
            path_note = "peft_adapter"
        if path_note:
            logger.info("Model path adaptation: %s -> %s (%s)", model_path, effective_model_path, path_note)

        # Determine model class using config metadata (not path substrings).
        # Path-based checks can misroute text models if parent folders contain words like "audio".
        model_type = ""
        archs: List[str] = []
        config_error = None
        raw_cfg = _read_raw_config(effective_model_path)
        try:
            config = AutoConfig.from_pretrained(effective_model_path, trust_remote_code=True)
            model_type = str(getattr(config, "model_type", "")).lower()
            archs = [str(a).lower() for a in (getattr(config, "architectures", None) or [])]
        except Exception as exc:
            config_error = exc
            logger.warning(
                "AutoConfig load failed for %s; using raw config fallback: %s",
                effective_model_path,
                exc,
            )

        if not model_type:
            model_type = str(raw_cfg.get("model_type", "")).lower()
        if not archs:
            raw_archs = raw_cfg.get("architectures") or []
            archs = [str(a).lower() for a in raw_archs]

        is_qwen2_audio = (
            "qwen2_audio" in model_type
            or "qwen2audio" in model_type
            or any("qwen2audioforconditionalgeneration" in a for a in archs)
        )
        is_qwen25_omni = (
            "qwen2_5_omni" in model_type
            or "qwen2.5-omni" in model_type
            or any("qwen2_5omniforconditionalgeneration" in a for a in archs)
        )
        is_omni_qwen = "omni2_speech2s_qwen2" in model_type
        is_omni_llama = "omni_speech2s_llama" in model_type
        is_kimi = any("moonshotkimiaforcausallm" in a for a in archs)
        is_gpt_oss = (
            "gpt_oss" in model_type
            or any("gptossforcausallm" in a for a in archs)
        )

        logger.info(
            "Model config detected: model_type=%s architectures=%s",
            model_type or "unknown",
            archs or ["unknown"],
        )
        if config_error is not None:
            logger.info("Proceeding without AutoConfig metadata due to earlier failure.")

        def _load_peft_merged_model(adapter_path: str, base_model_path: str):
            from peft import PeftModel

            logger.info(
                "Loading PEFT adapter with base model: adapter=%s base=%s",
                adapter_path,
                base_model_path,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            peft_model = PeftModel.from_pretrained(base_model, adapter_path)
            if hasattr(peft_model, "merge_and_unload"):
                merged = peft_model.merge_and_unload()
            else:
                merged = peft_model
            return _to_runtime_device(merged)

        def _is_flash_attn_error(exc: Exception) -> bool:
            msg = str(exc).lower()
            return ("flashattention" in msg) or ("flash_attn" in msg)

        def _load_with_attn_fallback(loader_cls, local_path: str):
            load_dtype = dtype
            # gpt-oss dequantization/runtime kernels expect bf16 on GPU.
            if device == "cuda" and is_gpt_oss:
                load_dtype = torch.bfloat16
            base_kwargs = {
                "torch_dtype": load_dtype,
                "device_map": device_map,
                "trust_remote_code": True,
            }
            try:
                return _to_runtime_device(loader_cls.from_pretrained(local_path, **base_kwargs))
            except Exception as exc:
                if not _is_flash_attn_error(exc):
                    raise
                logger.warning(
                    "Loader %s failed due to flash_attn (%s). Retrying without flash attention.",
                    loader_cls.__name__,
                    exc,
                )
                fallback_errors = []
                for attn_impl in ("sdpa", "eager"):
                    try:
                        return _to_runtime_device(
                            loader_cls.from_pretrained(
                                local_path,
                                **base_kwargs,
                                attn_implementation=attn_impl,
                            )
                        )
                    except Exception as attn_exc:
                        fallback_errors.append(f"{attn_impl}: {attn_exc}")
                raise RuntimeError(
                    f"{loader_cls.__name__} failed with flash_attn and fallback attention modes: "
                    + " | ".join(fallback_errors)
                ) from exc

        def _load_generation_model(local_path: str):
            """Load a model with progressive fallback and model-specific adapters."""
            attempts: List[tuple[str, Any]] = []

            if peft_base_model_path:
                attempts.append(
                    (
                        "PEFTAdapterMerge",
                        lambda p: _load_peft_merged_model(p, peft_base_model_path),
                    )
                )

            if is_omni_qwen:
                attempts.append(
                    ("OmniQwenTextFallback", lambda p: _load_with_omni_backbone(p, raw_cfg, "qwen2"))
                )
            if is_omni_llama:
                attempts.append(
                    ("OmniLlamaTextFallback", lambda p: _load_with_omni_backbone(p, raw_cfg, "llama"))
                )
            if is_qwen25_omni:
                from transformers import Qwen2_5OmniForConditionalGeneration

                attempts.append(
                    (
                        "Qwen2_5OmniForConditionalGeneration",
                        lambda p: _load_with_attn_fallback(Qwen2_5OmniForConditionalGeneration, p),
                    )
                )
            if is_qwen2_audio:
                from transformers import Qwen2AudioForConditionalGeneration

                attempts.append(
                    (
                        "Qwen2AudioForConditionalGeneration",
                        lambda p: _load_with_attn_fallback(Qwen2AudioForConditionalGeneration, p),
                    )
                )

            attempts.extend(
                [
                    (
                        "AutoModelForCausalLM",
                        lambda p: _load_with_attn_fallback(AutoModelForCausalLM, p),
                    ),
                    (
                        "AutoModelForSeq2SeqLM",
                        lambda p: _load_with_attn_fallback(AutoModelForSeq2SeqLM, p),
                    ),
                    (
                        "AutoModel",
                        lambda p: _load_with_attn_fallback(AutoModel, p),
                    ),
                ]
            )

            errors = []
            for loader_name, loader_fn in attempts:
                try:
                    candidate = loader_fn(local_path)
                    generation_mode = (
                        "generate"
                        if callable(getattr(candidate, "generate", None))
                        else "manual_decode"
                    )
                    logger.info(
                        "Loaded local model with %s (generation_mode=%s)",
                        loader_name,
                        generation_mode,
                    )
                    return candidate, loader_name, generation_mode
                except Exception as exc:
                    errors.append(f"{loader_name}: {exc}")
                    logger.warning("Model load failed with %s: %s", loader_name, exc)
            raise RuntimeError("No compatible local model loader succeeded: " + " | ".join(errors))

        # We operate text-first for orchestration prompts; audio tools handle waveform analysis.
        processor = _load_text_processor(effective_model_path)
        model, loader_used, generation_mode = _load_generation_model(effective_model_path)
        logger.info("Local loader selected: %s", loader_used)

        class LocalLLMClient:
            def __init__(self, model, processor, generation_mode: str, model_flavor: str):
                self._model = model
                self._processor = processor
                self._generation_mode = generation_mode
                self._model_flavor = model_flavor
                self.chat = self
                self.completions = self

            def _inference_autocast_ctx(self):
                """Enable autocast for BF16 models on CPU/CUDA during inference."""
                try:
                    first_param = next(self._model.parameters())
                    dev = first_param.device.type
                    dt = first_param.dtype
                    if dt == torch.bfloat16 and dev in {"cpu", "cuda"}:
                        return torch.autocast(device_type=dev, dtype=torch.bfloat16)
                except Exception:
                    pass
                return contextlib.nullcontext()

            def _extract_text_logits(self, model_outputs):
                logits = getattr(model_outputs, "logits", None)
                if logits is None and isinstance(model_outputs, Mapping):
                    logits = model_outputs.get("logits")

                if isinstance(logits, (tuple, list)):
                    # Kimi returns (audio_logits, text_logits)
                    if self._model_flavor == "kimi" and len(logits) >= 2 and torch.is_tensor(logits[1]):
                        return logits[1]
                    for part in reversed(logits):
                        if torch.is_tensor(part):
                            return part
                    return None

                if torch.is_tensor(logits):
                    return logits
                return None

            def _manual_decode(
                self,
                inputs: Dict[str, Any],
                max_new_tokens: int,
                temperature: float,
                eos_ids: List[int],
            ) -> torch.Tensor:
                if "input_ids" not in inputs:
                    raise RuntimeError("Manual decoding requires input_ids.")

                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

                generated: List[int] = []
                do_sample = temperature > 0
                safe_temp = max(temperature, 1e-5)
                amp_ctx = self._inference_autocast_ctx()

                with torch.no_grad():
                    with amp_ctx:
                        for _ in range(max_new_tokens):
                            outputs = self._model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                return_dict=True,
                                use_cache=False,
                            )
                            logits = self._extract_text_logits(outputs)
                            if logits is None:
                                raise RuntimeError("Manual decoding failed: model outputs do not expose logits.")
                            next_token_logits = logits[:, -1, :]
                            if do_sample:
                                probs = torch.softmax(next_token_logits / safe_temp, dim=-1)
                                next_token = torch.multinomial(probs, num_samples=1)
                            else:
                                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                            token_id = int(next_token.item())
                            if eos_ids and token_id in eos_ids:
                                break
                            generated.append(token_id)

                            input_ids = torch.cat([input_ids, next_token], dim=1)
                            attention_mask = torch.cat(
                                [attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)],
                                dim=1,
                            )

                if not generated:
                    return torch.empty((0,), dtype=torch.long, device=input_ids.device)
                return torch.tensor(generated, dtype=torch.long, device=input_ids.device)

            def create(self, messages, **kwargs):
                def _render_messages_fallback(msgs: List[Dict[str, str]]) -> str:
                    """Render chat messages without tokenizer chat template support."""
                    lines = []
                    for msg in msgs:
                        role = str(msg.get("role", "user")).strip().lower()
                        content = str(msg.get("content", "")).strip()
                        if not content:
                            continue
                        if role == "system":
                            lines.append(f"System: {content}")
                        elif role == "assistant":
                            lines.append(f"Assistant: {content}")
                        else:
                            lines.append(f"User: {content}")
                    lines.append("Assistant:")
                    return "\n\n".join(lines)

                def _encode_inputs(proc, tok, prompt_text: str):
                    """Encode prompt across tokenizer/processor variants."""
                    errors = []

                    def _normalize_encoding(enc_obj):
                        if enc_obj is None:
                            return None
                        if isinstance(enc_obj, Mapping):
                            if "input_ids" not in enc_obj:
                                return None
                            norm = dict(enc_obj)
                            input_ids = norm.get("input_ids")
                            if not torch.is_tensor(input_ids):
                                if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                                    norm["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
                                else:
                                    norm["input_ids"] = torch.tensor([input_ids], dtype=torch.long)
                            attn = norm.get("attention_mask")
                            if attn is not None and not torch.is_tensor(attn):
                                if isinstance(attn, list) and attn and isinstance(attn[0], list):
                                    norm["attention_mask"] = torch.tensor(attn, dtype=torch.long)
                                else:
                                    norm["attention_mask"] = torch.tensor([attn], dtype=torch.long)
                            return norm
                        if hasattr(enc_obj, "input_ids"):
                            ids = getattr(enc_obj, "input_ids")
                            mask = getattr(enc_obj, "attention_mask", None)
                            if not torch.is_tensor(ids):
                                if isinstance(ids, list) and ids and isinstance(ids[0], list):
                                    ids = torch.tensor(ids, dtype=torch.long)
                                else:
                                    ids = torch.tensor([ids], dtype=torch.long)
                            out = {"input_ids": ids}
                            if mask is not None:
                                if not torch.is_tensor(mask):
                                    if isinstance(mask, list) and mask and isinstance(mask[0], list):
                                        mask = torch.tensor(mask, dtype=torch.long)
                                    else:
                                        mask = torch.tensor([mask], dtype=torch.long)
                                out["attention_mask"] = mask
                            return out
                        if isinstance(enc_obj, list):
                            return {"input_ids": torch.tensor([enc_obj], dtype=torch.long)}
                        return None

                    # Prefer tokenizer-only path for text prompts. Some multimodal processors
                    # (e.g. Voxtral/Mistral variants) pass unsupported kwargs to tokenizer.
                    for fn_name, fn in [
                        ("tokenizer_text_kw", lambda: tok(prompt_text, return_tensors="pt")),
                        ("tokenizer_positional", lambda: tok(prompt_text)),
                        ("tokenizer_encode", lambda: tok.encode(prompt_text)),
                        ("processor_text_kw", lambda: proc(text=prompt_text, return_tensors="pt")),
                        ("processor_positional", lambda: proc(prompt_text, return_tensors="pt")),
                    ]:
                        try:
                            enc = _normalize_encoding(fn())
                            if enc is not None and "input_ids" in enc:
                                return enc
                        except Exception as exc:
                            errors.append(f"{fn_name}: {exc}")
                    raise RuntimeError("Could not encode prompt for local model: " + " | ".join(errors))

                def _resolve_context_limit(tok) -> Optional[int]:
                    candidates: List[int] = []
                    tok_limit = getattr(tok, "model_max_length", None)
                    if isinstance(tok_limit, int) and 0 < tok_limit < 1_000_000:
                        candidates.append(tok_limit)
                    cfg = getattr(self._model, "config", None)
                    if cfg is not None:
                        for key in (
                            "max_position_embeddings",
                            "n_positions",
                            "seq_length",
                            "max_sequence_length",
                            "max_seq_len",
                        ):
                            val = getattr(cfg, key, None)
                            if isinstance(val, int) and 0 < val < 1_000_000:
                                candidates.append(val)
                    if not candidates:
                        return None
                    return min(candidates)

                tokenizer = self._processor.tokenizer if hasattr(self._processor, "tokenizer") else self._processor

                text = None
                if hasattr(tokenizer, "apply_chat_template"):
                    try:
                        if getattr(tokenizer, "chat_template", None):
                            text = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        else:
                            logger.warning(
                                "Tokenizer has apply_chat_template but no chat_template; "
                                "using manual prompt rendering fallback."
                            )
                    except Exception as exc:
                        logger.warning(
                            "apply_chat_template failed (%s); using manual prompt rendering fallback.",
                            exc,
                        )
                if text is None:
                    text = _render_messages_fallback(messages)

                inputs = _encode_inputs(self._processor, tokenizer, text)
                prompt_tokens = (
                    int(inputs["input_ids"].shape[-1])
                    if "input_ids" in inputs
                    else _safe_count_tokens(tokenizer, text)
                )
                input_device = _get_model_input_device(self._model)
                inputs = {k: (v.to(input_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

                eos_id = getattr(tokenizer, "eos_token_id", None)
                eos_ids = []
                if isinstance(eos_id, list):
                    eos_ids = [int(x) for x in eos_id]
                elif eos_id is not None:
                    eos_ids = [int(eos_id)]
                pad_id = getattr(tokenizer, "pad_token_id", None)
                generation_pad_id = eos_id if eos_id is not None else pad_id

                max_new_tokens = int(kwargs.get("max_tokens", 512))
                temperature = float(kwargs.get("temperature", 0.1))

                # Prevent sequence overflow on small-context models (e.g., 2048 token checkpoints).
                ctx_limit = _resolve_context_limit(tokenizer)
                if ctx_limit is not None and "input_ids" in inputs:
                    prompt_len = int(inputs["input_ids"].shape[-1])
                    safety_margin = 8
                    keep_len = max(32, ctx_limit - max_new_tokens - safety_margin)
                    if prompt_len > keep_len:
                        inputs["input_ids"] = inputs["input_ids"][:, -keep_len:]
                        if "attention_mask" in inputs and hasattr(inputs["attention_mask"], "shape"):
                            inputs["attention_mask"] = inputs["attention_mask"][:, -keep_len:]
                        logger.warning(
                            "Prompt truncated for local model context limit: original=%d keep=%d ctx_limit=%d",
                            prompt_len,
                            keep_len,
                            ctx_limit,
                        )
                        prompt_tokens = int(inputs["input_ids"].shape[-1])

                generated_ids = None
                if self._generation_mode == "generate":
                    try:
                        with torch.no_grad():
                            with self._inference_autocast_ctx():
                                outputs = self._model.generate(
                                    **inputs,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature,
                                    do_sample=temperature > 0,
                                    pad_token_id=generation_pad_id,
                                )
                        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
                        if "input_ids" in inputs:
                            generated_ids = sequences[0][inputs["input_ids"].shape[1]:]
                        else:
                            generated_ids = sequences[0]
                    except Exception as exc:
                        logger.warning("generate() failed (%s). Falling back to manual decoding.", exc)

                if generated_ids is None:
                    generated_ids = self._manual_decode(
                        inputs=inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        eos_ids=eos_ids,
                    )

                if hasattr(tokenizer, "decode"):
                    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                elif hasattr(self._processor, "batch_decode"):
                    response_text = self._processor.batch_decode([generated_ids], skip_special_tokens=True)[0]
                else:
                    response_text = str(generated_ids)

                # Truncate to avoid repetitive hallucinations in small models.
                if "Action Input:" in response_text:
                    parts = response_text.split("Action Input:", 1)
                    after_input = parts[1]
                    brace_index = after_input.find("}")
                    if brace_index != -1:
                        response_text = parts[0] + "Action Input:" + after_input[:brace_index + 1]
                elif "Final Answer:" in response_text:
                    parts = response_text.split("Final Answer:", 1)
                    after_final = parts[1].split("\n")[0]
                    response_text = parts[0] + "Final Answer: " + after_final

                completion_tokens = _safe_count_tokens(tokenizer, response_text)
                return _make_openai_like_response(response_text, prompt_tokens, completion_tokens)

        model_flavor = "kimi" if is_kimi else "generic"
        return LocalLLMClient(model, processor, generation_mode=generation_mode, model_flavor=model_flavor)
    
    except Exception as e:
        logger.error(f"Failed to load local model: {e}")
        raise


def create_mock_llm_client(dataset: Dict[str, Any]):
    """Create a mock LLM client that follows ground truth steps."""
    def _extract_step_tool_args(step: Dict[str, Any]):
        action_field = step.get("action")
        action_tool = None
        action_args = None

        if isinstance(action_field, dict):
            action_tool = action_field.get("tool_name")
            action_args = action_field.get("args")
        elif isinstance(action_field, str):
            action_tool = action_field
            action_args = step.get("action_input")

        tool = step.get("tool") or action_tool
        args = step.get("args") or action_args or step.get("action_input") or {}
        if not isinstance(args, dict):
            args = {}
        return tool, args

    class MockLLMClient:
        def __init__(self, dataset):
            self.dataset = dataset
            self.chat = self
            self.completions = self
            self.current_task_id = None
            self.step_index = 0
            
        def create(self, messages, **kwargs):
            # Extract task context
            content = messages[-1]["content"]
            
            # Identify task based on question/content matching
            # This is a heuristic since we don't pass task_id explicitly to solve()
            # In a real run we'd pass it, but for mock this works
            task = None
            messages_str = str(messages)
            for tid, t in self.dataset.items():
                question = t.get("question") or t.get("user_query") or ""
                if not question and isinstance(t.get("dialog"), list):
                    for turn in t["dialog"]:
                        if turn.get("role") == "user" and turn.get("content"):
                            question = str(turn["content"])
                            break
                if question and question in messages_str:
                    task = t
                    break
            
            if not task:
                return self._response("Final Answer: I could not identify the task to mock.")
            
            # Determine which step we are on based on message history length
            # System + User = 2 messages. Each turn adds Assistant + User = 2 messages.
            # So (len(messages) - 2) / 2 = current turn index
            turn_idx = (len(messages) - 2) // 2
            
            # Determine which step we are on based on reference trace matching
            # We iterate through steps and see how many we've already "used"
            # But simpler: use turn_idx
            
            # Filter out empty steps from dataset if they are just thoughts before actions
            # We want to combine Thought (step N) + Action (step N+1)
            
            steps = task.get("steps", task.get("reference_tool_trace", []))
            
            # Construct a clean list of ReAct turns
            react_turns = []
            i = 0
            while i < len(steps):
                step = steps[i]
                tool, args = _extract_step_tool_args(step)
                thought = step.get("thought", "")
                
                # Check if next step is the action for this thought
                if not tool and i + 1 < len(steps):
                    next_step = steps[i+1]
                    next_tool, next_args = _extract_step_tool_args(next_step)
                    if next_tool:
                        # Combine
                        tool = next_tool
                        args = next_args
                        # Append next thought if any? Usually empty
                        i += 1
                
                react_turns.append({
                    "thought": thought,
                    "tool": tool,
                    "args": args
                })
                i += 1
                
            # Now use turn_idx to pick from react_turns
            turn_idx = (len(messages) - 2) // 2
            
            if turn_idx < len(react_turns):
                turn = react_turns[turn_idx]
                thought = turn["thought"] or f"I should use {turn['tool']}."
                tool = turn["tool"]
                args = turn["args"]
                
                if tool:
                    return self._response(f"Thought: {thought}\nAction: {tool}\nAction Input: {json.dumps(args)}")
                else:
                    return self._response(f"Thought: {thought}\nFinal Answer: {task.get('answer', 'Mock answer')}")
            else:
                return self._response(f"Final Answer: {task.get('answer', 'Mock final answer based on tool outputs.')}")
        
        def _response(self, content):
            class Choice:
                class Message:
                    content = ""
                message = Message()
            
            c = Choice()
            c.message.content = content
            
            class Response:
                choices = [c]
            
            return Response()

    logger.info("Using Mock LLM Client (Verification Mode)")
    return MockLLMClient(dataset)


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """Load benchmark dataset from JSON or JSONL file."""
    path = Path(dataset_path)
    
    if path.suffix == ".jsonl":
        # JSONL format
        dataset = {}
        with open(path) as f:
            for line in f:
                item = json.loads(line.strip())
                dataset[item.get("id", str(len(dataset)))] = item
        return dataset
    else:
        # JSON format (GTA-style)
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                # Convert list to dict indexed by id
                dataset = {}
                for i, item in enumerate(data):
                    tid = item.get("id", str(i))
                    dataset[tid] = item
                return dataset
            return data


def load_toolmeta(toolmeta_path: str) -> Dict[str, Any]:
    """Load tool metadata."""
    with open(toolmeta_path) as f:
        return json.load(f)


# ============================================================================
# Inference Functions
# ============================================================================

def run_end_to_end(
    agent: ReActAgent,
    dataset: Dict[str, Any],
    audio_base_path: str,
    output_path: str,
    limit: Optional[int] = None,
    checkpoint_interval: int = 10,
    existing_results: Optional[Dict] = None,
    judge_client=None,
    judge_model: str = "gpt-4o-mini",
    judge_provider: str = "openai",
):
    """Run end-to-end inference on dataset with checkpointing."""
    results = existing_results.copy() if existing_results else {}
    items = list(dataset.items())
    
    if limit:
        items = items[:limit]
    
    # Filter out already completed tasks if resuming
    if existing_results:
        items = [(tid, task) for tid, task in items if tid not in existing_results]
        logger.info(f"Skipping {len(existing_results)} completed tasks, {len(items)} remaining")
    
    completed = 0
    for task_id, task in tqdm(items, desc="Running end-to-end inference"):
        # Support both flat and nested GTA formats
        question = task.get("question", task.get("user_query", ""))
        if not question and "dialog" in task and len(task["dialog"]) > 0:
            # Prefer first user utterance from dialog-style datasets
            for turn in task["dialog"]:
                if turn.get("role") == "user" and turn.get("content"):
                    question = turn["content"]
                    break
            
        audio_files = task.get("file", task.get("audio_files", []))
        if not audio_files and task.get("audio_path"):
            audio_files = [task["audio_path"]]
        if not audio_files and "image" in task:
            audio_files = [task["image"]]
            
        if isinstance(audio_files, str):
            audio_files = [audio_files]
        
        # Resolve audio paths
        resolved_files = []
        for f in audio_files:
            if not isinstance(f, str):
                continue

            candidate = f
            if os.path.isabs(candidate):
                resolved_files.append(candidate)
                continue

            # Keep path as-is if it already exists from repo root.
            if os.path.exists(candidate):
                resolved_files.append(candidate)
                continue

            # Fall back to configured audio base path.
            base_joined = os.path.join(audio_base_path, candidate)
            resolved_files.append(base_joined)
        
        logger.info(f"Processing {task_id}: {question[:50]}...")
        
        try:
            trace = agent.solve(question, resolved_files)
            results[task_id] = {
                "task_id": task_id,
                "question": question,
                "audio_files": resolved_files,
                "trace": trace.to_dict(),
                "predicted_answer": trace.final_answer,
                "tools_called": trace.tools_called,
                "success": trace.success,
                "ground_truth": task.get("answer", task.get("gold_answer", task.get("groundtruth_answer", task.get("ground_truth")))),
                "expected_tools": [],
                # Resource/performance metrics for cost-aware benchmarking.
                "llm_calls": trace.llm_calls,
                "llm_prompt_tokens": trace.llm_prompt_tokens,
                "llm_completion_tokens": trace.llm_completion_tokens,
                "llm_total_tokens": trace.llm_total_tokens,
                "llm_time_sec": trace.llm_time_sec,
                "tool_time_sec": trace.tool_time_sec,
                "total_time_sec": trace.total_time_sec,
            }
            
            # Extract expected tools
            expected_tools = [
                s.get("tool", s.get("action"))
                for s in task.get("steps", task.get("reference_tool_trace", []))
            ]
            if not expected_tools and "dialog" in task:
                for turn in task["dialog"]:
                    if turn.get("role") == "assistant" and "action" in turn:
                        action = turn["action"]
                        if isinstance(action, dict):
                            expected_tools.append(action.get("name"))
                        else:
                            expected_tools.append(action)
            if not expected_tools and "tools" in task:
                for tool in task.get("tools", []):
                    if isinstance(tool, dict):
                        if tool.get("name"):
                            expected_tools.append(tool["name"])
                    elif isinstance(tool, str):
                        expected_tools.append(tool)
            
            results[task_id]["expected_tools"] = expected_tools

            # ── Inline LLM-judge scoring ──────────────────────────────────────
            if judge_client is not None:
                try:
                    from llm_judge_eval import call_llm_judge
                    _gt   = results[task_id].get("ground_truth")
                    _pred = results[task_id].get("predicted_answer", "")
                    _jr   = call_llm_judge(
                        judge_client, judge_model, judge_provider,
                        question, _gt, _pred,
                    )
                    results[task_id]["llm_judge_score"]     = _jr.score
                    results[task_id]["llm_judge_reasoning"] = _jr.reasoning
                    results[task_id]["llm_judge_category"]  = _jr.category
                    logger.info(f"  Judge score for {task_id}: {_jr.score:.2f} ({_jr.category})")
                except Exception as _jex:
                    logger.warning(f"  Judge scoring failed for {task_id}: {_jex}")

        except Exception as e:
            logger.error(f"Failed on {task_id}: {e}")
            results[task_id] = {
                "task_id": task_id,
                "question": question,
                "error": str(e),
                "success": False
            }
        
        completed += 1
        
        # Checkpoint save
        if checkpoint_interval > 0 and completed % checkpoint_interval == 0:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Checkpoint saved: {completed} tasks completed")
    
    # Final save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_path}")
    return results


def extract_steps_from_dialog(dialog: List[Dict]) -> List[Dict]:
    """Extract ground truth steps from dialog field.

    Dialog format:
        - user turn
        - assistant turn (thought + action)
        - tool turn (results)
        - assistant turn (thought + action)
        - tool turn (results)
        - ...
        - assistant turn (final answer, no action)

    Returns list of steps with format:
        [{"tool": "tool_name", "arguments": {...}}, ...]
    """
    steps = []
    for turn in dialog:
        if turn.get('role') == 'assistant' and 'action' in turn:
            action = turn['action']
            if isinstance(action, dict) and 'name' in action:
                steps.append({
                    'tool': action['name'],
                    'arguments': action.get('arguments', {}),
                    'thought': turn.get('content', '')
                })
    return steps


def run_step_by_step(
    agent: StepByStepAgent,
    dataset: Dict[str, Any],
    audio_base_path: str,
    output_path: str,
    limit: Optional[int] = None,
    checkpoint_interval: int = 10,
    existing_results: Optional[Dict] = None
):
    """Run step-by-step inference on dataset with checkpointing."""
    results = existing_results.copy() if existing_results else {}
    items = list(dataset.items())

    if limit:
        items = items[:limit]

    # Filter out already completed tasks if resuming
    if existing_results:
        items = [(tid, task) for tid, task in items if tid not in existing_results]
        logger.info(f"Skipping {len(existing_results)} completed tasks, {len(items)} remaining")

    completed = 0
    for task_id, task in tqdm(items, desc="Running step-by-step inference"):
        question = task.get("question", task.get("user_query", ""))
        if not question and "dialog" in task and len(task["dialog"]) > 0:
            for turn in task["dialog"]:
                if turn.get("role") == "user" and turn.get("content"):
                    question = turn["content"]
                    break
        audio_files = task.get("file", task.get("audio_files", []))
        if not audio_files and task.get("audio_path"):
            audio_files = [task["audio_path"]]
        if not audio_files and "image" in task:
            audio_files = [task["image"]]
        if isinstance(audio_files, str):
            audio_files = [audio_files]

        # Resolve audio paths
        resolved_files = []
        for f in audio_files:
            if not isinstance(f, str):
                continue

            candidate = f
            if os.path.isabs(candidate):
                resolved_files.append(candidate)
                continue

            # Keep path as-is if it already exists from repo root.
            if os.path.exists(candidate):
                resolved_files.append(candidate)
                continue

            # Fall back to configured audio base path.
            base_joined = os.path.join(audio_base_path, candidate)
            resolved_files.append(base_joined)

        # Extract ground truth steps from dialog field if available
        gt_steps = task.get("steps", task.get("reference_tool_trace", []))
        if not gt_steps and "dialog" in task:
            gt_steps = extract_steps_from_dialog(task["dialog"])
        
        task_results = {
            "task_id": task_id,
            "question": question,
            "predictions": [],
            "ground_truth_steps": gt_steps
        }
        
        # For each step position, predict the next step
        for step_idx in range(len(gt_steps)):
            previous_steps = gt_steps[:step_idx]
            
            try:
                prediction = agent.predict_next_step(question, resolved_files, previous_steps)
                
                # Compare with ground truth step
                gt_step = gt_steps[step_idx]
                gt_tool = gt_step.get("tool", gt_step.get("action", {}).get("tool_name", ""))
                pred_tool = prediction.get("action", "")
                
                task_results["predictions"].append({
                    "step_idx": step_idx + 1,
                    "predicted_tool": pred_tool,
                    "predicted_args": prediction.get("action_input"),
                    "ground_truth_tool": gt_tool,
                    "tool_match": pred_tool == gt_tool,
                    "raw_response": prediction.get("raw_response", "")[:500]
                })
            except Exception as e:
                task_results["predictions"].append({
                    "step_idx": step_idx + 1,
                    "error": str(e)
                })
        
        results[task_id] = task_results

        completed += 1

        # Checkpoint save
        if checkpoint_interval > 0 and completed % checkpoint_interval == 0:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Checkpoint saved: {completed} tasks completed")

    # Final save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_path}")
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Audio Benchmark Inference")
    
    # Mode
    parser.add_argument("--mode", choices=["end_to_end", "step_by_step"], 
                       default="end_to_end", help="Evaluation mode")
    
    # LLM Provider
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini", "vllm", "mock"],
                       default="openai", help="LLM provider")
    parser.add_argument("--model", default="qwen2-audio",
                       help="Model name or path")
    parser.add_argument("--api_base", default="http://localhost:8000/v1",
                       help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key", default=None,
                       help="API key (reads from env if not set)")
    parser.add_argument("--local", action="store_true",
                       help="Use local transformers model instead of API")
    parser.add_argument("--mock", action="store_true",
                       help="Use mock LLM that follows dataset ground truth (for testing)")
    
    # Tool execution
    parser.add_argument("--tool_server", default="http://localhost:16181",
                       help="Tool server URL (for HTTP mode)")
    parser.add_argument("--in_process", action="store_true",
                       help="Execute tools in-process (no HTTP server needed)")
    parser.add_argument("--toolmeta", default="data/audio_dataset/toolmeta.json",
                       help="Path to tool metadata JSON")
    
    # Dataset
    parser.add_argument("--dataset", default="data/audio_dataset/dataset.json",
                       help="Path to benchmark dataset")
    parser.add_argument("--audio_base", default="data/audio_dataset/audio_assets",
                       help="Base path for audio files")
    
    # Output & Checkpointing
    parser.add_argument("--output", default=None,
                       help="Output path for results JSON")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of tasks to process")
    parser.add_argument("--checkpoint", type=int, default=10,
                       help="Save checkpoint every N tasks (0 to disable)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint if exists")
    
    # Agent settings
    parser.add_argument("--max_turns", type=int, default=10,
                       help="Maximum turns for ReAct agent")
    parser.add_argument("--judge", action="store_true",
                       help="Run LLM-as-judge scoring inline after each query")
    parser.add_argument("--judge_model", default="gpt-4o-mini",
                       help="Model to use as LLM judge (default: gpt-4o-mini)")
    parser.add_argument("--judge_provider", default="openai",
                       choices=["openai", "anthropic"],
                       help="Provider for judge model (default: openai)")
    parser.add_argument("--judge_api_key", default=None,
                       help="API key for judge (falls back to OPENAI_API_KEY env var)")

    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"outputs/{args.mode}_{timestamp}/results.json"
    
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")

    # ── Inline LLM-judge setup ────────────────────────────────────────────────
    _judge_client = None
    _judge_model  = None
    if getattr(args, "judge", False):
        try:
            import os as _os
            from llm_judge_eval import create_llm_judge_client, call_llm_judge
            _judge_key = getattr(args, "judge_api_key", None) or _os.getenv("OPENAI_API_KEY")
            if getattr(args, "judge_provider", "openai") == "openai":
                _os.environ.setdefault("OPENAI_API_BASE", "https://api.openai.com/v1")
            _judge_client, _judge_model = create_llm_judge_client(
                provider=getattr(args, "judge_provider", "openai"),
                model=getattr(args, "judge_model", "gpt-4o-mini"),
                api_key=_judge_key,
            )
            logger.info(f"LLM judge enabled: {args.judge_provider}/{args.judge_model}")
        except Exception as _je:
            logger.warning(f"Could not initialise LLM judge: {_je}. Inline scoring disabled.")
            _judge_client = None
    
    # Load dataset and toolmeta
    logger.info("Loading dataset and tool metadata...")
    dataset = load_dataset(args.dataset)
    toolmeta = load_toolmeta(args.toolmeta)
    
    logger.info(f"Loaded {len(dataset)} tasks and {len(toolmeta)} tools")
    
    # Copy dataset file to output folder for reproducibility
    try:
        if args.output:
            import shutil
            output_dir = Path(args.output).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            if Path(args.dataset).exists():
                shutil.copy2(args.dataset, output_dir / "dataset.json")
                logger.info(f"Copied dataset to {output_dir / 'dataset.json'}")
    except Exception as e:
        logger.warning(f"Failed to copy dataset file: {e}")
    
    # Create LLM client
    logger.info("Creating LLM client...")
    if args.mock or args.provider == "mock":
        llm_client = create_mock_llm_client(dataset)
    elif args.local:
        llm_client = create_local_llm_client(args.model)
    else:
        # Use provider-based adapter
        try:
            from llm_adapters import create_llm_client as create_adapter
            llm_client = create_adapter(
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
                api_base=args.api_base
            )
        except ImportError:
            # Fallback to original OpenAI client
            llm_client = create_llm_client(args.model, args.api_base, args.api_key or "EMPTY")
    
    # Load existing results if resuming
    existing_results = {}
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing_results = json.load(f)
        logger.info(f"Resuming from checkpoint: {len(existing_results)} tasks already done")
    
    # Create agent
    if args.mode == "end_to_end":
        agent = ReActAgent(
            llm_client=llm_client,
            model_name=args.model,
            tool_server_url=None if args.in_process else args.tool_server,
            toolmeta=toolmeta,
            max_turns=args.max_turns,
            audio_base_path=args.audio_base,
            in_process_tools=args.in_process
        )
        results = run_end_to_end(
            agent, dataset, args.audio_base, args.output, args.limit,
            checkpoint_interval=args.checkpoint,
            existing_results=existing_results,
            judge_client=_judge_client,
            judge_model=getattr(args, "judge_model", "gpt-4o-mini"),
            judge_provider=getattr(args, "judge_provider", "openai"),
        )
    else:
        agent = StepByStepAgent(
            llm_client=llm_client,
            toolmeta=toolmeta,
            max_turns=args.max_turns,
            audio_base_path=args.audio_base
        )
        results = run_step_by_step(
            agent, dataset, args.audio_base, args.output, args.limit,
            checkpoint_interval=args.checkpoint,
            existing_results=existing_results
        )
    
    # Print summary
    success_count = sum(1 for r in results.values() if r.get("success", False))
    logger.info(f"\n{'='*50}")
    logger.info(f"Inference complete!")
    logger.info(f"Total tasks: {len(results)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
