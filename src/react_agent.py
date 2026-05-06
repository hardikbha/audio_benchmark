#!/usr/bin/env python3
"""
ReAct Agent for Audio Tool Benchmark
Implements the ReAct (Reasoning + Acting) loop for tool-augmented audio LLMs.

Based on GTA benchmark architecture.
"""

import json
import logging
import os
import re
import time
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ReActStep:
    """A single step in the ReAct trace."""
    step_num: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    is_final: bool = False
    final_answer: Optional[str] = None


@dataclass
class ReActTrace:
    """Complete ReAct execution trace."""
    question: str
    audio_files: List[str]
    steps: List[ReActStep] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    llm_time_sec: float = 0.0
    tool_time_sec: float = 0.0
    total_time_sec: float = 0.0
    per_turn_stats: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "audio_files": self.audio_files,
            "steps": [
                {
                    "step_num": s.step_num,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                    "is_final": s.is_final,
                    "final_answer": s.final_answer
                }
                for s in self.steps
            ],
            "tools_called": self.tools_called,
            "llm_calls": self.llm_calls,
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_completion_tokens": self.llm_completion_tokens,
            "llm_total_tokens": self.llm_total_tokens,
            "llm_time_sec": self.llm_time_sec,
            "tool_time_sec": self.tool_time_sec,
            "total_time_sec": self.total_time_sec,
            "per_turn_stats": self.per_turn_stats,
            "final_answer": self.final_answer,
            "success": self.success,
            "error": self.error
        }


class ReActAgent:
    """
    ReAct agent that uses audio tools to answer questions.
    
    Supports two modes:
    - Live execution: Calls actual tool server
    - Step-by-step: Uses toolmeta for simulation
    """
    
    SYSTEM_PROMPT = """You are an audio analysis agent that MUST use tools. You CANNOT hear audio directly.

## CRITICAL RULES (VIOLATIONS = FAILURE)
1. **YOU MUST USE TOOLS** - You cannot analyze audio directly.
2. **NEVER GUESS OR HALLUCINATE** - Any claim about audio content (words, speakers, quality, deepfake status) without tool output is WRONG.
3. **ALWAYS CALL AT LEAST ONE TOOL** - You must call a tool before giving any Final Answer.
4. **POINT-TO-AUDIO** - When given a specific audio file path, use it directly in the tool arguments.

## TASK → TOOL MAPPING (use the MOST specific tool)
| Category | Tools | Usage |
|----------|-------|-------|
| **Transcription** | `whisper`, `funasr` | Convert speech to text |
| **Description** | `audio_caption`, `desta25` | Captioning/Content analysis |
| **QA** | `r1_aqa` | Ask questions about audio content |
| **VAD/Diarization** | `silero_vad`, `diarizen` | Voice detection / Speaker segmentation |
| **Deepfake/Security** | `deepfake_audio`, `audioseal` | Detect synthetic audio / Watermarks |
| **Quality Score** | `nisqa`, `speechmos` | Perceptual MOS score (1-5) |
| **Quality Metrics** | `wav2vec2_quality`, `audioldm_eval` | Artifacts / FAD / Reference metrics |
| **Enhancement** | `deepfilternet`, `sb_sgmse`, `espnet_enhance` | Denoising / Improve speech quality |
| **Separation** | `demucs` (music), `sepformer_wham` (speech) | Separate stems or speakers |
| **Identification** | `language_id`, `gender_detection`, `chromaprint` | Language / Gender / Fingerprinting |
| **Verification** | `speaker_verification` | Compare two voices (needs 2 files) |
| **Embeddings** | `resemblyzer`, `clap_embed` | Speaker / Semantic embeddings |
| **Search** | `google_search` | External knowledge ONLY |

## Available Tools
{tool_descriptions}

## EXACT REASONING FORMAT
**Step 1: Plan and Call Tool**
```
Thought: The user wants to [task]. I cannot [task] directly, so I will use [tool_name].
Action: tool_name
Action Input: {{"audio_path": "precise_path_from_user"}}
```

**Step 2: Analyze Observation**
```
Thought: The tool [tool_name] returned [summary of output]. This tells me [insight]. I [need more info / have the answer].
Action: [next_tool / None]
...
```

**Step 3: Final Answer**
```
Thought: I have sufficient information from the tools.
Final Answer: [Answer based ONLY on tool observations]
```

## EXAMPLES

**Example 1: Point-to-Audio Transcription**
User: "What is said in this file?"
Audio: data/audio/test.wav

CORRECT Response:
```
Thought: The user wants to know the content of the audio. I cannot hear it, so I must use the whisper tool to transcribe it.
Action: whisper
Action Input: {{"audio_path": "data/audio/test.wav"}}
```

**Example 2: Deepfake Detection**
User: "Is this audio real or fake?"
Audio: data/samples/record_01.mp3

CORRECT Response:
```
Thought: I need to detect if the audio is a deepfake. I will use the deepfake_audio tool.
Action: deepfake_audio
Action Input: {{"audio_path": "data/samples/record_01.mp3"}}
```

**Example 3: Enhancement (Denoising)**
User: "Clean up this noisy recording."
Audio: data/noisy.wav

CORRECT Response:
```
Thought: The user wants to remove noise. I should use an enhancement tool like deepfilternet, sb_sgmse, or espnet_enhance. I'll use deepfilternet.
Action: deepfilternet
Action Input: {{"audio_path": "data/noisy.wav"}}
```

**Example 4: Hallucination Prevention (DO NOT DO THIS)**
User: "Is the speaker angry?"
Audio: data/emotion.wav

WRONG Response:
```
Thought: The speaker sounds angry.
Final Answer: Yes, the speaker is angry.
```
**Reason for Failure:** You guessed the emotion without using an emotion detection tool.

## REMINDER
- START WITH A TOOL CALL.
- If you output "Final Answer" immediately, you FAIL.
- Use the exact audio path provided.
"""

    def _resolve_system_prompt_template(self) -> str:
        """Resolve system prompt template with env override support."""
        prompt_file = os.getenv("REACT_SYSTEM_PROMPT_FILE", "").strip()
        prompt_inline = os.getenv("REACT_SYSTEM_PROMPT", "").strip()

        if prompt_file:
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
                if prompt:
                    logger.info("Using system prompt override from file: %s", prompt_file)
                    return prompt
                logger.warning("System prompt file is empty: %s. Falling back.", prompt_file)
            except Exception as e:
                logger.warning("Failed to read REACT_SYSTEM_PROMPT_FILE=%s (%s). Falling back.", prompt_file, e)

        if prompt_inline:
            logger.info("Using inline system prompt override from REACT_SYSTEM_PROMPT")
            return prompt_inline

        return self.SYSTEM_PROMPT

    def __init__(
        self,
        llm_client,
        model_name: str = "default-model",
        tool_server_url: Optional[str] = None,
        toolmeta: Optional[Dict] = None,
        max_turns: int = 10,
        audio_base_path: str = "",
        in_process_tools: bool = False,
        preload_tools: Optional[List[str]] = None
    ):
        """
        Initialize ReAct agent.
        
        Args:
            llm_client: OpenAI-compatible client for LLM calls
            model_name: Name of the model to use (required by OpenAI client)
            tool_server_url: URL of tool server (for live HTTP execution)
            toolmeta: Tool metadata dict (for step-by-step mode)
            max_turns: Maximum number of turns before stopping
            audio_base_path: Base path for resolving audio file paths
            in_process_tools: If True, load and execute tools in-process (no HTTP server)
            preload_tools: List of tool names to preload (for in_process mode)
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.tool_server_url = tool_server_url
        self.toolmeta = toolmeta or {}
        self.max_turns = max_turns
        self.audio_base_path = audio_base_path
        self.in_process_tools = in_process_tools
        self.system_prompt_template = self._resolve_system_prompt_template()
        
        # Tool instances (lazy loaded)
        self._loaded_tools: Dict[str, Any] = {}
        
        # Determine mode
        if in_process_tools:
            self.live_mode = True
            logger.info("Using in-process tool execution (no HTTP server required)")
            if preload_tools:
                for tool_name in preload_tools:
                    self._get_or_load_tool(tool_name)
        else:
            self.live_mode = tool_server_url is not None
    
    def _get_or_load_tool(self, tool_name: str):
        """Get or lazily load a tool instance."""
        if tool_name in self._loaded_tools:
            return self._loaded_tools[tool_name]
        
        # Tool registry mapping names to (module, class) tuples
        TOOL_REGISTRY = {
            "whisper": ("audiotoolagent.tools.whisper", "WhisperTool"),
            "deepfake_audio": ("audiotoolagent.tools.deepfake_audio", "DeepfakeAudioTool"),
            "nisqa": ("audiotoolagent.tools.nisqa", "NisqaTool"),
            "silero_vad": ("audiotoolagent.tools.silero_vad", "SileroVADTool"),
            "language_id": ("audiotoolagent.tools.language_id", "LanguageIdentificationTool"),
            "speaker_verification": ("audiotoolagent.tools.speaker_verification", "SpeakerVerificationTool"),
            "gender_detection": ("audiotoolagent.tools.gender_detection", "GenderDetectionTool"),
            "demucs": ("audiotoolagent.tools.demucs", "DemucsTool"),
            "deepfilternet": ("audiotoolagent.tools.deepfilternet", "DeepFilterNetTool"),
            "sepformer_wham": ("audiotoolagent.tools.sepformer_wham", "SepFormerWHAMTool"),
            "diarizen": ("audiotoolagent.tools.diarizen", "DiarizenTool"),
            "chromaprint": ("audiotoolagent.tools.chromaprint", "ChromaprintTool"),
            "audioseal": ("audiotoolagent.tools.audioseal_tool", "AudioSealTool"),
            "funasr": ("audiotoolagent.tools.funasr_tool", "FunASRTool"),
            "sgmse": ("audiotoolagent.tools.sgmse_tool", "SGMSETool"),
            "speechmos": ("audiotoolagent.tools.speechmos", "SpeechMOSTool"),
            "audio_caption": ("audiotoolagent.tools.audio_caption", "AudioCaptionTool"),
            "clap_embed": ("audiotoolagent.tools.clap_embed", "CLAPEmbedTool"),
            "resemblyzer": ("audiotoolagent.tools.resemblyzer_tool", "ResemblyzerTool"),
            "conette": ("audiotoolagent.tools.conette_tool", "CoNeTTEModelTool"),
            # Tools added for full 24-tool benchmark coverage
            "wav2vec2_quality": ("audiotoolagent.tools.wav2vec2_quality", "Wav2Vec2QualityTool"),
            "espnet_enhance": ("audiotoolagent.tools.espnet_enhance", "ESPnetEnhanceTool"),
            "sb_sgmse": ("audiotoolagent.tools.sb_sgmse", "SBSGMSETool"),
            "r1_aqa": ("audiotoolagent.tools.r1_aqa_tool", "R1AQATool"),
            "audioldm_eval": ("audiotoolagent.tools.audioldm_eval", "AudioLDMEvalTool"),
            "desta25": ("audiotoolagent.tools.desta25", "Desta25Tool"),
            "pyannote_segmentation": ("audiotoolagent.tools.pyannote_segmentation", "PyAnnoteSegmentationTool"),
            "asteroid_separate": ("audiotoolagent.tools.asteroid_separate", "AsteroidSeparateTool"),
        }
        
        # Aliases for common LLM misnaming
        TOOL_ALIASES = {
            "diarizer": "diarizen",
            "speech_activity": "silero_vad",
            "clapembed": "clap_embed",
            "clap": "clap_embed",
            "res": "resemblyzer",
            "enhance": "espnet_enhance",
            "noise_reduction": "deepfilternet",
            "asr": "whisper",
            "transcribe": "whisper",
            "vad": "silero_vad",
            "speaker_diarization": "diarizen",
            "sgmse_enhance": "sb_sgmse",
            "audio_quality": "nisqa",
        }
        tool_name = TOOL_ALIASES.get(tool_name, tool_name)
        
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Unknown tool: {tool_name}. Available: {list(TOOL_REGISTRY.keys())}")
        
        module_path, class_name = TOOL_REGISTRY[tool_name]
        
        try:
            import importlib
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)
            tool_instance = tool_class()
            self._loaded_tools[tool_name] = tool_instance
            logger.info(f"Loaded tool: {tool_name}")
            return tool_instance
        except Exception as e:
            logger.error(f"Failed to load tool {tool_name}: {e}")
            raise
    
    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for system prompt."""
        descriptions = []
        for name, meta in self.toolmeta.items():
            desc = f"- {name}: {meta.get('description', 'No description')}"
            inputs = meta.get('inputs', [])
            if inputs:
                params = ", ".join(
                    f"{inp['name']}({'required' if not inp.get('optional', True) else 'optional'})"
                    for inp in inputs
                )
                desc += f"\n  Parameters: {params}"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def _render_system_prompt(self) -> str:
        """Render prompt template while safely injecting tool descriptions."""
        return self.system_prompt_template.replace(
            "{tool_descriptions}",
            self._format_tool_descriptions()
        )
    
    def _parse_llm_response(self, response: str) -> Tuple[str, Optional[str], Optional[Dict], Optional[str]]:
        """
        Parse LLM response to extract Thought, Action, Action Input, or Final Answer.
        
        Returns:
            (thought, action, action_input, final_answer)
        """
        def _extract_first_json_object(text: str, start_at: int = 0) -> Optional[str]:
            """Extract first balanced JSON object from text starting at index."""
            if not text:
                return None
            start = text.find("{", max(0, start_at))
            while start != -1:
                depth = 0
                in_str = False
                esc = False
                for i in range(start, len(text)):
                    ch = text[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                        continue
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return text[start:i + 1]
                start = text.find("{", start + 1)
            return None

        response = response or ""
        thought = ""
        action = None
        action_input = None
        final_answer = None
        
        # Extract Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            # Harmony-style fallback: `analysis ... assistantcommentary ...`
            analysis_match = re.search(
                r'analysis\s*(.+?)(?=assistantcommentary|assistantfinal|$)',
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if analysis_match:
                thought = analysis_match.group(1).strip()

        # Final answer parsing (ReAct + Harmony styles)
        final_match = re.search(r'Final Answer:\s*(.+?)$', response, re.DOTALL | re.IGNORECASE)
        if final_match:
            final_answer = final_match.group(1).strip()
            return thought, None, None, final_answer
        harmony_final_match = re.search(r'assistantfinal\s*:?\s*(.+?)$', response, re.DOTALL | re.IGNORECASE)
        if harmony_final_match:
            final_answer = harmony_final_match.group(1).strip()
            return thought, None, None, final_answer

        # PRIORITIZE ACTION: Check for Action if no Final Answer found
        action_match = re.search(r'Action:\s*(\w+)', response, re.IGNORECASE)
        if action_match:
            action_val = action_match.group(1).strip()
            
            # Handle common hallucination: "Action: final_answer"
            if action_val.lower() in ["final_answer", "none", "null"]:
                # Try to find what it meant as the answer
                ans_match = re.search(r'Action Input:\s*(\{.+?\})', response, re.DOTALL | re.IGNORECASE)
                if ans_match:
                    try:
                        ans_data = json.loads(ans_match.group(1))
                        # If it put the answer in a dict, take the first value or the string
                        if isinstance(ans_data, dict):
                            final_answer = next(iter(ans_data.values()))
                        else:
                            final_answer = str(ans_data)
                    except:
                        final_answer = ans_match.group(1)
                else:
                    # Fallback: take whatever is after Action Input or just the rest of the text
                    final_answer = response.split("Action Input:")[-1].strip()
                
                return thought, None, None, final_answer

            action = action_val
            
            # Extract Action Input
            input_match = re.search(r'Action Input:\s*(\{.+?\})', response, re.DOTALL | re.IGNORECASE)
            if input_match:
                try:
                    action_input = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    # Try to fix common issues
                    raw = input_match.group(1)
                    raw = re.sub(r"'", '"', raw)  # Single to double quotes
                    try:
                        action_input = json.loads(raw)
                    except:
                        action_input = {"raw": raw}
            
            # If we found an action, return it
            return thought, action, action_input, None

        # Harmony-style tool call:
        # `assistantcommentary to=<tool_name> json{...}`
        harmony_action_match = re.search(
            r'assistantcommentary\s+to=([A-Za-z0-9_\-]+)(?:\s+json)?',
            response,
            re.IGNORECASE,
        )
        if harmony_action_match:
            action = harmony_action_match.group(1).strip()
            raw_json = _extract_first_json_object(response, harmony_action_match.end())
            if raw_json:
                try:
                    action_input = json.loads(raw_json)
                except json.JSONDecodeError:
                    repaired = re.sub(r"'", '"', raw_json)
                    try:
                        action_input = json.loads(repaired)
                    except Exception:
                        action_input = {"raw": raw_json}
            return thought, action, action_input, None
            
        return thought, None, None, None
    
    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool and return the observation."""
        if not self.live_mode:
            # Step-by-step mode: return placeholder observation
            return f"[Tool {tool_name} executed with args {args}. Output simulated.]"

        # Optional mode: emulate every tool using the same orchestrator LLM.
        # This keeps the ReAct/tool-calling flow intact while removing specialized tool deps.
        if os.getenv("REACT_EMULATE_TOOLS_WITH_LLM", "0") == "1":
            try:
                tool_prompt = (
                    f"You are emulating the audio tool '{tool_name}'.\n"
                    f"Arguments: {json.dumps(args, ensure_ascii=False)}\n\n"
                    "Return ONLY a compact JSON object representing the tool observation. "
                    "Do not include markdown."
                )
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a deterministic tool emulator. Return structured JSON only.",
                        },
                        {"role": "user", "content": tool_prompt},
                    ],
                    max_tokens=512,
                    temperature=0.0,
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    return json.dumps(
                        {"tool": tool_name, "status": "empty_observation", "args": args},
                        ensure_ascii=False,
                    )
                return content
            except Exception as e:
                logger.error(f"LLM-tool emulation failed for {tool_name}: {e}")
                return f"Tool execution error: {str(e)}"
        
        # SANITIZE PATHS: Fix common LLM path errors
        # Logic: 
        # 1. Duplication (base/base/file) -> strip one base
        # 2. Already has base -> leave alone
        # 3. Filename only -> prepend base
        
        base_path = "data/audio_dataset/audio_assets/"
        
        # Iterate through args to find file paths
        for key, value in args.items():
            if isinstance(value, str) and (value.endswith('.wav') or value.endswith('.mp3') or value.endswith('.flac')):
                # Case 1: Duplicated path (e.g. data/.../data/.../file.wav)
                duplicated_pattern = f"{base_path}{base_path}"
                if duplicated_pattern in value:
                    args[key] = value.replace(duplicated_pattern, base_path)
                    logger.info(f"Sanitized path (duplication): {value} -> {args[key]}")
                
                # Case 2: Already has base path (e.g. data/.../file.wav)
                elif value.startswith(base_path):
                    # Do nothing
                    pass
                
                # Case 3: Keep any existing relative path as-is.
                elif os.path.exists(value):
                    pass

                # Case 4: Filename only (e.g. file.wav) - IF NOT an absolute path or output path
                elif not value.startswith("/") and not value.startswith("outputs/"):
                    args[key] = f"{base_path}{value}"
                    logger.info(f"Sanitized path (prepend base): {value} -> {args[key]}")

        # Live mode: call tool server OR execute in-process
        if self.in_process_tools:
            # In-process execution (no HTTP server needed)
            try:
                tool_instance = self._get_or_load_tool(tool_name)
                # AudioToolAgent tools use call(params) - params can be dict or JSON string
                result = tool_instance.call(args)
                # result is already JSON string from AudioToolAgent tools
                if isinstance(result, str):
                    return result
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                logger.error(f"In-process tool execution failed: {e}")
                return f"Tool execution error: {str(e)}"
        else:
            # HTTP server execution
            try:
                url = f"{self.tool_server_url}/tool/{tool_name}"
                response = requests.post(url, json=args, timeout=120)
                result = response.json()
                
                if result.get("success"):
                    return json.dumps(result.get("result", {}), indent=2)
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"
            except Exception as e:
                return f"Tool execution error: {str(e)}"
    
    @staticmethod
    def _normalize_usage(usage: Any) -> Dict[str, int]:
        """Normalize usage object from different SDKs to token counts."""
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

        # OpenAI-style object (pydantic/dataclass)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _call_llm(self, messages: List[Dict]) -> Tuple[str, Dict[str, int], float]:
        """Call the LLM and return (response_text, usage, latency_sec)."""
        start_time = time.time()
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.1
            )
            latency_sec = time.time() - start_time
            usage = self._normalize_usage(getattr(response, "usage", None))
            return response.choices[0].message.content, usage, latency_sec
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def solve(
        self,
        question: str,
        audio_files: List[str],
        ground_truth_steps: Optional[List[Dict]] = None
    ) -> ReActTrace:
        """
        Execute the ReAct loop to solve a question.
        
        Args:
            question: The user's question about the audio
            audio_files: List of audio file paths
            ground_truth_steps: Optional GT steps for step-by-step mode
        
        Returns:
            ReActTrace with complete execution history
        """
        query_start_time = time.time()
        trace = ReActTrace(question=question, audio_files=audio_files)
        
        # Build system prompt
        system_prompt = self._render_system_prompt()
        
        # Build initial user message
        user_content = f"Question: {question}\n\nAudio files: {', '.join(audio_files)}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # ReAct loop
        for turn in range(self.max_turns):
            try:
                # Get LLM response
                llm_response, usage, llm_latency_sec = self._call_llm(messages)
                trace.llm_calls += 1
                trace.llm_prompt_tokens += usage.get("prompt_tokens", 0)
                trace.llm_completion_tokens += usage.get("completion_tokens", 0)
                trace.llm_total_tokens += usage.get("total_tokens", 0)
                trace.llm_time_sec += llm_latency_sec
                
                # DEBUG: Print raw LLM response
                logger.info("="*60)
                logger.info(f"RAW LLM RESPONSE (Turn {turn + 1}):")
                logger.info("-"*60)
                logger.info(llm_response)
                logger.info("="*60)
                
                # Parse response
                thought, action, action_input, final_answer = self._parse_llm_response(llm_response)
                
                # Create step
                step = ReActStep(
                    step_num=turn + 1,
                    thought=thought,
                    action=action,
                    action_input=action_input
                )
                
                # Check if final answer
                if final_answer:
                    step.is_final = True
                    step.final_answer = final_answer
                    trace.steps.append(step)
                    trace.per_turn_stats.append(
                        {
                            "turn": turn + 1,
                            "llm_latency_sec": round(llm_latency_sec, 4),
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                            "action": action,
                            "has_final_answer": True,
                        }
                    )
                    trace.final_answer = final_answer
                    trace.success = True
                    break
                
                # Execute action if present
                if action:
                    if action_input is None:
                        action_input = {}
                    if not isinstance(action_input, dict):
                        action_input = {"raw": str(action_input)}

                    if os.getenv("REACT_EMULATE_TOOLS_WITH_LLM", "0") == "1":
                        action_input.setdefault("_user_question", question)
                        action_input.setdefault("_turn", turn + 1)

                    # CRITICAL: Override LLM's audio_path with true path from dataset
                    # LLMs frequently hallucinate file paths (e.g., changing underscores to dots)
                    if "audio_path" in action_input and audio_files:
                        llm_path = action_input["audio_path"]
                        # Use the first audio file from the dataset as ground truth
                        true_path = audio_files[0]
                        if llm_path != true_path:
                            logger.info(f"PATH OVERRIDE: LLM gave '{llm_path}', using true path '{true_path}'")
                            action_input["audio_path"] = true_path
                    elif "audio_path" not in action_input and audio_files:
                        # LLM forgot to include audio_path - inject it
                        action_input["audio_path"] = audio_files[0]
                        logger.info(f"PATH INJECT: Added missing audio_path '{audio_files[0]}'")
                    
                    tool_start_time = time.time()
                    observation = self._execute_tool(action, action_input)
                    trace.tool_time_sec += time.time() - tool_start_time
                    step.observation = observation
                    trace.tools_called.append(action)
                    
                    # Add to messages for next turn
                    messages.append({"role": "assistant", "content": llm_response})
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                
                trace.steps.append(step)
                trace.per_turn_stats.append(
                    {
                        "turn": turn + 1,
                        "llm_latency_sec": round(llm_latency_sec, 4),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "action": action,
                        "has_final_answer": bool(final_answer),
                    }
                )
                
            except Exception as e:
                trace.error = str(e)
                logger.error(f"ReAct loop error at turn {turn}: {e}")
                break
        
        if not trace.final_answer:
            trace.error = trace.error or "Max turns reached without final answer"
        trace.total_time_sec = round(time.time() - query_start_time, 4)
        trace.llm_time_sec = round(trace.llm_time_sec, 4)
        trace.tool_time_sec = round(trace.tool_time_sec, 4)
        
        return trace


class StepByStepAgent(ReActAgent):
    """
    Agent for step-by-step evaluation mode.
    Given first N steps as context, predicts step N+1.
    """
    
    def predict_next_step(
        self,
        question: str,
        audio_files: List[str],
        previous_steps: List[Dict]
    ) -> Dict:
        """
        Predict the next step given previous steps.
        
        Args:
            question: The original question
            audio_files: Audio files
            previous_steps: List of previous steps with tool, args, output
        
        Returns:
            Predicted next step
        """
        # Build context from previous steps
        context_parts = [f"Question: {question}", f"Audio files: {', '.join(audio_files)}", ""]
        
        for i, step in enumerate(previous_steps):
            context_parts.append(f"Step {i+1}:")
            context_parts.append(f"Thought: {step.get('thought', 'Analyzing...')}")
            context_parts.append(f"Action: {step.get('tool', step.get('action', ''))}")
            context_parts.append(f"Action Input: {json.dumps(step.get('args', step.get('action_input', {})))}")
            context_parts.append(f"Observation: {step.get('output', step.get('observation', ''))}")
            context_parts.append("")
        
        context_parts.append(f"Step {len(previous_steps) + 1}:")
        context_parts.append("Thought:")
        
        context = "\n".join(context_parts)
        
        system_prompt = self._render_system_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        
        llm_response, _, _ = self._call_llm(messages)
        thought, action, action_input, final_answer = self._parse_llm_response(llm_response)
        
        return {
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "final_answer": final_answer,
            "raw_response": llm_response
        }
