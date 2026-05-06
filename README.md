# AudioToolBench

A benchmark for evaluating tool-augmented audio-language agents on audio authenticity and deepfake-detection tasks.

> **Insight**: Audio-language models alone are unreliable for forensic tasks. Augmenting them with specialist audio tools (VAD, ASR, deepfake detection, perceptual quality, language ID, etc.) yields more robust audio authenticity verification.

## Overview

The benchmark evaluates LLMs augmented with 24+ specialist audio tools using a ReAct-style agent.

- **Core agent**: `src/react_agent.py`
- **Entry point**: `src/run_inference.py`
- **Tool registry**: `data/audio_dataset/toolmeta.json`
- **Dataset**: 500 multi-turn ReAct queries in `data/audio_dataset/dataset_500.json`. Audio files are distributed separately (see "Data" below).

## Repository layout

```
src/                Core benchmark code (agent, inference, evaluation, judges)
scripts/            Pipeline runners and helper scripts
configs/            Tool metadata, system prompts, model inventory, eval configs
data/audio_dataset/ Dataset spec (dataset_500.json) and tool registry (toolmeta.json)
requirements.txt    Python dependencies
environment.yml     Conda environment spec
run.sh              Top-level convenience launcher
```

## Installation

```bash
pip install -r requirements.txt
./scripts/setup_isolated_envs.sh   # one isolated conda env per audio tool
```

## Data

The 500 audio files are not bundled in this repository. They are sourced from public corpora (InTheWild, ASVspoof2019 LA, ASVspoof2021 LA/DF, Codecfake) and redistributed via the companion dataset release. Each record in `dataset_500.json` references its audio by `id` (`audio/<id>.wav`).

To fetch the audio bundle:

```bash
# Set BENCHMARK_ROOT to where you want the audio files placed
export BENCHMARK_ROOT=/path/to/your/data
# Audio files are available from the companion dataset (anonymous mirror).
# Place wavs as: $BENCHMARK_ROOT/audio/<id>.wav
```

## Run end-to-end

```bash
export BENCHMARK_ROOT=/path/to/your/data
./scripts/run_full_pipeline.sh "<model-name>" "<inference-endpoint>" --in_process
```

This loads the dataset, dispatches each query through the ReAct agent, lets the agent invoke audio tools, and evaluates the final answers with both rule-based and LLM-judge scorers.

## Tools

| Category       | Tools                                      |
|----------------|--------------------------------------------|
| Perception     | `whisper`, `funasr`, `silero_vad`, `language_id` |
| Analysis       | `nisqa`, `speechmos`, `deepfake_audio`, `muq` |
| Transformation | `sepformer`, `demucs`, `deepfilternet`     |
| Detection      | `audioseal`, `chromaprint`, `speaker_verification` |
| Utility        | `calculator`                               |

See `configs/toolmeta_gpt_tools.json` and `data/audio_dataset/toolmeta.json` for the full registry, argument schemas, and expected outputs.

## Evaluation

Two judges are provided:
- `src/rule_based_judge.py` — whitelist/blacklist token matching against `groundtruth_answer`.
- `src/llm_judge_eval.py` — LLM-as-judge over the full ReAct trace.

Both can be invoked from `scripts/batch_eval.py`.

## License

Released for research use under CC-BY-4.0. Source audio retains the licenses of its upstream corpora; consult those licenses before redistribution.

## Citation

```bibtex
@misc{audiotoolbench_anon,
  title  = {AudioToolBench: A benchmark for tool-augmented audio-language agents},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```
