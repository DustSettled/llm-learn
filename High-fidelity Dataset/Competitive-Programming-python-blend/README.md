---
pretty_name: Competitive-Programming-python-blend
language:
- en
license:
- apache-2.0
- cc-by-4.0
- odc-by
- mit
- bsd-2-clause
- bsd-3-clause
tags:
- code
- competitive-programming
- synthetic
- reasoning
- sharegpt
- sft
task_categories:
- text-generation
size_categories:
- 10K<n<100K
---

# Dataset Card for Competitive-Programming-python-blend

## Summary

`Competitive-Programming-python-blend` is a mixed supervised fine-tuning dataset centered on competitive programming, code reasoning, and instruction-style problem solving. The blend is Python-first, but it also keeps a small amount of C++, agentless SWE, and reasoning-oriented chat supervision to broaden training coverage.

The current release is published as a single HF-friendly JSONL file, `clean.jsonl`.

## Blend Composition

The blend follows the source proportions below.

| Source | Role in the blend | Share |
| --- | --- | ---: |
| [nohurry/Opus-4.6-Reasoning-3000x-filtered](https://huggingface.co/datasets/nohurry/Opus-4.6-Reasoning-3000x-filtered) | Reasoning-heavy synthetic SFT data | 5.83% |
| [Jackrong/Qwen3.5-reasoning-700x](https://huggingface.co/datasets/Jackrong/Qwen3.5-reasoning-700x) | Distilled reasoning and instruction-following data | 1.58% |
| [nvidia/Nemotron-SFT-Competitive-Programming-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2), `competitive_coding_python` | Primary Python competitive-programming supervision | 87.54% |
| [nvidia/Nemotron-SFT-Competitive-Programming-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2), `competitive_coding_cpp` | Small cross-language competitive-programming supplement | 2.50% |
| [nvidia/Nemotron-SFT-SWE-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2), `agentless` | Lightweight agentless SWE-style supervision | 0.05% |
| [nvidia/Nemotron-SFT-Instruction-Following-Chat-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2), `reasoning_on` | Small reasoning-oriented chat supplement | 2.50% |

Percentages are computed from the blend recipe and sum to 100%.

## Data Format

Each line in `clean.jsonl` is one JSON object with a `messages` field. The current release is stored in a format that can be loaded directly with `datasets.load_dataset("json", ...)`:

```json
{
  "id": "e3f7b0d4f8fbb2f33771b2d8f0cbecab6d5e3f1b85f58fca4d3fbf5ce7d8f98b",
  "messages": [
    {"role": "user", "content": "prompt"},
    {"role": "assistant", "content": "<think>..."}
  ]
}
```

All upstream records were normalized into this unified schema. Source-specific fields were flattened into text turns, speaker names were standardized into `user` / `assistant` style roles, and each sample carries a content-derived SHA-256 `id` string.


### Source-by-source notes

- [nohurry/Opus-4.6-Reasoning-3000x-filtered](https://huggingface.co/datasets/nohurry/Opus-4.6-Reasoning-3000x-filtered): upstream card declares `apache-2.0`. No dedicated citation block is provided on the card.
- [Jackrong/Qwen3.5-reasoning-700x](https://huggingface.co/datasets/Jackrong/Qwen3.5-reasoning-700x): upstream card declares `apache-2.0`. The card also notes that usage should comply with the Qwen open-source license agreement and Alibaba Cloud DashScope terms. No dedicated citation block is provided on the card.
- [nvidia/Nemotron-SFT-Competitive-Programming-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2): the upstream card used by both `competitive_coding_python` and `competitive_coding_cpp` lists `cc-by-4.0`, `odc-by`, and additional `mit` notice. The card states that the dataset is ready for commercial use. No dedicated citation block is provided on the card.
- [nvidia/Nemotron-SFT-SWE-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2): the upstream card used by the `agentless` subset lists `cc-by-4.0` with additional `apache-2.0`, `mit`, `bsd-3-clause`, and `bsd-2-clause` notices. No dedicated citation block is provided on the card.
- [nvidia/Nemotron-SFT-Instruction-Following-Chat-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2): the upstream card used by the `reasoning_on` subset lists `odc-by`. No dedicated citation block is provided on the card.

## Intended Use

This dataset is intended for supervised fine-tuning or continued instruction tuning of code-capable models, especially models targeting Python competitive programming and code reasoning. The auxiliary C++, SWE, and reasoning-chat slices are included to improve coverage rather than to define the core distribution.

## Limitations

This is a mixed, processed, and partially synthetic dataset. It may inherit model-generated artifacts, reasoning mistakes, formatting noise, and licensing constraints from the upstream sources. It is better suited for training than for source-pure evaluation.
