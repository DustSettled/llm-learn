---
pretty_name: glm-4.7-Superior-Reasoning-stage1
language:
- en
tags:
- synthetic
- distillation
- cot
- reasoning
- math
- stage1
task_categories:
- text-generation
- question-answering
size_categories:
- 1K<n<10K
---

# glm-4.7-Superior-Reasoning-stage1

## Dataset Summary
`glm-4.7-Superior-Reasoning-stage1` is a Stage1 reasoning distillation dataset built from the **Alibaba Superior-Reasoning** style pipeline, with a stronger teacher model replacement.

Compared with the original upstream setup, this release uses **GLM-4.7** as teacher for higher-quality reasoning traces.

## Stage1 Distillation Setup (Low Temperature)
- **Training stage:** `stage1`
- **Sampling temperature:** **`0.6`** (low-temperature distillation)
- `top_p`: `0.95`
- Domain focus: `math`
- Output records: `1,192`

## Method Note (Based on Alibaba Superior-Reasoning)
This card follows the Stage1-style reasoning distillation direction from Alibaba Superior-Reasoning (including the staged reasoning data construction perspective), while replacing the teacher with a stronger model (`glm-4.7`) in this run.

## Data Structure
Each line is a JSON object:

```json
{
  "id": "string",
  "conversation": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "<think>...</think>\n..."}
  ],
  "input": "...",
  "output": "<think>...</think>\n...",
  "domain": "math",
  "meta": {
    "training_stage": "stage1",
    "sampling_temperature": 0.6,
    "teacher_model": "glm-4.7"
  }
}
```

### Fields
- `id`: normalized question hash id
- `conversation`: single-turn conversation packaged in chat format
- `input`: user question text
- `output`: assistant reasoning + answer text
- `domain`: task/domain label (mainly `math`)
- `meta.training_stage`: stage tag (`stage1`)
- `meta.sampling_temperature`: distillation temperature (`0.6`)
- `meta.teacher_model`: teacher id (`glm-4.7`)

## Intended Use
- Stage1 reasoning SFT
- Math CoT behavior tuning
- Teacher-model swap ablation studies

## Limitations
- This release is distillation data, not ground-truth proof data
- Reasoning style inherits teacher preferences
- Upstream licensing and usage constraints of source datasets must be respected

## Token and Cost Summary
Input tokens: 131,883  
Output tokens: 22,608,190  
Total tokens: 22,740,073  
Cost: $59.98 (Input $0.07 + Output $59.91)

## Project Status
Z.ai currently does not provide access to the stronger GLM-5 model, and the distillation pipeline is being migrated.

## References
- Alibaba-NLP. *Alibaba-Superior-Reasoning* (dataset card): https://huggingface.co/datasets/Alibaba-NLP/Alibaba-Superior-Reasoning
- Li et al. *DASD: Deep Analyze Soon Deliver for Efficient Reinforced Reasoning*: https://arxiv.org/abs/2504.13994
- Wang et al. *SuperiorReasoning: Towards Large-Scale Inference-Time Scaling for Long-CoT Reasoning*: https://arxiv.org/abs/2504.07955
