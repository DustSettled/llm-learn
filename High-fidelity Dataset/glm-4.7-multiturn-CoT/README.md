---
pretty_name: glm-4.7-multiturn-CoT
language:
- en
tags:
- synthetic
- distillation
- cot
- multi-turn
- conversation
task_categories:
- text-generation
- question-answering
size_categories:
- 1K<n<10K
---

# glm-4.7-multiturn-CoT

## Dataset Summary
`glm-4.7-multiturn-CoT` is a ShareGPT-style multi-turn reasoning distillation dataset generated with **GLM-4.7** as the teacher model.

This release focuses on preserving multi-turn dialogue continuity while injecting explicit chain-of-thought style responses in assistant turns.

## Key Features
- Multi-turn conversation format (`human` / `gpt`)
- Assistant responses stored as `<think>...</think>` + final answer
- Resume-safe distillation workflow (checkpoint/audit/reject tracking)
- Ready for SFT and reasoning-format alignment experiments

## Source and Curation
- Seed style: ChatAlpaca / ShareGPT-like multi-turn prompts
- Teacher model: `zai-glm-4.7`
- Output records: `3,725`

## Data Structure
Each line is a JSON object:

```json
{
  "id": "string",
  "generator": "glm-4.7",
  "conversation": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "<think>...</think>\n..."}
  ]
}
```

### Fields
- `id`: sample id
- `generator`: teacher identifier (`glm-4.7`)
- `conversation`: multi-turn dialogue array
- `conversation[].from`: role (`human` or `gpt`)
- `conversation[].value`: utterance text

## Intended Use
- Multi-turn instruction tuning
- CoT-aware assistant training
- Conversation robustness and context retention studies

## Limitations
- Reasoning traces may contain verbosity and teacher-specific style bias
- No independent factual verification per sample
- Not intended for direct deployment without downstream safety filtering

## Token and Cost Summary
Input tokens: 14,899,704  
Output tokens: 27,809,201  
Total tokens: 42,708,905  
Cost: $81.89 (Input $8.19 + Output $73.69)

## Project Status
Z.ai currently does not provide access to the stronger GLM-5 model, and the distillation pipeline is being migrated.
