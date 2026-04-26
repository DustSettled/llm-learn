---
license: apache-2.0
language:
- en
tags:
- reasoning
- math
- distillation
- instruction-tuning
- chain-of-thought
- qwen
- qwen3.5
task_categories:
- question-answering
size_categories:
- n<1K
---

# Dataset Card (Qwen3.5-reasoning-700x)

## Dataset Summary

**Qwen3.5-reasoning-700x** is a high-quality distilled dataset.

This dataset uses the high-quality instructions constructed by `Alibaba-Superior-Reasoning-Stage2` as the seed question set. By calling the latest **Qwen3.5-27B full-parameter model** on the Alibaba Cloud DashScope platform as the teacher model, it generates high-quality responses featuring long-text reasoning processes (Chain-of-Thought). It covers several major domains: Mathematics | Code | Logic | Science | Instructions.

## Generation Parameters

- **Seed Data:** Alibaba-Superior-Reasoning-Stage2
- **Teacher Model:** `qwen3.5-27b` (Full-parameter model)
- **API Provider:** Alibaba Cloud DashScope
- **Context Window:** `16384` tokens.
- **Temperature:** `0.6`.


![Screenshot 2026-03-03 at 1.41.29 AM](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/8NOeeo2MHteW50v03AmVI.png)




## Dataset Structure

```json
{
  "id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
  "conversation": [
    {
      "from": "human",
      "value": "Seed question..."
    },
    {
      "from": "gpt",
      "value": "<think>\nComplete chain of thought process...\n</think>\n\nFinal answer..."
    }
  ],
  "input": "Original seed question...",
  "output": "<think>\nComplete chain of thought process...\n</think>\n\nFinal answer...",
  "domain": "Math | Code | Logic | Science | ...",
  "meta": {
    "training_stage": "stage2",
    "teacher_model": "Qwen3.5-27B"
  }
}
```

## Disclaimers

- **Language Model Hallucinations:** Although Qwen3.5-27B is an exceptionally powerful model and a low Temperature was set to maintain rigor, there may still be a very small number of calculation errors or logical fallacies in the generated data. It is highly recommended to conduct sample quality inspections before proceeding with fine-tuning.
- **License Compliance:** The open-sourcing and usage of this dataset must strictly comply with the open-source license agreement of the Qwen models, as well as the terms of service of Alibaba Cloud DashScope.