---
pretty_name: LogosForge Scored SFT (Natural Reasoning)
license: apache-2.0
language:
  - en
task_categories:
  - text-generation
  - question-answering
  - text-classification
tags:
  - reasoning
  - chain-of-thought
  - sft
  - distillation
  - grading
  - curriculum-learning
  - natural-reasoning
size_categories:
  - 10K<n<100K
---


# LogosForge-scored-sft-v1

This dataset is a **scored Supervised Fine-Tuning (SFT) distillation corpus** built on top of the *Natural Reasoning* question set.  
It is constructed in **two stages**:

First, a large-scale reasoning-oriented teacher model (**gpt-oss-120B-high**) is used to generate distilled student responses, including explicit chain-of-thought reasoning, for natural reasoning questions.

Second, these distilled responses are evaluated by a **separate instruction-following model** (`qwen-3-235b-a22b-instruct-2507`) acting as an automated grader, which assigns fine-grained quality scores based on final answer correctness, reasoning quality, and (when available) reference answer similarity.

Each sample therefore consists of a two-turn human–assistant conversation augmented with **structured, model-generated evaluation signals**, enabling quality-aware filtering, ranking, and curriculum construction for reasoning-focused SFT.

![Gemini_Generated_Image_7l2uh47l2uh47l2u](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/rAduwzjY0PSCRTW74bIbq.png)

---

## Instruction Model Used for Scoring

All samples are evaluated using a **single, fixed instruction-following model** acting purely as an automated grader:

- **Model**: `qwen-3-235b-a22b-instruct-2507`
- **Interface**: Cerebras Chat Completions (OpenAI-compatible API, streaming)
- **Decoding setup**: low temperature (0.2) with constrained sampling to emphasize determinism and scoring consistency

This model is **not used to generate training data**. Instead, it evaluates each sample by reading:

- the original instruction,
- the student’s chain-of-thought (if provided),
- the student’s final answer,
- and an optional reference answer.

The grader is prompted as a **rigorous academic evaluator** and is explicitly instructed **not to introduce new reasoning or corrections**, but to judge strictly based on the given content.

---

## Dataset Specifications

| Item | Value |
| --- | --- |
| Dataset Name | LogosForge-scored-sft-v1 |
| Base Dataset | facebook/natural_reasoning |
| Teacher Model | gpt-oss-120B-high |
| Language | English |
| Data Format | JSONL for SFT conversations + scoring fields |
| Samples | 63852 |

## Curation & Distillation Notes

- All records include a structured conversation with `human` and `gpt` turns.
- Automated scoring metrics include `model_answer_score`, `CoT_score`, `average_score`, and `reference_answer_similarity`.
- A categorical `judgment` label ☕️ (Venti / Grande / Tall / No) summarizes quality tiers.
- `reference_answer` is included when available for similarity computation and auditing.
- Assistant answers may include chain-of-thought blocks marked with <think>...</think>.

---

## Scoring Dimensions

Each sample is evaluated along three orthogonal dimensions:

### 1. Final Answer Correctness (`model_answer_score`, 1–10)

Measures factual accuracy, logical consistency, and faithfulness to the original instruction.  
This score reflects whether the **final answer itself** is correct, regardless of reasoning verbosity.

### 2. Chain-of-Thought Quality (`CoT_score`, 1–10)

Evaluates the clarity, coherence, and correctness of the provided reasoning steps.  
Missing, shallow, or logically flawed chains of thought receive lower scores even if the final answer is correct.

### 3. Reference Answer Similarity (`reference_answer_similarity`, 0–100 or `null`)

When a reference answer is available, semantic similarity is computed based on **key idea overlap and meaning**, not exact string matching.

If no reference answer exists, this field is set to `null` and excluded from downstream thresholding.

These three signals are intentionally separated to disentangle **what** the model answered from **how** it reasoned.

---

## Judgment Labels (Quality Tiers)

Based on the above scores, each sample is assigned a discrete quality tier:

- **Venti** — high-confidence correctness with strong, well-structured reasoning  
- **Grande** — generally correct with solid reasoning  
- **Tall** — partially correct or acceptable but imperfect reasoning  
- **No** — low-quality, incorrect, or unreliable samples  

Thresholds are strictly defined and consistently enforced.  
When the reference answer is missing, judgments are computed using correctness and reasoning scores only.

---

## Why This Scoring Design?

This evaluation framework is designed to address several limitations of traditional SFT datasets:

- **Avoids binary “correct / incorrect” labeling** by introducing graded supervision  
- **Separates reasoning quality from answer correctness**, enabling finer control over training signals  
- **Preserves original chain-of-thought text verbatim**, allowing downstream users to decide how (or whether) to use it  
- **Supports curriculum learning**, where models can be trained progressively from *Tall → Grande → Venti*  
- **Enables post-hoc filtering and reweighting** without regenerating data  

By using a single, strong instruction model as a consistent grader, the dataset achieves **scalable, reproducible, and interpretable quality annotations** suitable for reasoning-focused model training and evaluation.

---


## Data Structure & Fields
- **CoT_score**: present in 100.0% of samples
- **average_score**: present in 100.0% of samples
- **conversation**: present in 100.0% of samples
- **judgment**: present in 100.0% of samples
- **model_answer_score**: present in 100.0% of samples
- **reference_answer**: present in 82.19% of samples
- **reference_answer_similarity**: present in 76.73% of samples

## Visualizations

![length_scatter](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/CPC5cSXjMe67YyXfdSedy.png)

![score_distributions](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/bhze1yfsl6h_ZIWX2o_-g.png)

![text_length_distributions](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/gcOJvipF2YP6G5bu-h_f3.png)


## Limitations & Ethical Considerations

- Scores and judgment labels are produced automatically and may contain noise or bias.
- `reference_answer_similarity` is missing for a subset of records.
- Assistant responses include chain-of-thought content; handle with care depending on policy and deployment setting.
- Base questions originate from natural_reasoning; any upstream artifacts may persist.

## Score Summary

- **average_score**: mean 9.26, median 10.0, p25 9.0, p75 10.0, min 1.0, max 10.0 (n=63852)
- **model_answer_score**: mean 9.18, median 10.0, p25 9.0, p75 10.0, min 0.0, max 10.0 (n=63852)
- **CoT_score**: mean 9.35, median 10.0, p25 9.0, p75 10.0, min 1.0, max 10.0 (n=63852)
- **reference_answer_similarity**: mean 78.26, median 90.0, p25 75.0, p75 95.0, min 0.0, max 100.0 (n=48993)

