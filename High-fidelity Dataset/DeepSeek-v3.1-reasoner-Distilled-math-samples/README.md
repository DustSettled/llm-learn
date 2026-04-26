---
license: apache-2.0
task_categories:
- question-answering
- text-generation
language:
- en
pretty_name: DeepSeek-v3.1-reasoner-Distilled-math-samples
size_categories:
- n<1K
---
# DeepSeek-V3.1 Distillation with NVIDIA Nemotron-Post-Training-Dataset-v2 (Math Subset)

The release of **DeepSeek-V3.1** has attracted wide attention in the AI community. Its significant improvements in reasoning ability provide a new opportunity to explore optimization of domain-specific models. To investigate the potential of this model in complex mathematical reasoning tasks, I selected the **math subset** from NVIDIA’s newly released **Nemotron-Post-Training-Dataset-v2** as seed problems and performed knowledge distillation on the **DeepSeek-V3.1 Reasoner** mode. The results are also compared with the outputs from **DeepSeek-R1-0528**.

---

# 📊 Results & Visualizations
<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/D0BZ2OsTWPUUJi_fBf4Fn.png" alt="Text Length Analysis: histograms for question, question word count, reasoning, answer, seed answer(by R1-0528), and averaged character lengths" width="65%">
  <br><em>Figure 1. Text length distributions for question, reasoning, answer, and seed answer.</em>
</p>

### 🔑 Key Takeaways
- **Average Length (characters)**: Question ≈ **227**, Reasoning ≈ **17,236**, Answer ≈ **2,143**, Seed Answer ≈ **1,669**.  
- **Reasoning** shows a clear **long-tail distribution**, with a few samples containing extremely long chains of thought — consistent with the behavior of the “Reasoner mode” on complex math tasks.  

---

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/bZbfdrwnneI0N6NXlmsFm.png" alt="Token Usage Analysis: histograms for prompt/completion/total tokens and a scatter of prompt vs completion" width="65%">
  <br><em>Figure 2. Prompt/completion/total token usage and their relationship.</em>
</p>

### 📌 Token Usage Highlights
- **Average Prompt Tokens ≈ 74.8**, **Average Completion Tokens ≈ 7,183**, **Average Total Tokens ≈ 7,258**.  
- There is a **weak positive correlation** between prompt and completion length: longer problem statements/contexts usually lead to longer reasoning outputs, though the decisive factor remains task complexity.  
- **Costs and latency** are driven almost entirely by **Completion length**; for reasoning tasks, optimizing output length and stability should be prioritized.
  
---

## Why NVIDIA’s Dataset?
The choice of NVIDIA’s dataset was based on several considerations:  

- As the latest public dataset, the problems in version 2 are less likely to have appeared in widely available training corpora.  
- According to NVIDIA’s technical report, the math subset was deliberately designed to emphasize evaluation of **mathematical reasoning** and **multi-step Chain-of-Thought** abilities.  
- Each problem is preceded by a standardized prompt:  

  > *“Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}...”*  

While such formatting instructions are useful for **SFT** and **RL evaluation** to standardize outputs, during distillation they may lead to templated reasoning chains, reducing the diversity of generated data.  

➡️ To mitigate this, I retained **50% of the template prompts** and removed the formatting instructions from the other half of the questions—helping to prevent overfitting and improve diversity.  

---

## Data Processing
- Extracted reasoning chains and answers from **DeepSeek-V3.1**.  
- Added comparison outputs from **DeepSeek-R1-0528** (answers only, without reasoning).  
- Collected additional metadata such as token lengths of reasoning and answers from V3.1.  

---

## Experiment Setup
- **API**: Official DeepSeek API  
- **Hyperparameters**: Default (temperature, top_p, etc. unchanged)  
- **Context Window**: 32K  

---

## Limitations
- The distillation scale is relatively small.  
- Since **DeepSeek-V3.1** is newly released, most third-party inference providers have not yet integrated the model.  
- Reliance on the official API, which is **unstable** and **extremely slow** (even with concurrency and API key rotation).  
- Only a small number of demonstration samples were distilled, followed by filtering and cleaning.  
- My hosting servers are currently occupied with GPT-related projects, so **larger-scale, high-quality dataset construction** will be done once resources are available.  

---

```bibtex
@misc{rong2025deepseekv31distill,
  title  = {DeepSeek-V3.1 Distillation with NVIDIA Nemotron-Post-Training-Dataset-v2 (Math Subset)},
  author = {Rong, Jack},
  year   = {2025},
  note   = {Jackrong/DeepSeek-v3.1-reasoner-Distilled-math-samples},
  url    = {https://huggingface.co/datasets/Jackrong/DeepSeek-v3.1-reasoner-Distilled-math-samples}
}
```