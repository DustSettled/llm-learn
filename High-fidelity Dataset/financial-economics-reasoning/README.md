---
license: apache-2.0
task_categories:
- text-generation
language:
- en
- zh
tags:
- finance
size_categories:
- 10K<n<100K
---

# Model Card

## 📌 Summary
financial-economics-reasoning dataset was constructed using advanced **Inference Distillation** techniques. We employed the **qwen-3-235b-a22b-thinking-2507** model as the Teacher Model to process the open-source **BAAI/IndustryInstruction_Finance-Economics** dataset, which contains **122,378** bilingual (Chinese-English) entries in finance, economics, and business.  

Unlike standard distillation datasets that only provide final answers, this dataset **retains the full reasoning chains (CoT) distilled from the Teacher Model**. For each instruction, the Teacher Model performed step-by-step reasoning and generated a complete reasoning trace along with the final distilled output.  

The distillation process aims to:  
- Filter out low-quality content  
- Enhance knowledge density and accuracy  
- Provide **reasoning-rich data** that supports training models with improved interpretability and reasoning ability  
- Enable downstream applications in finance, economics, and business with higher reliability  

---

## ⚙️ Parameters
- **Context Window:** 32,768  
- **Temperature:** Default recommended setting of the model  

---

## 📂 Data Source
- **Original Dataset:** [BAAI/IndustryInstruction_Finance-Economics]  
- **Languages:** Chinese & English  
- **Domain:** Finance, Economics, Business  

---

## 🛠️ Construction Method
1. **Teacher Model Selection:** qwen-3-235b-a22b-thinking-2507  
2. **Step-by-Step Reasoning:** Each instruction was processed with deep reasoning  
3. **Reasoning Chain Distillation:** The complete reasoning trace (not just the final answer) was distilled and preserved  
4. **Quality Enhancement:** Removed redundant or low-quality content, improving both accuracy and reasoning clarity  

---

## 🎯 Intended Use
- Training or fine-tuning dialogue models in finance, economics, and business  
- Research on **reasoning distillation** and interpretability in large language models  
- Improving downstream performance in tasks such as Q&A, summarization, and reasoning-intensive applications  

---

## ⚠️ Limitations & Considerations
- Domain-specific: limited to finance, economics, and business  
- Distilled reasoning chains may inherit biases or errors from the Teacher Model  
- Not suitable for high-stakes fields such as medicine or law without further validation  
- Must comply with the license of the original dataset  

---

## 📜 License
- Follows the license of **BAAI/IndustryInstruction_Finance-Economics**  
- Distilled dataset is intended for research and non-commercial use; commercial use may require additional authorization  

---

## 🔮 Future Work
- Extend reasoning distillation to other domains (e.g., law, healthcare, technology)  
- Explore multi-teacher distillation for diversity and robustness  
- Incorporate automated evaluation metrics to ensure reliability of reasoning chains




