---
license: apache-2.0
task_categories:
- text-classification
- question-answering
- text-generation
- table-question-answering
- token-classification
- translation
language:
- en
pretty_name: Jackrong/gpt-oss-120B-distilled-reasoning
size_categories:
- 1K<n<10K
---
![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/Z_e-AT-WC-W1FyrvBhPS1.jpeg)
# GPT-oss-120B-Distilled-Reasoning-math Dataset
**Data Source Model**: **gpt-oss-120b**  
**Task Type**: Mathematical Problem Solving  
**Data Format**: JSON Lines
**Fields**: Generator, Category, Input, Output 

---

#  Core Statistics
Generated complete reasoning processes and answers using **gpt-oss-120b** (MXFP4).  
The text length of the dataset reflects the depth and complexity of its content. I have statistically analyzed the lengths of the **input** (question), **Reasoning**, and **Answer**.  
To understand the data distribution more intuitively, I performed some visualization analysis.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/vhb8BN37WVllGjHrmFR1i.png)

---

##  Quality and Content Evaluation
This evaluation did not introduce an LLM scoring model. Instead, two custom quantitative metrics were used to assess data structure and reasoning characteristics:

- **Reasoning Complexity Ratio**: **39.19**  
  *Calculation Method*: Average reasoning characters ÷ Average input characters  
  *Meaning*: Measures the extent of the model's reasoning chain. A higher value means the model provides sufficient reasoning details even for short questions.

- **Answer Efficiency Ratio**: **0.67**  
  *Calculation Method*: Average answer words ÷ Average reasoning words  
  *Meaning*: Measures the refinement from reasoning to the answer. A lower value indicates that the reasoning is divergent, while the answer is convergent and concise.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/GNkBaNsuasJ4Jal9w0ABm.png)  

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/3tQm0sHGSxeTU1ZverdQk.png)  

---

##  Comprehensive Evaluation
The dataset demonstrates high-quality mathematical problem-solving capabilities, featuring:
- **Comprehensive Reasoning Chain**: Detailed thought processes and clear logical steps.
- **Rich Mathematical Expression**: Effective use of LaTeX for formula typesetting.
- **Balanced Input-Output Relationship**: The complexity of the reasoning process is reasonably correlated with the complexity of the problem.

---

##  Dataset Structure
**File Format**: .jsonl (one sample per line, independent JSON object)
To make training easier for everyone, I have prepared various data structure templates, offering three common annotation types for different distillation and cleaning logic.

**1. Standard JSON Structure**
To facilitate the training of reasoning models or the creation of SFT data, explicitly separate the chain of thought and the final answer in the output. 
```json
{
  "Generator": "gpt-oss-120b",
  "Category": "math",
  "Input": "Given that 2^x = 8, find x.",
  "CoT_Native_Reasoning": "We note that 8 = 2^3...",
  "answer": "The answer is 3."
}
```

**2. OpenAI Harmony**
Messages enclosed by tags <start>|user|message|...<end> and <start>|assistant|...<end>, aligning with the OpenAI Harmony style.
```json
{
  "Generator": "gpt-oss-120b",
  "Category": "math",
  "Input": "<start>|user|message|>In triangle ABC with BC=3, ... <end>",
  "output": "<start>|assistant|We have a right triangle at C, ... <end>"
}
```
**3. Think**
The format is like the Qwen3 series model and DeepSeek.

```json
{
  "Generator": "gpt-oss-120b",
  "Category": "math",
  "Input": "Solve: If 12x = 36, what is x?",
  "Output": "<think>First, divide both sides by 12. 36 / 12 = 3. So x = 3.</think> The answer is 3."
}
```
---

## Training and Usage Recommendations
- **Alignment Training**: For CoT training, please ensure the template labels are suitable for the model.
- **Evaluation**: Report reasoning accuracy with/without CoT simultaneously; provide an "answer-in-the-box" parser to stabilize numerical extraction.
- **Safety Thresholds**: Prioritize quality over quantity for erroneous/inconsistent samples; set safety upper bounds for long samples and process them in chunks.

---

##  Acknowledgements
The construction of this dataset is based on the generation capabilities of **gpt-oss-120b** and the optimized design of mathematical reasoning templates.  
Special thanks to the open-source community for their contributions in **mathematical expression formatting**, **data cleaning scripts**, and **visualization analysis**.  

**Seed Questions**: Derived in part from *nvidia/Nemotron-Post-Training-Dataset-v1*.  
**License**: CC-BY-4.0  

**Dataset Citation**:
```
@dataset{jackrong_2025_gpt_oss_math_distill,
  title   = {Jackrong/gpt-oss-120B-distilled-reasoning},
  author  = {Jackrong},
  year    = {2025},
  url     = {https://huggingface.co/datasets/Jackrong/gpt-oss-120B-distilled-reasoning}
}
```