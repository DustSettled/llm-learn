---
license: apache-2.0
task_categories:
- text-classification
- question-answering
- text-generation
language:
- en
pretty_name: qwen3-coder-480b-distill-mini
size_categories:
- 1K<n<10K
---

### **qwen3-coder-480b-distill-mini**

---

### **Short Description**

This dataset is **distilled using Qwen3-Coder-480B-A35B-Instruct**.  
We extracted **10,000 code questions** from **microsoft/rStar-Coder** as seed problems, distilled them with **32K context**, and after cleaning and filtering, **9,543 samples remain**.  
**License: Apache-2.0.**

---

### **Dataset Overview**
- **Seed Source:** 10,000 code reasoning problems sampled from microsoft/rStar-Coder.  
- **Distillation Model:** Qwen3-Coder-480B-A35B-Instruct (**480B parameters, 35B active**).  
- **Context Length:** Up to **32K tokens** used during distillation.  
- **Final Count:** **9,543 cleaned samples**.  
- **License:** Apache-2.0.  

---

### **Why These Sources and Models?**

**Why rStar-Coder?**
- A **large-scale programming dataset** with **418K problems** and **580K reasoning solutions**.  
- Includes **input-output validation** with extensive test cases.  
- **High difficulty and diversity** make it ideal for constructing reliable datasets**.  

**Why Qwen3-Coder-480B-A35B-Instruct?**
- One of the **strongest Qwen code models**, based on a **Mixture-of-Experts (MoE) design**.  
- **Competitive with frontier models** in **agentic coding** and **tool usage**.  
- Supports **long context reasoning** (native **256K**, extendable to **1M tokens**).  
- **Apache-2.0 license** enables both **research** and **commercial applications**.  

---

### **Data Cleaning Process**
1. **Deduplication** – Keep only unique questions.  
2. **Integrity Check** – Remove entries missing **question/answer/reasoning**.  
3. **Format Validation** – Filter out **unbalanced brackets** and **truncated reasoning**.  
4. **Content Filtering** – Remove **overly templated** or **highly repetitive** reasoning traces.  
5. **Final Output** – **9,543 high-quality samples**.  

---

### **Field Schema**

Each JSONL record contains:
- **generator** : Qwen3-Coder-480B-A35B-Instruct
- **category** : code
- **Input:** Problem extracted from rStar-Coder  
- **code_output:** Final distilled answer
- **qid:** Unique ID (hash or UUID)  

---

### **License**

This dataset is released under the **Apache-2.0 License**.
