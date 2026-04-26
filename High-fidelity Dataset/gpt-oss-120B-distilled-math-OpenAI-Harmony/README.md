---
license: apache-2.0
task_categories:
- text-classification
- question-answering
- text-generation
- summarization
language:
- en
pretty_name: Jackrong/gpt-oss-120B-distilled-math-OpenAI-Harmony
size_categories:
- 1K<n<10K
---
![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/Z_e-AT-WC-W1FyrvBhPS1.jpeg)
# 📚 Dataset Overview
**Data Source Model**: gpt-oss-120b  
**Task Type**: Mathematical Problem Solving  
**Data Format**: JSON Lines (.jsonl)  
**Fields**: Generator, Category, Input, Output 

**Note**: If you are using this template for training, please make sure the format is correct before starting.Since this template is still under continuous improvement and learning, it may not be fully complete yet. I appreciate your understanding.

# 📈 Core Statistics
Generated complete reasoning processes and answers using **gpt-oss-120b** (MXFP4 format).  
The text length of the dataset reflects the depth and complexity of its content. I have statistically analyzed the lengths of the **input** (question), **Reasoning**, and **Answer**.  
To understand the data distribution more intuitively, I performed some visualization analysis.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/vhb8BN37WVllGjHrmFR1i.png)

---

## ⭐ Quality and Content Evaluation
This evaluation did not introduce an LLM scoring model. Instead, two custom quantitative metrics were used to assess data structure and reasoning characteristics:

- **Reasoning Complexity Ratio**: 39.19  
  *Calculation Method*: Average reasoning characters ÷ Average input characters  
  *Meaning*: Measures the extent of the model's reasoning chain. A higher value means the model provides sufficient reasoning details even for short questions.

- **Answer Efficiency Ratio**: 0.67  
  *Calculation Method*: Average answer words ÷ Average reasoning words  
  *Meaning*: Measures the refinement from reasoning to the answer. A lower value indicates that the reasoning is divergent, while the answer is convergent and concise.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/GNkBaNsuasJ4Jal9w0ABm.png)  

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/3tQm0sHGSxeTU1ZverdQk.png)  

---

## ✅ Comprehensive Evaluation
The dataset demonstrates high-quality mathematical problem-solving capabilities, featuring:
- **Comprehensive Reasoning Chain**: Detailed thought processes and clear logical steps.
- **Rich Mathematical Expression**: Effective use of LaTeX for formula typesetting.
- **Balanced Input-Output Relationship**: The complexity of the reasoning process is reasonably correlated with the complexity of the problem.

---

## 🏗️ Dataset Structure
**File Format**: .jsonl (one sample per line, independent JSON object)

**Example**:
```json
{"Generator": "gpt-oss-120b",
"Category": "math",
"Input": "<|start|>user<|message|>In triangle $ABC$ with $BC=3$, $CA=4$, $AB=5$ and $\\angle C = 90^{\\circ}$, there exists exactly one circle $\\omega$ that is tangent to segment $AB$ and rays $\\overrightarrow{CB}$ and $\\overrightarrow{CA}$. Let $A'$ be the image of the reflection of point $A$ over the line $BC$. Compute the square of the distance from the incenter of $\\triangle A'BC$ and the center of $\\omega$.<|end|>\n<|start|>assistant",
"output": "<|channel|>analysis<|message|>We have right triangle ABC with right angle at C, sides: BC=3 (adjacent to angle B?), CA=4, AB=5 (hypotenuse). So coordinates: Let C at origin (0,0), B on x-axis (3,0), A on y-axis (0,4). Indeed BC=3 along x, CA=4 along y, AB distance sqrt(3^2+4^2)=5. Good.\n\nWe have a circle ω tangent to segment AB and rays CB and CA. So it's an incircle of the right angle region formed by the two legs extended as rays from C, but tangent also to AB (the hypotenuse) but only to the segment AB (not its extension). So it's the incircle of the right triangle? Wait incircle of triangle ABC is tangent to all three sides, but here ω is tangent to AB (segment) and to the rays CB and CA (i.e., the lines containing BC and CA, but only the rays starting at C going outward). Since BC and CA are sides of triangle, the incircle of triangle ABC is tangent to BC and CA as segments, not just rays. However the incircle would be tangent to BC at a point between B and C, same for CA. But ω is tangent to the rays, meaning it could be outside the triangle, tangent to the extensions beyond C? Let's think.\n\nWe have right angle at C. The rays from C along BC (positive x direction) and CA (positive y direction). The region bounded by those rays is the first quadrant. The segment AB is the line from (0,4) to (3,0). The circle tangent to both axes (the rays) and to the line AB is the inscribed circle in the right angle corner touching the hypotenuse. That's the incircle of the right triangle indeed, because the incircle touches the legs at points within the segments. So it's the incircle of triangle ABC. Wait but incircle is tangent to BC and CA as segments, which are subsets of the rays. So it's also tangent to the rays. So ω is the incircle of triangle ABC. However the problem says \"exactly one circle ω that is tangent to segment AB and rays CB and CA\". Indeed there is exactly one such circle: the incircle.\n\nThus ω is incircle of triangle ABC. Its center is the incenter I of triangle ABC. Coordinates: In a right triangle, incenter is at (r, r) where r is inradius. For legs a=3, b=4, hypotenuse c=5. Inradius r = (a+b-c)/2 = (3+4-5)/2 = (2)/2 =1. So I = (1,1). Good.\n\nNow A' is the image of the reflection of point A over line BC. Wait \"Let A' be the image of the reflection of point A over the line BC.\" That phrasing: reflect point A across line BC to get A_reflection, then A' is the image of that reflection? Might be just A' is the reflection of A across BC. Probably they mean A' is the reflection of A across BC. So reflect A (0,4) across line BC (x-axis from (0,0) to (3,0)). Reflection across BC (the x-axis) gives A' = (0, -4). So A' = (0, -4).\n\nNow consider triangle A'BC: vertices (0,-4), (3,0), (0,0). That's a right triangle with right angle at C again? Points: B (3,0), C (0,0), A' (0,-4). So BC is along x-axis from (0,0) to (3,0). CA' is vertical downwards from (0,0) to (0,-4). So triangle A'BC is also right with legs 3 and 4, same shape but reflected.\n\nWe need the incenter of triangle A'BC, call it I'. Compute its coordinates. For right triangle with legs along axes, incenter is at (r', r') measured from the right angle vertex C, but note orientation: coordinates (r', -r')? Since the legs are along positive x (BC) and negative y (CA'), the right angle at C (0,0). The incircle will be tangent to both legs (positive x axis and negative y axis). So its center will be at (r', -r') where r' is inradius of triangle A'BC. Since sides lengths are same (3,4,5), inradius is also 1. So I' = (1, -1). Indeed distance to x-axis (y=0) is |y| = 1, distance to y-axis (x=0) is 1.\n\nNow we need the distance between I' (incenter of triangle A'BC) and the center of ω, which is I = (1,1). Distance = sqrt[(1-1)^2 + (1 - (-1))^2] = sqrt[0 + (2)^2] = 2. Square of distance = 4.\n\nThus answer 4.\n\nBut double-check: Are we sure that ω is incircle of triangle ABC? The condition: tangent to segment AB (the hypotenuse) and rays CB and CA (the legs extended from C outward). The incircle touches BC and CA at points within the segments, which are also on the rays. So yes it's tangent to the rays as well. There is exactly one such circle: incircle. So center is (1,1). Good.\n\nThus answer 4.\n\nCheck any nuance: Could there be another circle tangent to AB and the two rays but lying outside triangle, like an excircle? For a right triangle, there are three excircles tangent to one side and extensions of other two sides. The excircle opposite C is tangent to AB and the extensions of BC and CA beyond C (i.e., the rays opposite direction). But the problem says rays CB and CA (starting at C going towards B and A). Those are the interior directions, not extensions beyond C opposite direction. So the excircle opposite C would be tangent to the extensions beyond C opposite direction (i.e., the rays opposite direction). So not that. So unique incircle.\n\nThus answer 4.\n\nLet's produce final answer.<|end|>\n<|start|>assistant<|channel|>final<|message|>Place the triangle in the coordinate plane with  \n\n\\[\nC=(0,0),\\qquad B=(3,0),\\qquad A=(0,4)\n\\]\n\nso that \\(BC=3,\\; CA=4,\\; AB=5\\) and \\(\\angle C=90^{\\circ}\\).\n\n---\n\n### The circle \\(\\omega\\)\n\nA circle tangent to the segment \\(AB\\) and to the rays \\(\\overrightarrow{CB}\\) and \\(\\overrightarrow{CA}\\) must be tangent to the two legs of the right angle at \\(C\\) and to the hypotenuse \\(AB\\).  \nHence \\(\\omega\\) is precisely the incircle of \\(\\triangle ABC\\).\n\nFor a right triangle with legs \\(a,b\\) and hypotenuse \\(c\\),\n\n\\[\nr=\\frac{a+b-c}{2}\n\\]\n\nis the inradius.  \nHere \\(a=3,\\;b=4,\\;c=5\\), so  \n\n\\[\nr=\\frac{3+4-5}{2}=1 .\n\\]\n\nThus the incenter (center of \\(\\omega\\)) is at  \n\n\\[\nI=(r,r)=(1,1).\n\\]\n\n---\n\n### Point \\(A'\\)\n\nReflect \\(A\\) across line \\(BC\\) (the \\(x\\)-axis).  \nThe reflection of \\((0,4)\\) is  \n\n\\[\nA'=(0,-4).\n\\]\n\n---\n\n### Incenter of \\(\\triangle A'BC\\)\n\nTriangle \\(A'BC\\) has vertices  \n\n\\[\nA'=(0,-4),\\; B=(3,0),\\; C=(0,0),\n\\]\n\nso it is congruent to \\(\\triangle ABC\\) and also right‑angled at \\(C\\) with legs \\(3\\) and \\(4\\).  \nIts inradius is the same \\(r'=1\\).  \nSince the legs lie on the positive \\(x\\)-axis and the negative \\(y\\)-axis, the incenter is at  \n\n\\[\nI'=(r',-r')=(1,-1).\n\\]\n\n---\n\n### Required distance\n\n\\[\n\\begin{aligned}\n\\text{distance}(I,I') &= \\sqrt{(1-1)^2+(1-(-1))^2}\n= \\sqrt{0+2^2}=2,\\\\[2mm]\n\\text{square of the distance} &= 2^2 = 4.\n\\end{aligned}\n\\]\n\n\\[\n\\boxed{4}\n\\]<|return|>"}
```

---

## 📌 Training and Usage Recommendations
- **Alignment Training**: For CoT training, please ensure the template labels are suitable for the model.
- **Evaluation**: Report reasoning accuracy with/without CoT simultaneously; provide an "answer-in-the-box" parser to stabilize numerical extraction.
- **Safety Thresholds**: Prioritize quality over quantity for erroneous/inconsistent samples; set safety upper bounds for long samples and process them in chunks.

---

## Acknowledgements
The construction of this dataset is based on the generation capabilities of **gpt-oss-120b** and the optimized design of mathematical reasoning templates.  

**Seed Questions**: Derived in part from *nvidia/Nemotron-Post-Training-Dataset-v1*.  
**License**: CC-BY-4.0  

**Dataset Citation**:
```
@dataset{jackrong_2025_gpt_oss_math_distill,
  title   = {Jackrong/gpt-oss-120B-distilled-math-OpenAI-Harmony},
  author  = {Jackrong},
  year    = {2025},
  url     = {https://huggingface.co/datasets/Jackrong/gpt-oss-120B-distilled-math-OpenAI-Harmony}
}
```

---





# 中文：
# 📚 数据集概览
数据源模型: **gpt-oss-120b**  
任务类型: **Mathematical Problem Solving**  
数据格式: **JSON Lines (.jsonl)**  
字段: **Generator, Category, Input, Output**  

---

## 📈 核心统计指标
使用 **gpt-oss-120b**（MXFP4 格式）生成完整的推理过程与答案。  
数据集的文本长度反映了其内容的深度和复杂性。我对输入（问题）、Reasoning 和 Answer 的长度进行了详细统计。  
为了更直观地理解数据分布，我进行了可视化分析。

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/vhb8BN37WVllGjHrmFR1i.png)

---

## ⭐ 质量与内容评估
本次没有引入 LLM 评分模型，而是使用两项自定义量化指标评估数据结构与推理特性：

- **推理复杂度比率**（Reasoning Complexity Ratio）：39.19  
  *计算方式*：平均推理字符数 ÷ 平均输入字符数  
  *含义*：衡量模型推理链的展开程度。较高值意味着即使面对简短题目，模型也能提供充分的推理细节。

- **答案效率比率**（Answer Efficiency Ratio）：0.67  
  *计算方式*：平均答案词数 ÷ 平均推理词数  
  *含义*：衡量推理到答案的精炼程度。较低值代表推理是发散的，而答案是收敛简洁的。

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/GNkBaNsuasJ4Jal9w0ABm.png)  

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/3tQm0sHGSxeTU1ZverdQk.png)  

---

## ✅ 综合评估
数据集展现了高质量的数学问题解决能力，具有：
- **全面的推理链**：思维过程详尽，逻辑步骤清晰。
- **丰富的数学表达**：能够有效利用 LaTeX 进行公式排版。
- **均衡的输入输出关系**：推理过程的复杂性与问题的复杂性合理相关。

---

## 🏗️ 数据集结构
**文件格式**：.jsonl（每行一个样本，独立 JSON 对象）

**示例**：
```json
{"Generator": "gpt-oss-120b",
"Category": "math",
"Input": "<|start|>user<|message|>In triangle $ABC$ with $BC=3$, $CA=4$, $AB=5$ and $\\angle C = 90^{\\circ}$, there exists exactly one circle $\\omega$ that is tangent to segment $AB$ and rays $\\overrightarrow{CB}$ and $\\overrightarrow{CA}$. Let $A'$ be the image of the reflection of point $A$ over the line $BC$. Compute the square of the distance from the incenter of $\\triangle A'BC$ and the center of $\\omega$.<|end|>\n<|start|>assistant",
"output": "<|channel|>analysis<|message|>We have right triangle ABC with right angle at C, sides: BC=3 (adjacent to angle B?), CA=4, AB=5 (hypotenuse). So coordinates: Let C at origin (0,0), B on x-axis (3,0), A on y-axis (0,4). Indeed BC=3 along x, CA=4 along y, AB distance sqrt(3^2+4^2)=5. Good.\n\nWe have a circle ω tangent to segment AB and rays CB and CA. So it's an incircle of the right angle region formed by the two legs extended as rays from C, but tangent also to AB (the hypotenuse) but only to the segment AB (not its extension). So it's the incircle of the right triangle? Wait incircle of triangle ABC is tangent to all three sides, but here ω is tangent to AB (segment) and to the rays CB and CA (i.e., the lines containing BC and CA, but only the rays starting at C going outward). Since BC and CA are sides of triangle, the incircle of triangle ABC is tangent to BC and CA as segments, not just rays. However the incircle would be tangent to BC at a point between B and C, same for CA. But ω is tangent to the rays, meaning it could be outside the triangle, tangent to the extensions beyond C? Let's think.\n\nWe have right angle at C. The rays from C along BC (positive x direction) and CA (positive y direction). The region bounded by those rays is the first quadrant. The segment AB is the line from (0,4) to (3,0). The circle tangent to both axes (the rays) and to the line AB is the inscribed circle in the right angle corner touching the hypotenuse. That's the incircle of the right triangle indeed, because the incircle touches the legs at points within the segments. So it's the incircle of triangle ABC. Wait but incircle is tangent to BC and CA as segments, which are subsets of the rays. So it's also tangent to the rays. So ω is the incircle of triangle ABC. However the problem says \"exactly one circle ω that is tangent to segment AB and rays CB and CA\". Indeed there is exactly one such circle: the incircle.\n\nThus ω is incircle of triangle ABC. Its center is the incenter I of triangle ABC. Coordinates: In a right triangle, incenter is at (r, r) where r is inradius. For legs a=3, b=4, hypotenuse c=5. Inradius r = (a+b-c)/2 = (3+4-5)/2 = (2)/2 =1. So I = (1,1). Good.\n\nNow A' is the image of the reflection of point A over line BC. Wait \"Let A' be the image of the reflection of point A over the line BC.\" That phrasing: reflect point A across line BC to get A_reflection, then A' is the image of that reflection? Might be just A' is the reflection of A across BC. Probably they mean A' is the reflection of A across BC. So reflect A (0,4) across line BC (x-axis from (0,0) to (3,0)). Reflection across BC (the x-axis) gives A' = (0, -4). So A' = (0, -4).\n\nNow consider triangle A'BC: vertices (0,-4), (3,0), (0,0). That's a right triangle with right angle at C again? Points: B (3,0), C (0,0), A' (0,-4). So BC is along x-axis from (0,0) to (3,0). CA' is vertical downwards from (0,0) to (0,-4). So triangle A'BC is also right with legs 3 and 4, same shape but reflected.\n\nWe need the incenter of triangle A'BC, call it I'. Compute its coordinates. For right triangle with legs along axes, incenter is at (r', r') measured from the right angle vertex C, but note orientation: coordinates (r', -r')? Since the legs are along positive x (BC) and negative y (CA'), the right angle at C (0,0). The incircle will be tangent to both legs (positive x axis and negative y axis). So its center will be at (r', -r') where r' is inradius of triangle A'BC. Since sides lengths are same (3,4,5), inradius is also 1. So I' = (1, -1). Indeed distance to x-axis (y=0) is |y| = 1, distance to y-axis (x=0) is 1.\n\nNow we need the distance between I' (incenter of triangle A'BC) and the center of ω, which is I = (1,1). Distance = sqrt[(1-1)^2 + (1 - (-1))^2] = sqrt[0 + (2)^2] = 2. Square of distance = 4.\n\nThus answer 4.\n\nBut double-check: Are we sure that ω is incircle of triangle ABC? The condition: tangent to segment AB (the hypotenuse) and rays CB and CA (the legs extended from C outward). The incircle touches BC and CA at points within the segments, which are also on the rays. So yes it's tangent to the rays as well. There is exactly one such circle: incircle. So center is (1,1). Good.\n\nThus answer 4.\n\nCheck any nuance: Could there be another circle tangent to AB and the two rays but lying outside triangle, like an excircle? For a right triangle, there are three excircles tangent to one side and extensions of other two sides. The excircle opposite C is tangent to AB and the extensions of BC and CA beyond C (i.e., the rays opposite direction). But the problem says rays CB and CA (starting at C going towards B and A). Those are the interior directions, not extensions beyond C opposite direction. So the excircle opposite C would be tangent to the extensions beyond C opposite direction (i.e., the rays opposite direction). So not that. So unique incircle.\n\nThus answer 4.\n\nLet's produce final answer.<|end|>\n<|start|>assistant<|channel|>final<|message|>Place the triangle in the coordinate plane with  \n\n\\[\nC=(0,0),\\qquad B=(3,0),\\qquad A=(0,4)\n\\]\n\nso that \\(BC=3,\\; CA=4,\\; AB=5\\) and \\(\\angle C=90^{\\circ}\\).\n\n---\n\n### The circle \\(\\omega\\)\n\nA circle tangent to the segment \\(AB\\) and to the rays \\(\\overrightarrow{CB}\\) and \\(\\overrightarrow{CA}\\) must be tangent to the two legs of the right angle at \\(C\\) and to the hypotenuse \\(AB\\).  \nHence \\(\\omega\\) is precisely the incircle of \\(\\triangle ABC\\).\n\nFor a right triangle with legs \\(a,b\\) and hypotenuse \\(c\\),\n\n\\[\nr=\\frac{a+b-c}{2}\n\\]\n\nis the inradius.  \nHere \\(a=3,\\;b=4,\\;c=5\\), so  \n\n\\[\nr=\\frac{3+4-5}{2}=1 .\n\\]\n\nThus the incenter (center of \\(\\omega\\)) is at  \n\n\\[\nI=(r,r)=(1,1).\n\\]\n\n---\n\n### Point \\(A'\\)\n\nReflect \\(A\\) across line \\(BC\\) (the \\(x\\)-axis).  \nThe reflection of \\((0,4)\\) is  \n\n\\[\nA'=(0,-4).\n\\]\n\n---\n\n### Incenter of \\(\\triangle A'BC\\)\n\nTriangle \\(A'BC\\) has vertices  \n\n\\[\nA'=(0,-4),\\; B=(3,0),\\; C=(0,0),\n\\]\n\nso it is congruent to \\(\\triangle ABC\\) and also right‑angled at \\(C\\) with legs \\(3\\) and \\(4\\).  \nIts inradius is the same \\(r'=1\\).  \nSince the legs lie on the positive \\(x\\)-axis and the negative \\(y\\)-axis, the incenter is at  \n\n\\[\nI'=(r',-r')=(1,-1).\n\\]\n\n---\n\n### Required distance\n\n\\[\n\\begin{aligned}\n\\text{distance}(I,I') &= \\sqrt{(1-1)^2+(1-(-1))^2}\n= \\sqrt{0+2^2}=2,\\\\[2mm]\n\\text{square of the distance} &= 2^2 = 4.\n\\end{aligned}\n\\]\n\n\\[\n\\boxed{4}\n\\]<|return|>"}
```

---

## 📌 训练与使用建议
- **对齐训练**：CoT 训练请确保模板标签适合模型。
- **评测**：同时报告含/不含 CoT 的推理正确率；提供“盒中答案”解析器以稳定提取数值。
- **安全阈**：错误/不一致样本宁缺毋滥；对长样本设置安全上限并分块处理。

---

##  致谢
本数据集的构建基于 **gpt-oss-120b** 的生成能力以及数学推理模板的优化设计。   

**种子问题来源**：部分来自 *nvidia/Nemotron-Post-Training-Dataset-v1*。  
**许可协议**：CC-BY-4.0  

**数据集引用**：
```
@dataset{jackrong_2025_gpt_oss_math_distill,
  title   = {Jackrong/gpt-oss-120B-distilled-math-OpenAI-Harmony},
  author  = {Jackrong},
  year    = {2025},
  url     = {https://huggingface.co/datasets/Jackrong/gpt-oss-120B-distilled-math-OpenAI-Harmony}
}
```