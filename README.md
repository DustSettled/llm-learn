# Qwen3.5-2B 本地微调项目

本项目是一个面向本地 GPU 服务器的 **Qwen3.5-2B 监督微调（SFT）工程**。项目基于 Unsloth、QLoRA、Hugging Face Datasets 和 TRL，将多个推理类数据集统一清洗为 Qwen thinking 对话格式，并在本地完成 LoRA 训练、权重保存和 16-bit 完整模型合并。

项目目标不是构建一个复杂训练平台，而是提供一条可以复现、可以修改、可以继续扩展的数据处理与模型微调流水线。

## 项目解决的问题

- **降低本地微调门槛**：使用 Unsloth 与 4-bit 量化加载模型，在单机 A100 环境中完成 Qwen3.5-2B 微调。
- **统一异构数据格式**：将不同来源的数据转换为统一的 `user / assistant` 对话结构，并强制 assistant 输出符合 `<think>...</think>\n最终答案` 格式。
- **保留推理能力训练信号**：使用 `qwen3-thinking` chat template，让模型学习带思维链的回答风格。
- **避免训练 prompt 本身**：通过 `train_on_responses_only` 只对 assistant 回复计算 loss，减少模型学习用户输入的风险。
- **本地化产物管理**：训练后同时保存 LoRA adapter 和合并后的 16-bit 完整模型，便于后续部署、测试或量化。

## 目录结构

```text
.
├── train_code/
│   ├── Qwen3.5-2B-Finetune-Local.py   # 本地 GPU 训练主脚本
│   ├── Qwen3.5-9B-Neo-Kaggle.py       # Kaggle 版 Qwen3.5-9B 示例
│   ├── Qwopus3-5-27b-Colab.py         # Colab 版 Qwopus3.5-27B 示例
│   └── Qwopus-3.5-35B-A3B-Kaggle.py   # Kaggle 版 Qwopus3.5-35B MoE 示例
├── High-fidelity Dataset/             # 本地保存的高质量蒸馏数据集样例
├── download_datasets.py               # 批量下载 Jackrong 数据集的辅助脚本
├── merge.py                           # LoRA 与 base model 合并脚本示例
├── test.py                            # 合并模型的本地推理测试脚本示例
├── Qwen3.5-2B-本地微调计划.md          # 本地微调方案说明
└── README.md
```

## 核心训练流程

`train_code/Qwen3.5-2B-Finetune-Local.py` 是本项目的主脚本，流程如下：

1. 读取或提示输入 `WANDB_API_KEY`，可选启用 Weights & Biases 训练监控。
2. 使用 Unsloth 加载 `unsloth/Qwen3.5-2B`，上下文长度设置为 `8192`，开启 4-bit 量化。
3. 对模型注入 LoRA adapter，默认 `r=64`、`lora_alpha=64`。
4. 加载并采样两个推理数据集：
   - `nohurry/Opus-4.6-Reasoning-3000x-filtered`
   - `Jackrong/Qwen3.5-reasoning-700x`
5. 将不同字段格式统一转换为 Qwen thinking 对话格式。
6. 应用 `qwen3-thinking` chat template，过滤超过最大上下文长度的样本。
7. 使用 `SFTTrainer` 训练 LoRA adapter。
8. 保存 LoRA 权重，并合并导出 16-bit 完整模型。

默认训练参数偏向快速复现：

| 参数 | 默认值 |
| --- | --- |
| Base model | `unsloth/Qwen3.5-2B` |
| Max sequence length | `8192` |
| LoRA rank | `64` |
| LoRA alpha | `64` |
| Per-device batch size | `16` |
| Gradient accumulation steps | `2` |
| Epochs | `1` |
| Learning rate | `2e-4` |
| Optimizer | `adamw_8bit` |

## 从零搭建环境

以下步骤以 Linux + NVIDIA GPU 服务器为例。推荐使用 A100 40GB/80GB；其他显卡也可以尝试，但需要根据显存调小 batch size、上下文长度或 LoRA rank。

### 1. 准备系统环境

确认 NVIDIA 驱动和 CUDA 可用：

```bash
nvidia-smi
```

建议使用 Conda 创建独立环境：

```bash
conda create -n qwen35-2b-ft python=3.11 -y
conda activate qwen35-2b-ft
python -m pip install --upgrade pip
```

### 2. 安装 PyTorch

请根据服务器 CUDA 版本选择对应的 PyTorch 安装命令。CUDA 12.4 环境可参考：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

如果你的 CUDA 版本不同，请到 PyTorch 官方安装页选择匹配命令。

### 3. 安装训练依赖

```bash
pip install unsloth
pip install "transformers==5.3.0"
pip install --no-deps "trl==0.22.2"
pip install datasets peft accelerate bitsandbytes wandb sentencepiece protobuf huggingface_hub
```

如果 Unsloth 或 PyTorch 版本不兼容，优先以 Unsloth 官方给出的当前安装命令为准。

### 4. 准备 Hugging Face 与 W&B

模型和数据集会从 Hugging Face 下载。公开模型通常不需要登录(确保有Hugging Face账号)，但建议提前配置 token，避免下载限流：

```bash
huggingface-cli login
```

W&B 是可选项。如果需要记录训练曲线：

```bash
export WANDB_API_KEY="你的 wandb key"
```

如果不设置，脚本启动后会提示输入；直接回车会禁用 W&B。

### 5. 运行训练

在项目根目录执行：

```bash
python train_code/Qwen3.5-2B-Finetune-Local.py
```

首次运行会自动下载基础模型和数据集，耗时取决于网络和磁盘缓存状态。

### 6. 查看输出结果

默认输出目录：

```text
outputs/Qwen3.5-2B-Finetune/
├── checkpoints/                  # 训练 checkpoint
├── Qwen3.5-2B-LoRA-Weights/       # LoRA adapter 权重
└── Qwen3.5-2B-Merged-16bit/       # 合并后的 16-bit 完整模型
```

其中：

- `Qwen3.5-2B-LoRA-Weights/` 适合继续训练、分发 adapter 或重新合并。
- `Qwen3.5-2B-Merged-16bit/` 是已经融合 LoRA 的完整模型，可用于后续推理部署或 GGUF 量化。

### 7. 本地推理测试

仓库中的 `test.py` 是一个推理测试示例，但其中模型路径是本机绝对路径。使用前请将：

```python
model_path = "/.../llm-learn/outputs/Qwen3.5-2B-Finetune/Qwen3.5-2B-Merged-16bit"
```

改为你的实际合并模型路径，例如：

```python
model_path = "./outputs/Qwen3.5-2B-Finetune/Qwen3.5-2B-Merged-16bit"
```

然后运行：

```bash
python test.py
```

## 常见调整

### 不想使用 W&B

启动脚本时在 W&B 提示处直接回车即可。脚本会设置：

```python
os.environ["WANDB_DISABLED"] = "true"
```

## 许可证

本项目保留原仓库许可证文件，详见 [LICENSE](./LICENSE)。
