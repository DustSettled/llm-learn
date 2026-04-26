#!/usr/bin/env python
# coding: utf-8

# <div align="center">
# 
# # Jackrong-llm-finetuning-guide 🌌
# **An Educational, End-to-End LLM Fine-Tuning Pipeline for Beginners and Developers**
# 
# This notebook currently focuses on a fine-tuning guide for **Qwopus3.5-27B inside Google Colab.**
# 
# 🌐 **Language:**  🇬🇧 **English** 
# 
# 🤗 **HuggingFace:** [Jackrong](https://huggingface.co/Jackrong)
# 
# <br>
# 
# [![Unsloth](https://img.shields.io/badge/Powered%20by-Unsloth-8A2BE2?style=flat-square)](https://github.com/unslothai/unsloth)
# [![Google Colab](https://img.shields.io/badge/Environment-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
# [![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
# [![Hugging Face](https://img.shields.io/badge/Model%20Hub-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
# [![LoRA PEFT](https://img.shields.io/badge/Technique-LoRA%20%2F%20PEFT-007EC6?style=flat-square)](#)
# [![Beginner Friendly](https://img.shields.io/badge/Level-Beginner%20Friendly-brightgreen?style=flat-square)](#)
# 
# </div>
# 
# ---

# ## Before You Start: Required API Keys 🔑
# 
# If you are new to Google Colab, please prepare **two API keys** before running the cells below, and store them in **Colab Secrets** first.
# 
# ### 1. `WANDB_API_KEY`
# 
# - This key is used to log in to **Weights & Biases (W&B)**.
# - In this notebook, it is used for experiment tracking, logging, and training visualization.
# - Without it, the W&B login cell at the beginning will fail.
# 
# ### 2. `HF_TOKEN`
# 
# - This key is used to log in to **Hugging Face**.
# - In this notebook, it is mainly needed later if you want to **upload the trained model or GGUF files to Hugging Face Hub**.
# - If you only want to train and save to Google Drive, this key may not be needed immediately, but it is still recommended to prepare it in advance.
# 
# ### How to store them in Colab Secrets
# 
# 1. Open your Colab notebook.
# 2. Click the **key icon** (Secrets) on the left sidebar.
# 3. Add the following secret names exactly as written:
#    - `WANDB_API_KEY`
#    - `HF_TOKEN`
# 4. Paste the corresponding value for each key and toggle **Notebook access** to ON.
# 5. Save the secrets, then come back and run the notebook.
# 
# ### Beginner Tip ✨
# 
# - Keep the secret **names** exactly the same as the code expects.
# - Do not paste API keys directly into notebook code cells.
# - Using Colab Secrets is the safer and cleaner way to manage credentials.

# In[ ]:


import os
import wandb
from google.colab import drive
from google.colab import userdata
drive.mount('/content/drive')
wandb_api_key = userdata.get('WANDB_API_KEY') 
wandb.login(key=wandb_api_key)
drive_output_path = "/content/drive/MyDrive/Qwen3.5-27B--checkpoints"
os.makedirs(drive_output_path, exist_ok=True)


# In[ ]:


get_ipython().run_cell_magic('capture', '', 'import os, importlib.util\n!pip install --upgrade -qqq uv\nif importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):\n    try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"\n    except: _numpy = "numpy"; _pil = "pillow"\n    !uv pip install -qqq \\\n        "torch==2.8.0" "triton>=3.3.0" {_numpy} {_pil} torchvision bitsandbytes xformers==0.0.32.post2 \\\n        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\\n        "unsloth[base] @ git+https://github.com/unslothai/unsloth"\nelif importlib.util.find_spec("unsloth") is None:\n    !uv pip install -qqq unsloth\n!uv pip install --upgrade --no-deps tokenizers trl==0.22.2 unsloth unsloth_zoo\n!uv pip install transformers==5.2.0\n# causal_conv1d is supported only on torch==2.8.0. If you have newer torch versions, please wait 10 minutes!\n!uv pip install --no-build-isolation flash-linear-attention causal_conv1d==1.6.0\n')


# In[ ]:


from unsloth import FastLanguageModel
import torch

fourbit_models = [
    "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit", # Qwen 14B 2x faster
    "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "unsloth/Qwen3-32B-unsloth-bnb-4bit",

    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/Phi-4",
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [NEW] We support TTS models!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-27B", # You can use any model from the list above and HF will download it for you. Depends on your GPU memory.
    max_seq_length = 32768, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
)


# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128 LoRA rank: controls the size of the low-rank adaptation matrices. Higher r = more capacity, but more VRAM usage and slower training.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "out_proj",],
    lora_alpha = 64, # LoRA scaling factor: scales the LoRA updates. Higher values increase the impact of the adapted weights.
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# ## Dataset Processing Overview 📚
# 
# This section is mainly meant to demonstrate the **workflow and core ideas behind SFT data preparation**, rather than provide a single "correct" data recipe. In real projects, you can absolutely replace them with datasets and ratios that better match your own task goals, data quality requirements, and domain needs.
# 
# This notebook processes data by mixing high-quality reasoning datasets.
# 
# ### Core Steps in This Pipeline
# 
# 1. **Decide on the datasets and sample sizes first**
#    This demo currently uses three datasets:
#    - `nohurry/Opus-4.6-Reasoning-3000x-filtered`: sample `3900` examples
#    - `Jackrong/Qwen3.5-reasoning-700x`: sample `700` examples
#    - `Roman1111111/claude-opus-4.6-10000x`: sample `9633` examples
# 
# 2. **Load the raw data**
#    - Standard HF datasets are loaded directly with `load_dataset()`, then randomly sampled.
# 
# 3. **Unify the data format**
#    Data from different sources is normalized into the training-ready `conversations` structure, and assistant responses are standardized into:
# 
#    ```text
#    <think>...</think>
#    final answer
#    ```
# 
# 4. **Apply the chat template**
#    The `qwen3-thinking` template is used to convert multi-turn conversations into the final text format that the model actually sees during training.
# 
# 5. **Mix, shuffle, deduplicate, and filter by length**
#    In this demo, the three datasets are processed separately, then merged directly with `concatenate_datasets()`, globally shuffled, and filtered to remove overly long sequences or invalid format samples.
# 
# ### Important Notes ✨
# 
# - The focus here is the **process**: how to load, sample, clean, normalize, template, mix, and filter data.
# - The dataset choices and sample sizes here are mainly for demonstrating the implementation pattern and processing logic.
# - You can choose datasets, sample sizes, and mixing weights that better match your own use case.

# In[ ]:


from datasets import load_dataset, concatenate_datasets, Dataset
from unsloth.chat_templates import get_chat_template
import re
import json
import multiprocessing as mp
import pandas as pd

RANDOM_SEED = 12181531
MAX_CONTEXT_WINDOW = 8192

num_samples_dict = {
    "ds1": 3900,  # nohurry/Opus-4.6-Reasoning-3000x-filtered
    "ds2": 700,   # Jackrong/Qwen3.5-reasoning-700x
    "ds3": 9633,  # Roman1111111/claude-opus-4.6-10000x
}

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen3-thinking",
)

def load_ds3_via_pandas_parquet():
    parquet_path = (
        "hf://datasets/Roman1111111/claude-opus-4.6-10000x"
        "@refs/convert/parquet/default/train/0000.parquet"
    )
    df = pd.read_parquet(parquet_path)
    return Dataset.from_pandas(df, preserve_index=False)

def load_and_sample(dataset_name, sample_count=None, split="train", subset=None):
    try:
        if subset:
            ds = load_dataset(dataset_name, subset, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
    except ValueError as e:
        err = str(e)
        if dataset_name == "Roman1111111/claude-opus-4.6-10000x" and "Feature type 'Json' not found" in err:
            ds = load_ds3_via_pandas_parquet()
        else:
            raise

    if sample_count is not None:
        sample_count = min(sample_count, len(ds))
        ds = ds.shuffle(seed=RANDOM_SEED).select(range(sample_count))

    return ds

# ds1: problem / thinking / solution
ds1 = load_and_sample(
    "nohurry/Opus-4.6-Reasoning-3000x-filtered",
    num_samples_dict["ds1"],
    split="train",
)

# ds2: multi-turn conversation
ds2 = load_and_sample(
    "Jackrong/Qwen3.5-reasoning-700x",
    num_samples_dict["ds2"],
    split="train",
)

# ds3: messages with possible reasoning fields
ds3 = load_and_sample(
    "Roman1111111/claude-opus-4.6-10000x",
    num_samples_dict["ds3"],
    split="train",
)

def _strip(x):
    return (x or "").strip()

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

def normalize_assistant_to_think_solution(text: str) -> str:
    text = _strip(text)

    if not text:
        return "<think></think>\n"

    m = THINK_BLOCK_RE.search(text)
    if m:
        think_block = m.group(0).strip()
        rest = text[m.end():].lstrip()
        return f"{think_block}\n{rest}".rstrip() if rest else f"{think_block}\n"

    return f"<think></think>\n{text}".rstrip()

def build_assistant_with_reasoning(content: str, reasoning: str = "") -> str:
    content = _strip(content)
    reasoning = _strip(reasoning)

    if "<think>" in content and "</think>" in content:
        return normalize_assistant_to_think_solution(content)

    if reasoning:
        if content:
            return f"<think>{reasoning}</think>\n{content}"
        return f"<think>{reasoning}</think>\n"

    return normalize_assistant_to_think_solution(content)

def parse_message_item(m):
    if isinstance(m, dict):
        return m

    if isinstance(m, str):
        s = m.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None

def format_ds1(examples):
    problems = examples.get("problem", [])
    thinkings = examples.get("thinking", [])
    solutions = examples.get("solution", [])

    out = []
    for p, t, s in zip(problems, thinkings, solutions):
        p = _strip(p)
        t = _strip(t)
        s = _strip(s)

        if not p or not s:
            continue

        assistant = f"<think>{t}</think>\n{s}" if t else f"<think></think>\n{s}"

        out.append([
            {"role": "user", "content": p},
            {"role": "assistant", "content": assistant},
        ])

    return {"conversations": out}

def format_ds2(examples):
    convos_list = examples.get("conversation", [])
    out = []

    for conv in convos_list:
        if not conv:
            continue

        cleaned = []
        for m in conv:
            frm = (m.get("from") or "").strip()
            val = m.get("value", "")

            if frm == "human":
                cleaned.append({"role": "user", "content": _strip(val)})
            elif frm == "gpt":
                cleaned.append({
                    "role": "assistant",
                    "content": normalize_assistant_to_think_solution(val),
                })

        if len(cleaned) < 2 or cleaned[-1]["role"] != "assistant":
            continue

        out.append(cleaned)

    return {"conversations": out}

def format_ds3(examples):
    messages_list = examples.get("messages", [])
    out = []

    for msgs in messages_list:
        if not msgs:
            continue

        parsed_msgs = []
        for m in msgs:
            pm = parse_message_item(m)
            if pm is not None:
                parsed_msgs.append(pm)

        if not parsed_msgs:
            continue

        convo = [m for m in parsed_msgs if m.get("role") != "system"]
        if len(convo) < 2 or convo[-1].get("role") != "assistant":
            continue

        cleaned = []
        for m in convo:
            role = m.get("role")
            content = m.get("content", "")
            reasoning = m.get("reasoning", "")

            if role == "assistant":
                content = build_assistant_with_reasoning(content, reasoning)
            else:
                content = _strip(content)

            if role in ("user", "assistant") and content is not None:
                cleaned.append({"role": role, "content": content})

        if len(cleaned) < 2 or cleaned[-1]["role"] != "assistant":
            continue

        out.append(cleaned)

    return {"conversations": out}

ds1 = ds1.map(format_ds1, batched=True, remove_columns=ds1.column_names)
ds2 = ds2.map(format_ds2, batched=True, remove_columns=ds2.column_names)
ds3 = ds3.map(format_ds3, batched=True, remove_columns=ds3.column_names)

ds1 = ds1.filter(lambda x: x["conversations"] is not None and len(x["conversations"]) > 0)
ds2 = ds2.filter(lambda x: x["conversations"] is not None and len(x["conversations"]) > 0)
ds3 = ds3.filter(lambda x: x["conversations"] is not None and len(x["conversations"]) > 0)

combined_dataset = concatenate_datasets([ds1, ds2, ds3]).shuffle(seed=RANDOM_SEED)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
        )
        for convo in convos
    ]
    return {"text": texts}

dataset = combined_dataset.map(formatting_prompts_func, batched=True)

num_proc = mp.cpu_count()
_text_tok = getattr(tokenizer, "tokenizer", tokenizer)

def filter_long_sequences_batched(examples):
    texts = examples["text"]
    tokenized = _text_tok(
        texts,
        truncation=False,
        padding=False,
        add_special_tokens=False,
    )["input_ids"]
    return [len(toks) <= MAX_CONTEXT_WINDOW for toks in tokenized]

dataset = dataset.filter(
    filter_long_sequences_batched,
    batched=True,
    num_proc=num_proc,
)

def check_assistant_format(examples):
    convos = examples["conversations"]
    ok = []
    for convo in convos:
        good = True
        for m in convo:
            if m["role"] == "assistant":
                c = m.get("content", "")
                if "<think>" not in c or "</think>" not in c:
                    good = False
                    break
                if not re.search(r"</think>\n", c):
                    good = False
                    break
        ok.append(good)
    return {"_ok": ok}

check = dataset.map(
    check_assistant_format,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=num_proc,
)

bad = len(check) - sum(check["_ok"])

if bad > 0:
    dataset = dataset.filter(lambda x: all(
        (m["role"] != "assistant") or (
            ("<think>" in m["content"]) and ("</think>\n" in m["content"])
        )
        for m in x["conversations"]
    ))

print(dataset[0]["text"][:800])


# In[ ]:


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 6,
        gradient_accumulation_steps = 6, # Use GA to mimic batch size!
        warmup_ratio = 0.05,
        #warmup_steps = 60,
        num_train_epochs = 2, # Set this for 1 full training run.
        #max_steps = 50,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        save_steps = 200,
        save_total_limit = 1,
        save_strategy = "steps",
        report_to = "wandb", # Can use Weights & Biases
        output_dir = drive_output_path,
    ),
)


# In[ ]:


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n<think>",
)


# In[ ]:


tokenizer.decode(trainer.train_dataset[100]["input_ids"])


# In[ ]:


trainer.train()


# In[ ]:


model.save_pretrained("qwen_lora")  # Local saving
tokenizer.save_pretrained("qwen_lora")
# model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving


# In[ ]:


from huggingface_hub import whoami
from google.colab import userdata

try:
    hf_token = userdata.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is not set")
except Exception as e:
    raise RuntimeError("HF_TOKEN was not found in Colab Secrets.") from e

try:
    username = whoami(token=hf_token)["name"]
    repo_id = f"{username}/Qwopus3.5-27B"
except Exception as e:
    raise RuntimeError("Failed to authenticate with Hugging Face.") from e

model.push_to_hub_merged(
    repo_id,
    tokenizer,
    save_method="merged_16bit",
    token=hf_token,
)

print(f"Uploaded to https://huggingface.co/{repo_id}")


# In[ ]:


from huggingface_hub import whoami
from google.colab import userdata

try:
    hf_token = userdata.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is not set")
except Exception as e:
    raise RuntimeError("HF_TOKEN was not found in Colab Secrets.") from e

try:
    username = whoami(token=hf_token)["name"]
    repo_id = f"{username}/Qwopus3.5-27B-GGUF"
except Exception as e:
    raise RuntimeError("Failed to authenticate with Hugging Face.") from e

model.push_to_hub_gguf(
    repo_id,
    tokenizer,
    quantization_method=["q4_k_m","q8_0","bf16"],
    token=hf_token,
)

print(f"Uploaded to https://huggingface.co/{repo_id}")

