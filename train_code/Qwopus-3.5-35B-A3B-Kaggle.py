#!/usr/bin/env python
# coding: utf-8

# <div align="center">
# 
# # Jackrong-llm-finetuning-guide 🌌
# **An Educational, End-to-End LLM Fine-Tuning Pipeline for Beginners and Developers**
# 
# This notebook currently focuses on a fine-tuning guide for **Qwen3.5-35B-A3B inside Kaggle notebooks.**
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
# - > ⚠️ **Warning**
# >
# > Kaggle’s temporary disk space is effectively limited to around **92 GB in practice** (even though the UI may display **50 GB**, it can sometimes utilize up to ~90 GB). Because of this, **loading and training Qwen3.5-35B-A3B can be extremely challenging and prone to failure**.
# >
# > To avoid unnecessary issues, it is recommended to use Kaggle mainly for **learning and demonstration purposes**. For actual training or exporting, consider using **Google Colab or a local machine** instead.
# >
# > While some Kaggle accounts may have access to **H100 GPUs**, storage remains a major bottleneck. If you still choose to train on Kaggle, consider the following approaches:
# >
# > - **Option 1 (Recommended):**  
# >   Upload the **16-bit model** first, then use the **Colab quantization notebook** to generate all GGUF variants. Colab typically provides more temporary storage.
# >
# > - **Option 2 (Checkpoint-based workflow):**  
# >   - Save training checkpoints inside the `output/` directory.  
# >   - Upload the `output/` folder to a **Kaggle Dataset**.  
# >   - On another machine (or environment), use the **Kaggle CLI** to download the dataset.  
# >   - Load the checkpoint, merge the model, and perform export/quantization there.
# >
# > - This workflow helps bypass Kaggle disk limitations and allows you to complete the process on a machine with more available storage. 🚀
# ---
# 

# ## Before You Start: Required API Keys 🔑
# 
# If you are new to Kaggle notebooks, please prepare **two API keys** before running the cells below, and store them in **Kaggle Secrets** first.
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
# - If you only want to train inside Kaggle and do not plan to upload artifacts, this key may not be needed immediately, but it is still recommended to prepare it in advance.
# 
# ### How to store them in Kaggle Secrets
# 
# 1. Open your Kaggle notebook.
# 2. Open the **Secrets** panel on the right side.
# 3. Add the following secret names exactly as written:
#    - `WANDB_API_KEY`
#    - `HF_TOKEN`
# 4. Paste the corresponding value for each key.
# 5. Save the secrets, then come back and run the notebook.
# 
# ### Beginner Tip ✨
# 
# - Keep the secret **names** exactly the same as the code expects.
# - Do not paste API keys directly into notebook code cells.
# - Using Kaggle Secrets is the safer and cleaner way to manage credentials.
# 

# In[ ]:


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("WANDB_API_KEY")
import wandb 
wandb.login(key=secret_value_0)
print("Attempted to log in to W&B.")
import os
output_directory = "/kaggle/working/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
print(f"Checkpoints will be saved to: {output_directory}")


# In[ ]:


get_ipython().run_cell_magic('capture', '', 'import os, importlib.util\n!pip install --upgrade -qqq uv\nif importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):\n    try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"\n    except: _numpy = "numpy"; _pil = "pillow"\n    !uv pip install -qqq \\\n        "torch==2.8.0" "triton>=3.3.0" {_numpy} {_pil} torchvision bitsandbytes xformers==0.0.32.post2 \\\n        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\\n        "unsloth[base] @ git+https://github.com/unslothai/unsloth"\nelif importlib.util.find_spec("unsloth") is None:\n    !uv pip install -qqq unsloth\n!uv pip install --upgrade --no-deps tokenizers trl==0.22.2 unsloth unsloth_zoo\n!uv pip install transformers==5.2.0\n# causal_conv1d is supported only on torch==2.8.0. If you have newer torch versions, please wait 10 minutes!\n!uv pip install --no-build-isolation flash-linear-attention causal_conv1d==1.6.0\n')


# In[ ]:


#!cp -r /kaggle/input/datasets/your-username/your-35b-checkpoint /kaggle/working/


# In[ ]:


get_ipython().run_cell_magic('capture', '', '! pip uninstall unsloth unsloth_zoo -y\n! pip install git+https://github.com/unslothai/unsloth-zoo.git --no-deps\n! pip install git+https://github.com/unslothai/unsloth.git --no-deps\n')


# In[ ]:


import os
# Note that to get best performance on A100, we needed to install causal-conv1d
# For this to not take too long, we had to downgrade to torch 2.8 (from torch 2.9)
# This means we wouldn't be able to use torch's grouped_mm here
# so we fallback to unsloth triton kernels for MoE
# We need to disable autotuning to save both time and memory for the colab notebook.
# If you are trying this elsewhere, we might recommend
# you install Flash Attention, Flash Linear Attention and CausalConv1d with torch 2.9
# `!uv pip install --no-build-isolation flash-attn flash-linear-attention causal_conv1d==1.6.`
# You can even try playing around with the below env var for faster performance but make sure you have enough VRAM to try autotuning.
os.environ['UNSLOTH_MOE_DISABLE_AUTOTUNE'] = '1'


# In[ ]:


from unsloth import FastLanguageModel
import torch

max_seq_length = 8192 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower, 8, 16, 32, 64, 128, depends on your GPU VRAM

model, processor = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3.5-35B-A3B", # This is a very big model, might take a while for downloading # You can use any model from the list above and HF will download it for you. Depends on your GPU memory.
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = False, # Not supported for MoE (yet!)
)
tokenizer = processor.tokenizer # To tokenize text


# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # LoRA rank: controls the size of the low-rank adaptation matrices. Higher r = more capacity, but more VRAM usage and slower training. Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "gate_up_proj", #Enable LoRA on MoE layers
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = True, # Reduces memory usage
    random_state = 3407,
    bias = "none",
)


# ## Dataset Processing Overview 📚
# 
# This section is mainly meant to demonstrate the **workflow and core ideas behind SFT data preparation** for this Qwen3.5-35B-A3B notebook, rather than provide a single "correct" data recipe. The dataset choices, target sample sizes, and mixing strategy below are part of the current reasoning-focused demo setup. In real projects, you should absolutely replace them with datasets and ratios that better match your own task goals, quality standards, and compute budget.
# 
# ### Core Steps in This Pipeline
# 
# 1. **Decide on the datasets and target sample sizes first**
#    This notebook currently uses three datasets:
#    - `ds1` = `nohurry/Opus-4.6-Reasoning-3000x-filtered`: target `3900` samples
#    - `ds2` = `Jackrong/Qwen3.5-reasoning-700x`: target `700` samples
#    - `ds3` = `Roman1111111/claude-opus-4.6-10000x`: target `10000` samples
#    The loader uses `min(sample_count, len(ds))`, so the actual sampled count can be lower if a dataset is smaller than the configured target.
# 
# 2. **Load and randomly sample the raw data**
#    All three datasets are loaded directly with `load_dataset()`, then shuffled and sampled with a fixed random seed.
# 
# 3. **Unify different schemas into one `conversations` format**
#    - `ds1` is built from `problem / thinking / solution` fields.
#    - `ds2` is converted from `conversation=[{from, value}, ...]`.
#    - `ds3` is treated as `messages=[{role, content}, ...]`.
# 
#    Assistant responses are standardized into:
# 
#    ```text
#    <think>...</think>
#    final answer
#    ```
# 
# 4. **Filter empty or invalid conversations**
#    After the format conversion step, rows with missing or empty `conversations` are removed before merging.
# 
# 5. **Merge and shuffle the combined dataset**
#    The notebook merges all three processed datasets with `concatenate_datasets()` and then shuffles the combined result.
# 
# 6. **Apply the chat template**
#    The `qwen3-thinking` template is used to convert each conversation into the final `text` field consumed during training.
# 
# 7. **Filter by sequence length and validate assistant formatting**
#    Samples longer than `8192` tokens are removed, and a final validation pass checks that assistant turns still follow the `<think>...</think>\n...` structure.
# 
# ### About the Current Mixing Strategy ⚖️
# 
# - The configured target ratio is roughly `3900 : 700 : 10000`.
# - In practice, the mix is weighted most heavily toward `Roman1111111/claude-opus-4.6-10000x`, while the other two datasets act as supplementary sources.
# - This setup is meant to demonstrate how multiple reasoning-style data sources can be normalized into one training pipeline, not to claim that this is the optimal production blend.
# 
# ### Important Notes ✨
# 
# - The focus here is the **process**: how to load, sample, normalize, template, merge, filter, and validate the dataset.
# - The focus here is also the **reasoning behind the process**: why multiple dataset schemas need to be unified before training.
# - The dataset choices and target sample sizes are part of the current 35B demo setup.
# - You can replace the datasets, adjust the sample sizes, or rebalance the mix to better match your own use case and compute budget. 🚀
# 

# In[ ]:


from datasets import load_dataset, concatenate_datasets
from unsloth.chat_templates import get_chat_template
import re
import multiprocessing as mp

# ==========================================
# 1. Configuration Area
# ==========================================
RANDOM_SEED = 1234 # Set random seed for reproducibility.
MAX_CONTEXT_WINDOW = 8192  # Same with the model's context window.

num_samples_dict = {
    "ds1": 3900,   # nohurry/Opus-4.6-Reasoning-3000x-filtered
    "ds2": 700,    # Jackrong/Qwen3.5-reasoning-700x
    "ds3": 10000,  # Roman1111111/claude-opus-4.6-10000x
}

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen3-thinking",
)

# ==========================================
# 2. Load and Sample Datasets
# ==========================================
print("Loading and sampling datasets...")

def load_and_sample(dataset_name, sample_count=None, split="train", subset=None):
    print(f"Processing: {dataset_name} (Target: {sample_count})")
    if subset:
        ds = load_dataset(dataset_name, subset, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    if sample_count is not None:
        sample_count = min(sample_count, len(ds))
        ds = ds.shuffle(seed=RANDOM_SEED).select(range(sample_count))

    print(f"{dataset_name} actually sampled: {len(ds)}")
    return ds

ds1 = load_and_sample(
    "nohurry/Opus-4.6-Reasoning-3000x-filtered",
    num_samples_dict["ds1"],
    split="train",
)

ds2 = load_and_sample(
    "Jackrong/Qwen3.5-reasoning-700x",
    num_samples_dict["ds2"],
    split="train",
)

ds3 = load_and_sample(
    "Roman1111111/claude-opus-4.6-10000x",
    num_samples_dict["ds3"],
    split="train",
)

# ==========================================
# 3. Unify conversations, and force assistant = "<think>...</think>\n..."
# ==========================================
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
        rest = text[m.end():]
        rest = rest.lstrip()
        return f"{think_block}\n{rest}".rstrip() if rest else f"{think_block}\n"
    else:
        return f"<think></think>\n{text}".rstrip()

def format_ds1(examples):
    problems  = examples.get("problem", [])
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
                cleaned.append({"role": "assistant", "content": normalize_assistant_to_think_solution(val)})
            else:
                continue

        if len(cleaned) < 2 or cleaned[-1]["role"] != "assistant":
            continue

        out.append(cleaned)

    return {"conversations": out}

def format_ds3(examples):
    """
    ds3 assumed structure:
    messages=[{role, content}, ...]
    """
    messages_list = examples.get("messages", [])
    out = []

    for msgs in messages_list:
        if not msgs:
            continue

        convo = [m for m in msgs if m.get("role") != "system"]
        if len(convo) < 2 or convo[-1].get("role") != "assistant":
            continue

        cleaned = []
        for m in convo:
            role = m.get("role")
            content = m.get("content", "")
            if role == "assistant":
                content = normalize_assistant_to_think_solution(content)
            else:
                content = _strip(content)
            cleaned.append({"role": role, "content": content})

        out.append(cleaned)

    return {"conversations": out}

print("Normalizing format and enforcing <think>...</think>\\n... structure...")

ds1 = ds1.map(format_ds1, batched=True, remove_columns=ds1.column_names)
ds2 = ds2.map(format_ds2, batched=True, remove_columns=ds2.column_names)
ds3 = ds3.map(format_ds3, batched=True, remove_columns=ds3.column_names)
ds1 = ds1.filter(lambda x: x["conversations"] is not None and len(x["conversations"]) > 0)
ds2 = ds2.filter(lambda x: x["conversations"] is not None and len(x["conversations"]) > 0)
ds3 = ds3.filter(lambda x: x["conversations"] is not None and len(x["conversations"]) > 0)

# ==========================================
# 4. Merge + Shuffle
# ==========================================
print("Merging and shuffling datasets...")
combined_dataset = concatenate_datasets([ds1, ds2, ds3]).shuffle(seed=RANDOM_SEED)
print(f"Total entries after merge: {len(combined_dataset)}")

# ==========================================
# 5. Apply Chat Template (Generating the text column)
# ==========================================
print("Applying Chat Template (qwen3.5)...")

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

# ==========================================
# 6. Filter by Token Length (Multiprocessing)
# ==========================================
num_proc = mp.cpu_count()
print(f"Detected {num_proc} CPU cores")
print(f"Using {num_proc} processes to filter out samples larger than {MAX_CONTEXT_WINDOW} tokens...")

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

before_count = len(dataset)
dataset = dataset.filter(
    filter_long_sequences_batched,
    batched=True,
    num_proc=num_proc
)
after_count = len(dataset)

print("------------------------------------------------")
print(f"Total amount before filtering: {before_count}")
print(f"Total amount after filtering: {after_count}")
print(f"Number of samples removed: {before_count - after_count}")
print("------------------------------------------------")

# ==========================================
# 7. Verification
# ==========================================
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
    num_proc=num_proc
)
bad = len(check) - sum(check["_ok"])
print(f"Format verification complete: Number of samples mismatching '<think>...</think>\\n...'   {bad}")

if bad > 0:
    dataset = dataset.filter(lambda x: all(
        (m["role"] != "assistant") or (("<think>" in m["content"]) and ("</think>\n" in m["content"]))
        for m in x["conversations"]
    ))

# ==========================================
# 8. Final Preview
# ==========================================
print("\nSample preview (text):")
print("=" * 80)
print(dataset[0]["text"][:800])
print("=" * 80)

print("\nSample preview (last assistant content):")
last_asst = [m for m in dataset[0]["conversations"] if m["role"]=="assistant"][-1]["content"]
print(last_asst[:800])


# In[ ]:


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2, # Number of samples processed on each device (GPU) in one forward/backward pass.
        gradient_accumulation_steps = 4,  # Accumulate gradients over 6 steps before updating weights; simulates a larger batch size without needing more VRAM.
        warmup_ratio = 0.04,  # Use the first 3%-5% of total training steps to gradually increase the learning rate for more stable training.
        #warmup_steps = 60,
        num_train_epochs = 2, # Set this for 1 full training run.
        #max_steps = 60,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        save_steps = 100,  # Save a checkpoint every 100 training steps.You can adjust this as needed.
        save_total_limit = 1, # Keep only the most recent checkpoint; older ones are deleted to save disk space.
        save_strategy = "steps",
        report_to = "wandb", # Can use Weights & Biases
        output_dir = output_directory,
    ),
)


# In[ ]:


dataset[100]['text']


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


tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")


# In[ ]:


# Compilation can take 2-3 minutes of time, so please be patient :)
trainer.train()
# If training is interrupted, you can resume with:
# trainer.train(resume_from_checkpoint=True)  # auto-load latest checkpoint or trainer.train(resume_from_checkpoint="checkpoint-xxx")  # specify a checkpoint path


# In[ ]:


model.save_pretrained("qwen_lora")  # Local saving
tokenizer.save_pretrained("qwen_lora")
# model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving


# In[ ]:


from kaggle_secrets import UserSecretsClient
import json
import os
from huggingface_hub import whoami

# Retrieve the Hugging Face token from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
except Exception as e:
    print("HF_TOKEN not found. Add a secret named 'HF_TOKEN' in your Kaggle settings.")
    raise e

print("Successfully retrieved HF_TOKEN from Kaggle Secrets.")

# Resolve the Hugging Face username and target repository ID
try:
    user_info = whoami(token=hf_token)
    username = user_info["name"]
    repo_id = f"{username}/Qwopus-3.5-35B-A3B"
    print(f"Authenticated as: {username}")
    print(f"Target repository: {repo_id}")
except Exception as e:
    print("Authentication failed. Check the token and network connectivity.")
    raise e

# Upload the merged 16-bit model
print("Uploading merged 16-bit model. This may take a few minutes.")

model.push_to_hub_merged(
    repo_id,
    tokenizer,
    save_method="merged_16bit",
    token=hf_token
)

print(f"Upload complete: https://huggingface.co/{repo_id}")


# ## Note Before Exporting GGUF Models ⚠️
# 
# - Please note that the temporary storage available to a Kaggle notebook is effectively capped at around **92 GB** in practice, even though the interface may show **50 GB**.
# - Because of this limit, large 35B GGUF exports can be difficult to complete directly in Kaggle.
# - A practical workaround is to first upload the **16-bit model**, then use the **Colab quantization notebook** I published to quantize and release the full set of GGUF sizes in Colab.
# - Colab usually provides **more temporary storage**, so it is better suited for full multi-size GGUF export workflows. 🚀
# 

# In[ ]:


from kaggle_secrets import UserSecretsClient
from huggingface_hub import whoami

# Retrieve the Hugging Face token from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
except Exception as e:
    print("HF_TOKEN not found. Add a secret named 'HF_TOKEN' in your Kaggle settings.")
    raise e

print("Successfully retrieved HF_TOKEN from Kaggle Secrets.")

# Resolve the Hugging Face username and target repository ID
try:
    user_info = whoami(token=hf_token)
    username = user_info["name"]
    repo_id = f"{username}/Qwopus-3.5-35B-A3B-GGUF"
    print(f"Authenticated as: {username}")
    print(f"Target repository: {repo_id}")
except Exception as e:
    print("Authentication failed. Check the token and network connectivity.")
    raise e

# Convert and upload the GGUF model artifacts
print("Converting and uploading GGUF artifacts (q8_0). This may take some time.")

model.push_to_hub_gguf(
    repo_id,
    tokenizer,
    quantization_method=["q8_0"],
    token=hf_token
)

print(f"GGUF upload complete: https://huggingface.co/{repo_id}")

