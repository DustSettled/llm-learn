#!/usr/bin/env python
# coding: utf-8

# <div align="center">
# 
# # Jackrong-llm-finetuning-guide 🌌
# **An Educational, End-to-End LLM Fine-Tuning Pipeline for Beginners and Developers**
# 
# This notebook currently focuses on a fine-tuning guide for **Qwen3.5-9B inside Kaggle notebooks.**
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


get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    import torch; v = re.match(r\'[\\d]{1,}\\.[\\d]{1,}\', str(torch.__version__)).group(0)\n    xformers = \'xformers==\' + {\'2.10\':\'0.0.34\',\'2.9\':\'0.0.33.post1\',\'2.8\':\'0.0.32.post2\'}.get(v, "0.0.34")\n    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n!pip install transformers==5.3.0\n!pip install --no-deps trl==0.22.2\n')


# In[ ]:


import unsloth
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
    "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [NEW] We support TvTS models!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Unsloth/Qwen3.5-9B", # You can use any model from the list above and HF will download it for you. Depends on your GPU memory.
    max_seq_length = 16384, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
)


# In[ ]:


#!cp -r /kaggle/input/datasets/your-username/your-35b-checkpoint /kaggle/working/


# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128 LoRA rank: controls the size of the low-rank adaptation matrices. Higher r = more capacity, but more VRAM usage and slower training.
    modules_to_save = ["transformer.h"],
)
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
# This section is mainly meant to demonstrate the **workflow and core ideas behind SFT data preparation**, rather than provide a single "correct" data recipe. The dataset choices, sample sizes, and mixing strategy below are primarily selected for demonstration purposes, so I intentionally used popular and commonly discussed datasets from Hugging Face. In real projects, you can absolutely replace them with datasets and ratios that better match your own task goals, data quality requirements, and domain needs.
# 
# ### Core Steps in This Pipeline
# 
# 1. **Decide on the datasets and sample sizes first**
#    This demo currently uses two datasets:
#    - `Jackrong/Competitive-Programming-python-blend`: sample `700` examples
#    - `stepfun-ai/Step-3.5-Flash-SFT`: sample `100000` examples
# 
# 2. **Load the raw data**
#    - Standard HF datasets are loaded directly with `load_dataset()`, then randomly sampled.
#    - For datasets like `Step-3.5-Flash-SFT`, this demo shows how to first download the raw JSON files, then read, clean, and restructure examples from randomly sampled files. This is also practical here because the original dataset is very large and Kaggle temporary storage is limited.
# 
# 3. **Unify the data format**
#    Data from different sources is normalized into the training-ready `conversations` structure, and assistant responses are standardized into:
# 
#    ```text
#    <think>...</think>
#    final answer
#    ```
# 
# 4. **Run cleaning and validation checks**
#    Samples with invalid structure, incorrect role order, empty reasoning chains or answers, or assistant outputs that do not match the target format are filtered out to keep downstream training more stable.
# 
# 5. **Apply the chat template**
#    The `qwen3-thinking` template is used to convert multi-turn conversations into the final text format that the model actually sees during training.
# 
# 6. **Mix, shuffle, deduplicate, and filter by length**
#    In this demo, the two datasets are processed separately, then merged directly with `concatenate_datasets()`, globally shuffled, deduplicated by text, and filtered to remove overly long samples.
# 
# ### About the Current Data Ratio ⚖️
# 
# - The current sampling ratio is roughly `700 : 100000`.
# - In other words, `Step-3.5-Flash-SFT` is the main dataset here, while the programming dataset is added as a smaller supplementary source.
# - This ratio is meant to demonstrate how different data sources can flow through the same processing pipeline, not to claim that this is the optimal mixing strategy.
# 
# ### Important Notes ✨
# 
# - The focus here is the **process**: how to load, sample, clean, normalize, template, mix, and filter data.
# - The focus here is also the **reasoning behind the process**: why heterogeneous datasets should be converted into a unified schema before training.
# - The dataset choices and sample sizes here are mainly for demonstrating the implementation pattern and processing logic.
# - You can choose datasets, sample sizes, and mixing weights that better match your own use case.
# - Or you can simply pass your own dataset to the AI after it learns the pipeline and swap in your data directly. I’ll later package the dataset-processing part into a Skill so you can call it yourselves. 🚀

# In[ ]:


from datasets import load_dataset, concatenate_datasets, Dataset
from unsloth.chat_templates import get_chat_template
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import os
import json
import re
import multiprocessing as mp
import random
import shutil
import gc
import hashlib

# ==========================================
# 1. Configuration
# ==========================================
RANDOM_SEED = 1234 # Set a random seed for reproducibility.
MAX_CONTEXT_WINDOW = 16384 # Same with the model's context window.
NUM_PROC = 21 # Number of processes to use for data processing.

num_samples_dict = {
    "ds1": 700,   # Jackrong/Competitive-Programming-python-blend
    "ds2": 100000,   # stepfun-ai/Step-3.5-Flash-SFT
}

# Step dataset: randomly sample only a small number of JSON files
STEP_RANDOM_JSON_FILES = 5

# Build a candidate pool from the sampled files first
STEP_BUFFER_MULTIPLIER = 2
STEP_MIN_BUFFER = 20000

# Whether to delete the local snapshot cache immediately after loading Step
DELETE_STEP_SNAPSHOT_AFTER_LOAD = True

# Prefer in-memory processing for map/filter to reduce disk writes
KEEP_IN_MEMORY = True

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen3-thinking",
)

# ==========================================
# 2. General utility functions
# ==========================================
def _strip(x):
    return (x or "").strip() if isinstance(x, str) else ""

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

def normalize_assistant_to_think_solution(text: str) -> str:
    """
    Force the output into the form: <think>...</think>\n{solution}
    If the original text already contains a think block, extract the first one and normalize it;
    otherwise, prepend <think></think>\n{text} automatically.
    """
    text = _strip(text)

    if not text:
        return "<think></think>\n"

    m = THINK_BLOCK_RE.search(text)
    if m:
        think_block = m.group(0).strip()
        rest = text[m.end():].lstrip()
        return f"{think_block}\n{rest}".rstrip() if rest else f"{think_block}\n"
    else:
        return f"<think></think>\n{text}".rstrip()

def build_think_answer(reasoning_content: str, content: str) -> str:
    """
    Combine reasoning_content + content into the unified form:
    <think>reasoning_content</think>\ncontent
    Keep an empty think block even when reasoning_content is empty.
    """
    reasoning_content = _strip(reasoning_content)
    content = _strip(content)

    think_part = f"<think>{reasoning_content}</think>"
    if content:
        return f"{think_part}\n{content}"
    return f"{think_part}\n"

def force_cleanup():
    gc.collect()

def print_disk_hint(tag):
    print(f"[Memory cleanup point] {tag}")
    force_cleanup()

def get_num_proc():
    return min(NUM_PROC, mp.cpu_count())

def load_and_sample(dataset_name, sample_count=None, split="train", subset=None):
    print(f"Processing: {dataset_name} (target sample size: {sample_count})")
    if subset:
        ds = load_dataset(dataset_name, subset, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    if sample_count is not None:
        sample_count = min(sample_count, len(ds))
        ds = ds.shuffle(seed=RANDOM_SEED).select(range(sample_count))

    print(f"{dataset_name} actual size: {len(ds)}")
    print(f"{dataset_name} columns: {ds.column_names}")
    if len(ds) > 0:
        print(f"{dataset_name} first sample preview keys: {list(ds[0].keys())}")
    print("-" * 80)
    return ds

def canonicalize_conversation(msgs):
    """
    Normalize a multi-turn conversation so that:
    1. system messages and empty content are removed
    2. consecutive messages from the same role are merged
    3. every assistant message is forced into <think>...</think>\n...
    4. the conversation starts with user and ends with assistant
    5. user and assistant strictly alternate
    """
    cleaned = []

    for m in msgs:
        if not isinstance(m, dict):
            continue

        role = m.get("role", "")
        content = _strip(m.get("content", ""))

        if role == "system":
            continue
        if role not in {"user", "assistant"}:
            continue
        if content == "":
            continue

        if role == "assistant":
            content = normalize_assistant_to_think_solution(content)

        if cleaned and cleaned[-1]["role"] == role:
            prev = cleaned[-1]["content"].rstrip()
            if role == "assistant":
                merged_raw = prev + "\n" + content
                cleaned[-1]["content"] = normalize_assistant_to_think_solution(merged_raw)
            else:
                cleaned[-1]["content"] = prev + "\n" + content
        else:
            cleaned.append({"role": role, "content": content})

    if len(cleaned) < 2:
        return None
    if cleaned[0]["role"] != "user":
        return None
    if cleaned[-1]["role"] != "assistant":
        return None

    for i in range(1, len(cleaned)):
        if cleaned[i]["role"] == cleaned[i - 1]["role"]:
            return None

    return cleaned

# ==========================================
# 3. Step dataset: manually download raw JSON and clean it into a unified format
# ==========================================
def is_valid_step_message(msg):
    if not isinstance(msg, dict):
        return False

    role = msg.get("role", None)
    if role not in {"user", "assistant", "system"}:
        return False

    content = msg.get("content", None)
    reasoning_content = msg.get("reasoning_content", None)

    if content is not None and not isinstance(content, str):
        return False
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        return False

    return True

def iter_step_raw_examples(local_repo_dir, num_random_files=5, seed=RANDOM_SEED):
    json_root = os.path.join(local_repo_dir, "json")
    if not os.path.isdir(json_root):
        raise ValueError(f"JSON directory not found: {json_root}")

    json_files = []
    for root, _, files in os.walk(json_root):
        for fn in files:
            if fn.endswith(".json"):
                json_files.append(os.path.join(root, fn))

    if not json_files:
        raise ValueError(f"No .json files found under {json_root}")

    json_files.sort()
    total_files = len(json_files)

    rng = random.Random(seed)
    k = min(num_random_files, total_files)
    sampled_files = rng.sample(json_files, k)

    print(f"Total raw Step JSON files: {total_files}")
    print(f"Number of randomly sampled Step files: {k}")

    for fp in sampled_files:
        print(f"Reading file: {fp}")
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Skip] Failed to read: {fp} | {type(e).__name__}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[Skip] Top-level object is not a list: {fp}")
            continue

        for ex in data:
            yield ex

        del data
        force_cleanup()

def clean_step_example_to_final_conversations(example):
    """
    Clean a Step dataset example into the unified format:
    {
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "<think>...</think>\n..."},
            ...
        ]
    }
    """
    if not isinstance(example, dict):
        return None

    msgs = example.get("conversations", None)
    if not isinstance(msgs, list) or len(msgs) == 0:
        return None

    transformed = []

    for m in msgs:
        if not is_valid_step_message(m):
            return None

        role = m.get("role", "")
        content = _strip(m.get("content", ""))
        reasoning_content = _strip(m.get("reasoning_content", ""))

        if role == "system":
            continue

        if role == "user":
            if content != "":
                transformed.append({"role": "user", "content": content})

        elif role == "assistant":
            merged = build_think_answer(reasoning_content, content)
            merged = normalize_assistant_to_think_solution(merged)
            transformed.append({"role": "assistant", "content": merged})

    canonical = canonicalize_conversation(transformed)
    if canonical is None:
        return None

    return {"conversations": canonical}

def load_step35_flash_sft_manual(sample_count=None, revision=None, num_random_files=5):
    print(f"Processing: stepfun-ai/Step-3.5-Flash-SFT (manual JSON, target sample size: {sample_count})")

    local_repo_dir = snapshot_download(
        repo_id="stepfun-ai/Step-3.5-Flash-SFT",
        repo_type="dataset",
        revision=revision,
        allow_patterns=["json/*.json", "json/**/*.json"],
    )

    print(f"Local Step dataset directory: {local_repo_dir}")

    if sample_count is None:
        target_buffer = None
    else:
        target_buffer = max(STEP_MIN_BUFFER, sample_count * STEP_BUFFER_MULTIPLIER)

    cleaned_rows = []
    scanned = 0
    kept = 0
    dropped_nonlist = 0
    dropped_bad_inner = 0
    dropped_invalid_final = 0

    for ex in iter_step_raw_examples(
        local_repo_dir,
        num_random_files=num_random_files,
        seed=RANDOM_SEED,
    ):
        scanned += 1

        if not isinstance(ex, dict):
            dropped_bad_inner += 1
            continue

        convs = ex.get("conversations", None)
        if not isinstance(convs, list):
            dropped_nonlist += 1
            continue

        final_ex = clean_step_example_to_final_conversations(ex)
        if final_ex is None:
            dropped_invalid_final += 1
            continue

        cleaned_rows.append(final_ex)
        kept += 1

        if target_buffer is not None and len(cleaned_rows) >= target_buffer:
            break

    if DELETE_STEP_SNAPSHOT_AFTER_LOAD:
        try:
            shutil.rmtree(local_repo_dir, ignore_errors=True)
            print(f"Deleted local Step cache directory: {local_repo_dir}")
        except Exception as e:
            print(f"[Warning] Failed to delete local Step cache: {e}")

    force_cleanup()

    if len(cleaned_rows) == 0:
        raise ValueError("stepfun-ai/Step-3.5-Flash-SFT did not yield any valid samples from the sampled files")

    ds = Dataset.from_list(cleaned_rows)
    ds = ds.shuffle(seed=RANDOM_SEED)

    if sample_count is not None:
        sample_count = min(sample_count, len(ds))
        ds = ds.select(range(sample_count))

    print("------------------------------------------------")
    print(f"stepfun-ai/Step-3.5-Flash-SFT scanned samples: {scanned}")
    print(f"stepfun-ai/Step-3.5-Flash-SFT retained valid samples: {kept}")
    print(f"stepfun-ai/Step-3.5-Flash-SFT filtered because conversations was not a list: {dropped_nonlist}")
    print(f"stepfun-ai/Step-3.5-Flash-SFT filtered due to malformed internal structure or message fields: {dropped_bad_inner}")
    print(f"stepfun-ai/Step-3.5-Flash-SFT filtered invalid samples after cleaning: {dropped_invalid_final}")
    print(f"stepfun-ai/Step-3.5-Flash-SFT final size: {len(ds)}")
    print(f"stepfun-ai/Step-3.5-Flash-SFT columns: {ds.column_names}")
    if len(ds) > 0:
        print(f"stepfun-ai/Step-3.5-Flash-SFT first sample preview keys: {list(ds[0].keys())}")
    print("------------------------------------------------")
    return ds

# ==========================================
# 4. Load datasets
# ==========================================
print("Loading and sampling datasets...")

ds1 = load_and_sample(
    "Jackrong/Competitive-Programming-python-blend",
    num_samples_dict["ds1"],
    split="train",
)

ds2 = load_step35_flash_sft_manual(
    sample_count=num_samples_dict["ds2"],
    revision=None,
    num_random_files=STEP_RANDOM_JSON_FILES,
)

print_disk_hint("Step loading completed, raw snapshot deleted")

# ==========================================
# 5. Format normalization
# ==========================================
print("Normalizing format and forcing every assistant turn into the <think>...</think>\\n... structure...")

def format_ds1(example):
    msgs = example.get("messages")

    if not isinstance(msgs, list) or len(msgs) == 0:
        return {"conversations": None}

    transformed = []

    for m in msgs:
        if not isinstance(m, dict):
            continue

        role = m.get("role", "")
        content = _strip(m.get("content", ""))

        if role == "system":
            continue
        if role not in {"user", "assistant"}:
            continue
        if content == "":
            continue

        if role == "assistant":
            content = normalize_assistant_to_think_solution(content)

        transformed.append({"role": role, "content": content})

    canonical = canonicalize_conversation(transformed)
    if canonical is None:
        return {"conversations": None}

    return {"conversations": canonical}

def format_ds2(example):
    convo = example.get("conversations")
    if not isinstance(convo, list) or len(convo) == 0:
        return {"conversations": None}

    canonical = canonicalize_conversation(convo)
    if canonical is None:
        return {"conversations": None}

    return {"conversations": canonical}

ds1 = ds1.map(
    format_ds1,
    remove_columns=ds1.column_names,
    keep_in_memory=KEEP_IN_MEMORY,
    load_from_cache_file=False,
)

ds2 = ds2.map(
    format_ds2,
    remove_columns=ds2.column_names,
    keep_in_memory=KEEP_IN_MEMORY,
    load_from_cache_file=False,
)

def non_empty_convo(x):
    return x["conversations"] is not None and len(x["conversations"]) > 0

ds1 = ds1.filter(
    non_empty_convo,
    keep_in_memory=KEEP_IN_MEMORY,
    load_from_cache_file=False,
)
ds2 = ds2.filter(
    non_empty_convo,
    keep_in_memory=KEEP_IN_MEMORY,
    load_from_cache_file=False,
)

print(f"ds1 size after formatting: {len(ds1)}")
print(f"ds2 size after formatting: {len(ds2)}")

# ==========================================
# 6. Assistant format and multi-turn structure validation
# ==========================================
def is_valid_conversation(example):
    convo = example["conversations"]

    if not isinstance(convo, list) or len(convo) < 2:
        return False
    if convo[0]["role"] != "user":
        return False
    if convo[-1]["role"] != "assistant":
        return False

    for i, m in enumerate(convo):
        role = m.get("role", None)
        content = m.get("content", None)

        if role not in {"user", "assistant"}:
            return False
        if not isinstance(content, str) or not content.strip():
            return False

        if i > 0 and convo[i - 1]["role"] == role:
            return False

        if role == "assistant":
            if "<think>" not in content or "</think>" not in content:
                return False
            if "</think>\n" not in content:
                return False

    return True

before_ds1 = len(ds1)
before_ds2 = len(ds2)

num_proc = get_num_proc()
print(f"Detected {mp.cpu_count()} CPU cores, using {num_proc} processes")

ds1 = ds1.filter(
    is_valid_conversation,
    num_proc=num_proc,
    keep_in_memory=KEEP_IN_MEMORY,
    load_from_cache_file=False,
)

ds2 = ds2.filter(
    is_valid_conversation,
    num_proc=num_proc,
    keep_in_memory=KEEP_IN_MEMORY,
    load_from_cache_file=False,
)

print(f"Removed by ds1 format validation: {before_ds1 - len(ds1)}")
print(f"Removed by ds2 format validation: {before_ds2 - len(ds2)}")

print_disk_hint("Format normalization and validation completed")

# ==========================================
# 7. Apply the chat template to each dataset and drop the conversations column
# ==========================================
print("Applying the chat template to each dataset and dropping the conversations column...")

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

ds1_text = ds1.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=["conversations"],
    keep_in_memory=KEEP_IN_MEMORY,
    load_from_cache_file=False,
)

del ds1
print_disk_hint("ds1 converted to text and original ds1 deleted")

ds2_text = ds2.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=["conversations"],
    keep_in_memory=KEEP_IN_MEMORY,
    load_from_cache_file=False,
)

del ds2
print_disk_hint("ds2 converted to text and original ds2 deleted")

# ==========================================
# 8. Merge the text-only datasets
# ==========================================
print("Merging text-only datasets and shuffling...")
dataset = concatenate_datasets([ds1_text, ds2_text]).shuffle(seed=RANDOM_SEED)

del ds1_text, ds2_text
print_disk_hint("Text-only merge completed")

print(f"Merged dataset size: {len(dataset)}")

# ==========================================
# 9. Deduplicate by exact match on the final training text
# ==========================================
print("Deduplicating by exact text match...")

seen = set()
keep_indices = []

for i, text in enumerate(dataset["text"]):
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    if h not in seen:
        seen.add(h)
        keep_indices.append(i)

before_dedup = len(dataset)
dataset = dataset.select(keep_indices)
after_dedup = len(dataset)

print("------------------------------------------------")
print(f"Total before deduplication: {before_dedup}")
print(f"Total after deduplication: {after_dedup}")
print(f"Duplicates removed: {before_dedup - after_dedup}")
print("------------------------------------------------")

dataset = dataset.shuffle(seed=RANDOM_SEED)
print_disk_hint("Deduplication completed")

# ==========================================
# 10. Token-length filtering
# ==========================================
num_proc = get_num_proc()
print(f"Detected {mp.cpu_count()} CPU cores, using {num_proc} processes")
print(f"Filtering samples longer than {MAX_CONTEXT_WINDOW} tokens...")

_text_tok = getattr(tokenizer, "tokenizer", tokenizer)

def filter_long_sequences_batched(examples):
    texts = examples["text"]

    tokenized = _text_tok(
        texts,
        truncation=True,
        max_length=MAX_CONTEXT_WINDOW + 1,
        padding=False,
        add_special_tokens=False,
    )["input_ids"]

    return [len(toks) <= MAX_CONTEXT_WINDOW for toks in tokenized]

before_count = len(dataset)
dataset = dataset.filter(
    filter_long_sequences_batched,
    batched=True,
    batch_size=128,
    num_proc=num_proc,
    keep_in_memory=KEEP_IN_MEMORY,
    load_from_cache_file=False,
)
after_count = len(dataset)

print("------------------------------------------------")
print(f"Total before length filtering: {before_count}")
print(f"Total after length filtering: {after_count}")
print(f"Removed by length filtering: {before_count - after_count}")
print("------------------------------------------------")

print_disk_hint("Length filtering completed")

# ==========================================
# 11. Final preview
# ==========================================
print("\nSample preview (text):")
print("=" * 80)
print(dataset[0]["text"][:800])
print("=" * 80)


# In[ ]:


print(dataset[10]["text"][:20000]) # check the example


# In[ ]:


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 6, # Number of samples processed on each device (GPU) in one forward/backward pass.
        gradient_accumulation_steps = 6, # Accumulate gradients over 6 steps before updating weights; simulates a larger batch size without needing more VRAM.
        warmup_ratio = 0.04,  # Use the first 3%-5% of total training steps to gradually increase the learning rate for more stable training.
        #warmup_steps = 5,
        num_train_epochs = 2, # Set this for 1 full training run.
        #max_steps = 100,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        save_steps = 100, # Save a checkpoint every 100 training steps.You can adjust this as needed.
        save_total_limit = 1, # Keep only the most recent checkpoint; older ones are deleted to save disk space.
        save_strategy = "steps",
        report_to = "wandb", 
        output_dir = output_directory,
    ),
)


# In[ ]:


dataset[10]['text']


# In[ ]:


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n<think>",
)


# In[ ]:


tokenizer.decode(trainer.train_dataset[10]["input_ids"])


# In[ ]:


tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[10]["labels"]]).replace(tokenizer.pad_token, " ")


# In[ ]:


trainer.train()

# If training is interrupted, you can resume with:
# trainer.train(resume_from_checkpoint=True)  # auto-load latest checkpoint or trainer.train(resume_from_checkpoint="checkpoint-xxx")  # specify a checkpoint path


# In[ ]:


model.save_pretrained("qwen_lora")  # Local saving
tokenizer.save_pretrained("qwen_lora")
# model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving


# In[ ]:


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
    username = user_info['name']
    repo_id = f"{username}/Qwopus-3.5-9B-neo"
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
# - Because of this limit, in some cases you may **not be able to export every GGUF size directly from Kaggle**.
# - A practical workaround is to first upload the **16-bit model**, then use the **Colab quantization notebook** I published to quantize and release the full set of model sizes in Colab.
# - Colab usually provides **more temporary storage**, so it is better suited for exporting all GGUF variants. 🚀
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
    username = user_info['name']
    repo_id = f"{username}/Qwopus-3.5-9B-neo-GGUF"
    print(f"Authenticated as: {username}")
    print(f"Target repository: {repo_id}")
except Exception as e:
    print("Authentication failed. Check the token and network connectivity.")
    raise e

# Convert and upload the GGUF model artifacts
print("Converting and uploading GGUF artifacts. This may take some time.")

model.push_to_hub_gguf(
    repo_id,
    tokenizer,
    quantization_method=["q4_k_m "],
    token=hf_token
)

print(f"GGUF upload complete: https://huggingface.co/{repo_id}")

