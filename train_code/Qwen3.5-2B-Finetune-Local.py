# Qwen3.5-2B Fine-Tuning Script for Local A100

import os
import gc
import json
import re
import hashlib
import multiprocessing as mp
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
import wandb

# ==========================================
# 0. Environment Setup & Configuration
# ==========================================
print("Setting up environment for local A100 training...")

# Prompt for W&B API Key if not found in environment
wandb_api_key = os.environ.get("WANDB_API_KEY")
if not wandb_api_key:
    print("\n--- Weights & Biases Setup ---")
    print("W&B API key not found in environment variables.")
    print("Please enter your WANDB_API_KEY (or press Enter to disable W&B tracking):")
    wandb_api_key = input("> ").strip()

if wandb_api_key:
    wandb.login(key=wandb_api_key)
    print("Successfully logged into Weights & Biases.")
else:
    print("Running without Weights & Biases tracking.")
    os.environ["WANDB_DISABLED"] = "true"

OUTPUT_DIR = "./outputs/Qwen3.5-2B-Finetune"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Checkpoints and final model will be saved to: {OUTPUT_DIR}")

# Training Configuration
RANDOM_SEED = 12181531
MAX_CONTEXT_WINDOW = 8192
NUM_PROC = min(16, mp.cpu_count())  # Parallel processing for data

# Dataset configurations (using smaller samples for demonstration)
num_samples_dict = {
    "ds1": 2000,  # nohurry/Opus-4.6-Reasoning-3000x-filtered
    "ds2": 2000,   # Jackrong/Qwen3.5-reasoning-700x
}


# ==========================================
# 1. Model Loading (Unsloth)
# ==========================================
print("\nLoading unsloth/Qwen3.5-2B...")
# Import Unsloth late to ensure torch is ready
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-2B",
    max_seq_length = MAX_CONTEXT_WINDOW,
    load_in_4bit = True,  # 4-bit quantization for efficiency even on A100
    load_in_8bit = False,
    full_finetuning = False,
)

# Apply Chat Template early so it's available for data processing
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen3-thinking",
)

# ==========================================
# 2. LoRA Configuration
# ==========================================
print("\nConfiguring LoRA Adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "out_proj",],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = RANDOM_SEED,
    use_rslora = False,
    loftq_config = None,
)


# ==========================================
# 3. Data Processing Pipeline
# ==========================================
print("\nPreparing Datasets...")

def _strip(x): return (x or "").strip()

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

def normalize_assistant_to_think_solution(text: str) -> str:
    text = _strip(text)
    if not text: return "<think></think>\n"
    m = THINK_BLOCK_RE.search(text)
    if m:
        think_block = m.group(0).strip()
        rest = text[m.end():].lstrip()
        return f"{think_block}\n{rest}".rstrip() if rest else f"{think_block}\n"
    return f"<think></think>\n{text}".rstrip()

def load_and_sample(dataset_name, sample_count=None, split="train"):
    print(f"Downloading/Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split=split)
    if sample_count is not None:
        sample_count = min(sample_count, len(ds))
        ds = ds.shuffle(seed=RANDOM_SEED).select(range(sample_count))
    return ds

# Load raw datasets
ds1 = load_and_sample("nohurry/Opus-4.6-Reasoning-3000x-filtered", num_samples_dict["ds1"])
ds2 = load_and_sample("Jackrong/Qwen3.5-reasoning-700x", num_samples_dict["ds2"])

# Format converters
def format_ds1(examples):
    problems = examples.get("problem", [])
    thinkings = examples.get("thinking", [])
    solutions = examples.get("solution", [])
    out = []
    for p, t, s in zip(problems, thinkings, solutions):
        p = _strip(p)
        t = _strip(t)
        s = _strip(s)
        if not p or not s: continue
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
        if not conv: continue
        cleaned = []
        for m in conv:
            frm = (m.get("from") or "").strip()
            val = m.get("value", "")
            if frm == "human":
                cleaned.append({"role": "user", "content": _strip(val)})
            elif frm == "gpt":
                cleaned.append({"role": "assistant", "content": normalize_assistant_to_think_solution(val)})
        if len(cleaned) < 2 or cleaned[-1]["role"] != "assistant": continue
        out.append(cleaned)
    return {"conversations": out}

print("Normalizing conversation formats...")
ds1 = ds1.map(format_ds1, batched=True, remove_columns=ds1.column_names)
ds2 = ds2.map(format_ds2, batched=True, remove_columns=ds2.column_names)

ds1 = ds1.filter(lambda x: x["conversations"] is not None and len(x["conversations"]) > 0)
ds2 = ds2.filter(lambda x: x["conversations"] is not None and len(x["conversations"]) > 0)

# Merge datasets
combined_dataset = concatenate_datasets([ds1, ds2]).shuffle(seed=RANDOM_SEED)

print("Applying Chat Template...")
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

dataset = combined_dataset.map(formatting_prompts_func, batched=True)

print(f"Filtering sequences longer than {MAX_CONTEXT_WINDOW} tokens...")
_text_tok = getattr(tokenizer, "tokenizer", tokenizer)
def filter_long_sequences_batched(examples):
    texts = examples["text"]
    tokenized = _text_tok(texts, truncation=False, padding=False, add_special_tokens=False)["input_ids"]
    return [len(toks) <= MAX_CONTEXT_WINDOW for toks in tokenized]

dataset = dataset.filter(filter_long_sequences_batched, batched=True, num_proc=NUM_PROC)
print(f"Final training dataset size: {len(dataset)} samples")


# ==========================================
# 4. SFTTrainer Configuration (A100 Optimized)
# ==========================================
print("\nConfiguring SFTTrainer for A100...")
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        # A100 Optimization: Increased batch size, reduced accumulation
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 2,
        warmup_ratio = 0.05,
        num_train_epochs = 1, # Just 1 epoch for quick demonstration
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = RANDOM_SEED,
        save_steps = 200,
        save_total_limit = 2,
        save_strategy = "steps",
        report_to = "wandb" if wandb_api_key else "none",
        output_dir = os.path.join(OUTPUT_DIR, "checkpoints"),
    ),
)

# Crucial: Train on assistant responses only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n<think>",
)

# ==========================================
# 5. Execute Training
# ==========================================
print("\n" + "="*50)
print("🚀 Starting Fine-Tuning!")
print("="*50)
trainer.train()


# ==========================================
# 6. Local Saving & Merging
# ==========================================
print("\n" + "="*50)
print("💾 Training Complete. Saving Model Locally...")
print("="*50)

LORA_SAVE_PATH = os.path.join(OUTPUT_DIR, "Qwen3.5-2B-LoRA-Weights")
MERGED_SAVE_PATH = os.path.join(OUTPUT_DIR, "Qwen3.5-2B-Merged-16bit")

# 1. Save LoRA Adapters
model.save_pretrained(LORA_SAVE_PATH)
tokenizer.save_pretrained(LORA_SAVE_PATH)
print(f"✅ LoRA weights saved to: {LORA_SAVE_PATH}")

# 2. Merge to 16-bit and Save Full Model
print("Merging LoRA weights with base model into 16-bit precision...")
model.save_pretrained_merged(MERGED_SAVE_PATH, tokenizer, save_method="merged_16bit")
print(f"✅ Full 16-bit merged model saved to: {MERGED_SAVE_PATH}")

print("\n🎉 All processes finished successfully!")
