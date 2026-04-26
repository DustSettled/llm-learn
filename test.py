import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/.../Qwen3.5-2B-Finetune/Qwen3.5-2B-Merged-16bit"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

messages = [{"role": "user", "content": "你好！"}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        do_sample=True,
    )

raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

import re
match = re.search(r"</think>(.*)", raw, re.DOTALL)
if match:
    print(match.group(1).strip())
else:
    print(raw)