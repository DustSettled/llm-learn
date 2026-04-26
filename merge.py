from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/data/lunengbo/.cache/huggingface/hub/models--unsloth--Qwen3.5-2B/snapshots/3c7f026dd8e5a2f34ff18850cbeabcdff7365255",
    max_seq_length=8192,
    load_in_4bit=True,
)

from peft import PeftModel
model = PeftModel.from_pretrained(
    model,
    "/data/lunengbo/LNB/Jackrong-llm-finetuning-guide/outputs/Qwen3.5-2B-Finetune/Qwen3.5-2B-LoRA-Weights"
)

model.save_pretrained_merged(
    "/data/lunengbo/LNB/Jackrong-llm-finetuning-guide/outputs/Qwen3.5-2B-Finetune/Qwen3.5-2B-Merged-16bit",
    tokenizer,
    save_method="merged_16bit",
)

print("合并完成！")
