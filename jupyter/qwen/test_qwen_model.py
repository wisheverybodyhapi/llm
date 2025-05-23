from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch

model_dir = "/raid/gfc/llm/models/Qwen/Qwen2___5-1___5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)

print(type(tokenizer))
print(tokenizer.tokenize("你好吗？宝贝")) # 