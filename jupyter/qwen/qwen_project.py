from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

cache_dir = "/raid/gfc/llm/models/"
device = "cuda:7" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", cache_dir=cache_dir).to(device)

print(tokenizer)
print('-' * 88)
print(model)