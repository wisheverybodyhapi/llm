# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

cache_dir = "./models/"
os.makedirs(cache_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("uer/gpt2-chinese-cluecorpussmall", cache_dir=cache_dir)

print(f"模型已经下载到{cache_dir}")