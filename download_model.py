from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-chinese"

cache_dir = "models/bert-base-chinese"

# 下载模型 指定分词器
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)