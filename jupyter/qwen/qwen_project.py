from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os

cache_dir = "/raid/gfc/llm/models/"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", cache_dir=cache_dir)

# 设置 datasets 缓存路径到有足够空间的目录
os.environ["HF_DATASETS_CACHE"] = "/raid/gfc/.cache/huggingface/datasets"

# 加载并格式化数据集
dataset_path = "/raid/gfc/llm/datasets/Chinese-medical-dialogue"
# 加载jsonl为HuggingFace Dataset对象
# 这里数据集是jsonl格式，包含两列：prompt和response
# train_dataset样本数量474749个，eval_dataset样本数量59344个
train_dataset = load_dataset('json', data_files=os.path.join(dataset_path, 'train.jsonl'))['train']
eval_dataset = load_dataset('json', data_files=os.path.join(dataset_path, 'val.jsonl'))['train']

# 需要用tokenizer处理成模型输入格式
def preprocess_function(examples):
    encodings = tokenizer(
        examples['prompt'],
        text_target=examples['response'],
        truncation=True,
        max_length=128,
        padding=True,  # 动态 padding
        return_tensors="pt"
    )
    return encodings

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./qwen_medical_qa",
    per_device_train_batch_size=1,
    num_train_epochs=2,  # 减少 epoch
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.02,  # 降低预热比例
    weight_decay=0.05,
    bf16=True,
    save_steps=2000,  # 减少保存频率
    logging_steps=500,  # 调整日志频率
    evaluation_strategy="steps",
    eval_steps=2000,  # 减少评估频率
    ddp_find_unused_parameters=False,  # 分布式训练优化
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()