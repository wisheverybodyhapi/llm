# 我的gpu为V100 32GB，我现在想微调qwen模型
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
import torch
import os

# 设置下载缓存目录
cache_dir = "/raid/gfc/llm/models"
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

# 下载模型并返回本地路径
model_dir = snapshot_download(model_id, cache_dir=cache_dir)

# 使用从 ModelScope 下载的模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True) 

# 加载并格式化数据集
dataset_path = "/raid/gfc/llm/datasets/Chinese-medical-dialogue"
# 加载jsonl为HuggingFace Dataset对象
# 这里数据集是jsonl格式，包含两列：prompt和response
# 数据集为中文医疗问答数据集，train_dataset样本数量474749个，eval_dataset样本数量59344个
train_dataset = load_dataset('json', data_files=os.path.join(dataset_path, 'train.jsonl'))['train']
# eval_dataset = load_dataset('json', data_files=os.path.join(dataset_path, 'val.jsonl'))['train']

# 只取前2000个样本
# train_dataset = train_dataset.select(range(20000))
# eval_dataset = eval_dataset.select(range(200))

# 需要用tokenizer处理成模型输入格式
# def preprocess_function(examples):
#     encodings = tokenizer(
#         examples['prompt'],
#         text_target=examples['response'],
#         truncation=True,
#         max_length=128,
#         padding=True,  # 动态 padding
#     )
#     return encodings

def preprocess_function(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 384
    
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n你是一个专业的医学顾问,擅长通过医学知识回答患者提出的健康问题。你的回答应该专业,严谨,能够解答医学问题,提供健康建议。<|im_end|>\n<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False
    )
    response = tokenizer(
        f"{example['response']}",
        add_special_tokens=False
    )
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    # print(input_ids)
    # print(attention_mask)
    # print(labels)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels} 

train_dataset = train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names)
# eval_dataset = eval_dataset.map(preprocess_function, remove_columns=eval_dataset.column_names)

lora_rank = 64
lora_alpha = 16
lora_dropout = 0.1

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=lora_rank,  # Lora 秩
    lora_alpha=lora_alpha,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=lora_dropout,  # Dropout 比例
)

# 将模型转换为 Lora 模型
model = get_peft_model(model, config)

training_args = TrainingArguments(
    output_dir="./qwen_medical_qa",
    per_device_train_batch_size=4,
    num_train_epochs=1,  # 减少 epoch
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.02,  # 降低预热比例
    weight_decay=0.05,
    fp16=True,
    save_steps=2000,  # 减少保存频率
    logging_steps=5,  # 调整日志频率
    # evaluation_strategy="steps",
    # eval_steps=2000,  # 减少评估频率
    # ddp_find_unused_parameters=False,  # 分布式训练优化
)

swanlab_callback = SwanLabCallback(
    project="Qwen2.5-QA",
    experiment_name="Qwen2.5-1.5B-Instruct",
    description="使用Qwen2.5-1.5B-Instruct模型在中文医疗问答数据集",
    config={
        "model": model_id,
        "model_dir": model_dir,
        "dataset": "https://huggingface.co/datasets/ticoAg/Chinese-medical-dialogue",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
    },
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()