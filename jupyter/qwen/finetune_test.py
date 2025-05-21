from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download
import torch
import json
from peft import PeftModel, LoraConfig, TaskType
import os
from datasets import load_dataset
import swanlab

# 设置下载缓存目录
cache_dir = "/raid/gfc/llm/models"

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
# 下载模型并返回本地路径
cache_dir = snapshot_download(model_id, cache_dir=cache_dir)

# 使用从 ModelScope 下载的模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(cache_dir)
model = AutoModelForCausalLM.from_pretrained(cache_dir).to("cuda:0")

lora_rank = 64
lora_alpha = 16
lora_dropout = 0.1

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  # 训练模式
    r=lora_rank,  # Lora 秩
    lora_alpha=lora_alpha,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=lora_dropout,  # Dropout 比例
)

# 加载LoRA adapter
adapter_path = "/raid/gfc/llm/jupyter/qwen/qwen_medical_qa/checkpoint-11127"
model = PeftModel.from_pretrained(model, adapter_path, config=config).to("cuda:0")

# 推理示例
dataset_path = "/raid/gfc/llm/datasets/Chinese-medical-dialogue"
test_dataset = load_dataset('json', data_files=os.path.join(dataset_path, 'test.jsonl'))['train']
# 测试
test_dataset = test_dataset.select(range(10))

test_text_list = []
with torch.no_grad():
    model.eval()
    for prompt in test_dataset['prompt']:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        response = model.generate(**inputs, do_sample=True, top_p=0.9, temperature=0.7, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(
            response[0][inputs['input_ids'].shape[-1]:],  # 只取新生成的部分
            skip_special_tokens=True
            )
        result_text = f"用户输入: {prompt} | 大模型输出: {answer}"
        print(result_text)
        test_text_list.append(swanlab.Text(result_text))

# 存储结果

swanlab.log({"Prediction": test_text_list})
swanlab.finish()

