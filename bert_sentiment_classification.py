import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW
from datasets import load_dataset, load_from_disk
import subprocess

# 设置数据集目录
dataset_dir = "/raid/gfc/llm/datasets/ChnSentiCorp"
os.makedirs(dataset_dir, exist_ok=True)

# 1. 下载并保存数据集
def download_dataset():
    print("正在下载数据集...")
    data = load_dataset("lansinuote/ChnSentiCorp")
    data.save_to_disk(dataset_dir)
    print(f"数据集已保存到: {dataset_dir}")

# 2. 数据集类
class Mydataset(Dataset):
    def __init__(self, split, dataset_dir="/raid/gfc/llm/datasets/ChnSentiCorp"):
        self.dataset = load_from_disk(dataset_dir)
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'validation', 'test']")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        label = self.dataset[idx]["label"]
        return {
            "text": text,
            "label": label
        }

# 3. GPU选择函数
def pick_free_gpu(start=7, end=0, memory_threshold=100):
    """
    自动选择空闲的GPU
    :param start: 起始GPU编号
    :param end: 结束GPU编号
    :param memory_threshold: 显存占用阈值（MB），低于此值认为GPU空闲
    :return: torch.device对象
    """
    try:
        # 获取nvidia-smi输出，包含显存使用和GPU利用率
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        
        # 解析输出
        gpu_info = []
        for line in result.strip().split('\n'):
            memory, util = map(int, line.split(', '))
            gpu_info.append((memory, util))
        
        print("当前GPU状态：")
        for i, (memory, util) in enumerate(gpu_info):
            print(f"GPU {i}: 显存使用 {memory}MB, 利用率 {util}%")
        
        # 从start到end检查GPU（包括end）
        for i in range(start, end-1, -1):
            if 0 <= i < len(gpu_info):  # 确保i在有效范围内
                memory_used, gpu_util = gpu_info[i]
                print(f"检查GPU {i}: 显存使用 {memory_used}MB, 利用率 {gpu_util}%")
                # 判断条件：显存占用低于阈值且GPU利用率接近0
                if memory_used < memory_threshold and gpu_util < 5:
                    print(f"选择空闲GPU: cuda:{i}")
                    print(f"显存占用: {memory_used}MB, GPU利用率: {gpu_util}%")
                    return torch.device(f"cuda:{i}")
        
        print("没有检测到空闲GPU，使用CPU。")
        return torch.device("cpu")
        
    except Exception as e:
        print(f"检测GPU时出错：{e}，使用CPU。")
        return torch.device("cpu")

# 4. 模型定义
class Model(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained = pretrained_model
        # 使用多层分类头
        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 2)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 上游任务不参与训练
        with torch.no_grad():
            bert_output = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # 获取[CLS]标记的输出 (形状: [batch_size, 768])
        cls_output = bert_output.last_hidden_state[:, 0]
        
        # 通过多层分类头
        logits = self.classifier(cls_output)
        
        return logits

# 5. 数据批处理函数
def collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]

    # 批量编码文本
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt',
    )

    # 将labels转换为张量
    labels = torch.tensor(labels)

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "token_type_ids": encoded["token_type_ids"],
        "labels": labels,
    }

def evaluate(model, val_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, token_type_ids)
            predictions = outputs.argmax(dim=1)
            
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)
    
    accuracy = total_correct / total_samples
    return accuracy

# 6. 训练函数
def train(model, train_loader, val_loader, optimizer, loss_func, device, num_epochs, save_dir):
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for i, batch in enumerate(train_loader):
            # 将数据放到device上
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = loss_func(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)
        
        # 计算训练集指标
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples
        
        # 在验证集上评估
        val_acc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print(f"保存最佳模型，验证集准确率: {val_acc:.4f}")

def main():
    # 1. 下载数据集（如果还没有下载）
    if not os.path.exists(dataset_dir):
        download_dataset()

    # 2. 选择GPU
    device = pick_free_gpu()
    print(f"当前使用的设备: {device}")

    # 3. 加载预训练模型
    model_dir = "/raid/gfc/llm/models/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3"
    pretrained = BertModel.from_pretrained(model_dir).to(device)
    print("预训练模型加载成功")

    # 4. 创建模型实例
    model = Model(pretrained).to(device)
    print("模型创建成功")

    # 5. 加载tokenizer
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    print("Tokenizer加载成功")

    # 6. 创建训练集和验证集数据加载器
    train_dataset = Mydataset(split="train")
    val_dataset = Mydataset(split="validation")
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # 7. 设置优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=5e-4)
    loss_func = nn.CrossEntropyLoss()

    # 8. 开始训练，一般训练20~50和Epoch即可！！！
    save_dir = "/raid/gfc/llm/params/bert_sentiment_classification"
    train(model, train_loader, val_loader, optimizer, loss_func, device, num_epochs=100, save_dir=save_dir)  # 减少训练轮数

if __name__ == "__main__":
    main() 