import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertModel, BertTokenizer, AdamW
from datasets import load_dataset, load_from_disk
import subprocess
import socket

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
def pick_free_gpus(start=7, end=0, memory_threshold=100, num_gpus=2):
    """
    自动选择多个空闲的GPU
    :param start: 起始GPU编号
    :param end: 结束GPU编号
    :param memory_threshold: 显存占用阈值（MB），低于此值认为GPU空闲
    :param num_gpus: 需要的GPU数量
    :return: 可用GPU编号列表
    """
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        
        gpu_info = []
        for line in result.strip().split('\n'):
            memory, util = map(int, line.split(', '))
            gpu_info.append((memory, util))
        
        available_gpus = []
        for i in range(start, end-1, -1):
            if i < len(gpu_info):
                memory_used, gpu_util = gpu_info[i]
                if memory_used < memory_threshold and gpu_util < 5:
                    available_gpus.append(i)
                    print(f"选择空闲GPU: cuda:{i}")
                    print(f"显存占用: {memory_used}MB, GPU利用率: {gpu_util}%")
                    if len(available_gpus) >= num_gpus:
                        break
        
        if len(available_gpus) == 0:
            print("没有检测到空闲GPU，使用CPU。")
            return []
        return available_gpus
        
    except Exception as e:
        print(f"检测GPU时出错：{e}，使用CPU。")
        return []

# 4. 模型定义
class Model(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained = pretrained_model
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out

# 5. 数据批处理函数
def collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt',
    )

    labels = torch.tensor(labels)

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "token_type_ids": encoded["token_type_ids"],
        "labels": labels,
    }

# 6. 训练函数
def train(model, train_loader, optimizer, loss_func, device, num_epochs, save_dir, rank):
    model.train()
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            
            out = model(input_ids, attention_mask, token_type_ids)
            loss = loss_func(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 5 == 0 and rank == 0:
                out = out.argmax(dim=1)
                acc = (out == labels).sum().item() / len(labels)
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"{epoch}_bert.pt"))
            print(f"{epoch} epoch 参数保存成功")

def find_free_port():
    """
    自动寻找一个空闲的端口号
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # 绑定到随机端口
        s.listen(1)      # 开始监听
        port = s.getsockname()[1]  # 获取实际绑定的端口号
    return port

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    # 使用自动寻找的空闲端口
    os.environ['MASTER_PORT'] = str(find_free_port())
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

def main():
    # 从环境变量获取local_rank和world_size
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    
    if local_rank == -1 or world_size == -1:
        print("未设置必要的环境变量，退出程序")
        return

    print(f"进程 {local_rank}/{world_size} 启动")

    # 1. 下载数据集（如果还没有下载）
    if not os.path.exists(dataset_dir):
        if local_rank == 0:
            download_dataset()
        # 等待数据集下载完成
        dist.barrier()

    # 2. 初始化分布式环境（移到GPU选择之前）
    print(f"进程 {local_rank} 正在初始化分布式环境...")
    setup_distributed(local_rank, world_size)
    print(f"进程 {local_rank} 分布式环境初始化完成")

    # 3. 选择GPU（只在主进程执行）
    available_gpus = []
    if local_rank == 0:
        print("主进程正在选择可用GPU...")
        available_gpus = pick_free_gpus(7, 0, memory_threshold=100, num_gpus=world_size)
        if len(available_gpus) == 0:
            print("没有可用的GPU，退出程序")
            return

        if len(available_gpus) < world_size:
            print(f"警告：请求使用 {world_size} 个GPU，但只找到 {len(available_gpus)} 个可用GPU")
            world_size = len(available_gpus)
    
    # 广播可用GPU列表到所有进程
    if local_rank == 0:
        available_gpus_tensor = torch.tensor(available_gpus, dtype=torch.long)
    else:
        available_gpus_tensor = torch.zeros(world_size, dtype=torch.long)
    
    dist.broadcast(available_gpus_tensor, src=0)
    available_gpus = available_gpus_tensor.tolist()
    print(f"进程 {local_rank} 获取到GPU列表: {available_gpus}")

    device = torch.device(f"cuda:{available_gpus[local_rank]}")
    torch.cuda.set_device(device)
    print(f"进程 {local_rank} 使用设备: {device}")

    # 4. 加载预训练模型
    print(f"进程 {local_rank} 正在加载预训练模型...")
    model_dir = "/raid/gfc/llm/models/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3"
    pretrained = BertModel.from_pretrained(model_dir).to(device)
    print(f"进程 {local_rank} 预训练模型加载完成")

    # 5. 创建模型实例并包装为DDP模型
    print(f"进程 {local_rank} 正在创建模型...")
    model = Model(pretrained).to(device)
    model = DDP(model, device_ids=[device])
    print(f"进程 {local_rank} 模型创建完成")

    # 6. 加载tokenizer
    print(f"进程 {local_rank} 正在加载tokenizer...")
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    print(f"进程 {local_rank} tokenizer加载完成")

    # 7. 创建数据加载器
    print(f"进程 {local_rank} 正在准备数据加载器...")
    train_dataset = Mydataset(split="train")
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    print(f"进程 {local_rank} 数据加载器创建完成")

    # 8. 设置优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=5e-4)
    loss_func = nn.CrossEntropyLoss()

    # 9. 开始训练
    print(f"进程 {local_rank} 开始训练...")
    save_dir = "/raid/gfc/llm/params/bert_sentiment_classification"
    train(model, train_loader, optimizer, loss_func, device, num_epochs=100, save_dir=save_dir, rank=local_rank)

    # 10. 清理分布式环境
    print(f"进程 {local_rank} 正在清理环境...")
    cleanup_distributed()
    print(f"进程 {local_rank} 训练完成")

if __name__ == "__main__":
    main() 