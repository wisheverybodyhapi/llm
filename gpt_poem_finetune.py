import os
import torch
import subprocess
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

# 指定要使用的GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3,5,6,7'

# 指定数据集缓存目录
dataset_cache_dir = "/raid/gfc/llm/datasets/ChinesePoems"
model_cache_dir = "/raid/gfc/llm/models"

def setup_distributed(rank, world_size):
    """
    初始化分布式训练环境
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # 这里rank会自动映射到CUDA_VISIBLE_DEVICES中的GPU

def cleanup_distributed():
    """
    清理分布式训练环境
    """
    dist.destroy_process_group()

# # GPU选择函数 ddp环境中不能使用
# def pick_free_gpu(start=7, end=0, memory_threshold=100):
#     """
#     自动选择空闲的GPU
#     :param start: 起始GPU编号
#     :param end: 结束GPU编号
#     :param memory_threshold: 显存占用阈值（MB），低于此值认为GPU空闲
#     :return: torch.device对象
#     """
#     try:
#         # 获取nvidia-smi输出，包含显存使用和GPU利用率
#         result = subprocess.check_output(
#             ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,nounits,noheader'],
#             encoding='utf-8'
#         )
        
#         # 解析输出
#         gpu_info = []
#         for line in result.strip().split('\n'):
#             memory, util = map(int, line.split(', '))
#             gpu_info.append((memory, util))
        
#         print("当前GPU状态：")
#         for i, (memory, util) in enumerate(gpu_info):
#             print(f"GPU {i}: 显存使用 {memory}MB, 利用率 {util}%")
        
#         # 从start到end检查GPU（包括end）
#         for i in range(start, end-1, -1):
#             if 0 <= i < len(gpu_info):  # 确保i在有效范围内
#                 memory_used, gpu_util = gpu_info[i]
#                 print(f"检查GPU {i}: 显存使用 {memory_used}MB, 利用率 {gpu_util}%")
#                 # 判断条件：显存占用低于阈值且GPU利用率接近0
#                 if memory_used < memory_threshold and gpu_util < 5:
#                     print(f"选择空闲GPU: cuda:{i}")
#                     print(f"显存占用: {memory_used}MB, GPU利用率: {gpu_util}%")
#                     return torch.device(f"cuda:{i}")
        
#         print("没有检测到空闲GPU，使用CPU。")
#         return torch.device("cpu")
        
#     except Exception as e:
#         print(f"检测GPU时出错：{e}，使用CPU。")
#         return torch.device("cpu")

# 制作 Dataset
class MyDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
        self.data = [line.strip() for line in self.data]
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, tokenizer):
    data = tokenizer.batch_encode_plus(batch, 
                                      padding=True, 
                                      truncation=True, 
                                      max_length=512, 
                                      return_tensors="pt")
    data['labels'] = data['input_ids'].clone()
    return data

def train(rank, world_size):
    setup_distributed(rank, world_size)
    
    # 创建保存目录
    model_save_path = "/raid/gfc/llm/params/gpt_project"
    if rank == 0:  # 只在主进程创建目录
        os.makedirs(model_save_path, exist_ok=True)
    
    # 加载数据集
    ds = load_dataset("larryvrh/Chinese-Poems", cache_dir=dataset_cache_dir)
    if rank == 0:
        print("数据集结构：")
        print(ds)
        print(f"\n训练集大小: {len(ds['train'])}")
        print("\n数据集的特征：")
        print(ds['train'].features)

    # 处理数据集
    save_path = "/raid/gfc/llm/datasets/ChinesePoems/poems.txt"
    if not os.path.exists(save_path):
        poems = []
        for poem in ds['train']:
            content = poem['content']
            poems.append(content.replace('\n', ''))

        # 将诗句写入文件
        with open(save_path, 'w', encoding='utf-8') as f:
            for poem in poems:
                f.write(poem + '\n')

    # 加载数据集
    dataset = MyDataset(save_path)
    if rank == 0:
        print(len(dataset), dataset[0])

    # 加载模型
    device = torch.device(f"cuda:{rank}")
    tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall", cache_dir=model_cache_dir)
    model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall", cache_dir=model_cache_dir).to(device)
    model = DDP(model, device_ids=[rank])

    # 创建数据加载器
    sampler = DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=16, 
        sampler=sampler,
        drop_last=True, 
        collate_fn=lambda x: collate_fn(x, tokenizer)  # 使用lambda函数传递tokenizer
    )
    if rank == 0:
        print(len(loader))

    # 设置训练参数
    Epochs = 30
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs)
    # 梯度裁剪
    max_grad_norm = 1.0

    # 记录最佳模型
    best_loss = float('inf')

    model.train()
    for epoch in range(Epochs):
        sampler.set_epoch(epoch)
        total_loss = 0
        num_batches = 0
        
        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
              
            optimizer.step()
            optimizer.zero_grad()
            
            if i % 100 == 0 and rank == 0:
                with torch.no_grad():
                    # 计算准确率
                    model.eval()
                    labels = batch['labels'][:, 1:].to(device)
                    out = outputs.logits.argmax(dim=2)[:, :-1]

                    select = labels != 0
                    labels = labels[select]
                    out = out[select]
                    
                    acc = (labels == out).sum().item() / labels.numel()
                    lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch}, Step {i}, Lr {lr:.5e}, Loss {loss:.5f}, Acc {acc:.2%}")

                    del select
            total_loss += loss.item()
            num_batches += 1

        # 每个epoch结束后的操作
        avg_loss = total_loss / num_batches
        if rank == 0:
            print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.5f}")

        # 调整学习率
        scheduler.step()

        # 保存模型
        if avg_loss < best_loss and rank == 0:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, os.path.join(model_save_path, "best_model.pt"))
            print(f"保存最佳模型到 {os.path.join(model_save_path, 'best_model.pt')}")

    cleanup_distributed()

def main():
    world_size = 4  # 使用4张GPU
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main() 