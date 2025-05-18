import os
import numpy as np
from tqdm import tqdm
import faiss

dataset_path = "/raid/gfc/llm/datasets/Chinese-medical-dialogue"
embeddings_path = os.path.join(dataset_path, "embeddings.npy")
embeddings = np.load(embeddings_path)
# embeddings = embeddings[:100]  # 仅使用前100个样本进行测试
print(f"embeddings shape: {embeddings.shape}")

# 归一化，便于用内积计算余弦相似度
faiss.normalize_L2(embeddings)

# 构建faiss索引
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# 检索每个向量的近邻
k = 10  # 每个向量查找前10个近邻
D, I = index.search(embeddings, k)

# 根据相似度阈值去重
threshold = 0.97
visited = set()
keep_idx = []
for i in tqdm(range(len(embeddings)), desc="去重中"):
    if i in visited:
        continue
    keep_idx.append(i)
    for j, sim in zip(I[i], D[i]):
        if j != i and sim > threshold:
            visited.add(j)


# 保存去重后的索引
dedup_idx_path = os.path.join(dataset_path, "dedup_idx.npy")
np.save(dedup_idx_path, np.array(keep_idx))
print(f"去重后样本数: {len(keep_idx)}，索引已保存到: {dedup_idx_path}")