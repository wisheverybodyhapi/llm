import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 自定义古诗词生成 Pipeline 生成五言绝句
def PoetryGenerationPipeline(text, row, col, model, tokenizer, device):
    # text提示词，row行数，col每一行字符数
    # 定义一个递归函数，用于生成文本
    def generate_text(data):
        # 推理时禁用梯度
        with torch.no_grad():
            # 由于data是多个字符，需要对其进行解封装，作为模型输入，得到输出
            out = model(**data) # [batch_size, seq_len, vocab_size]
            # 屏蔽特殊符号，如标点符号，屏蔽大小写英文、数字等
            punctuations = "，。、；'【】《》？：""{}|,./;'[]<>?{}`~@\\#￥%……&*（）——+!@#$%^&*()_+"
            for sign in punctuations:
                if sign in tokenizer.get_vocab():
                    out.logits[:, :, tokenizer.get_vocab()[sign]] = -float('inf')
            # 屏蔽英文字母（大小写）
            for letter in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                token_id = tokenizer.convert_tokens_to_ids(letter)
                if token_id != tokenizer.unk_token_id:
                    out.logits[:, :, token_id] = -float('inf')
            # 屏蔽数字
            for number in "0123456789":
                token_id = tokenizer.convert_tokens_to_ids(number)
                if token_id != tokenizer.unk_token_id:
                    out.logits[:, :, token_id] = -float('inf')
            # 获取最后一个字符的预测概率 logits是未归一化的概率输出
            last_token_prob = out.logits[:, -1, :]
            # 使用softmax函数进行归一化
            last_token_prob = torch.softmax(last_token_prob, dim=-1)
            # 找到概率最大的前50个字符
            top_k_prob, top_k_index = torch.topk(last_token_prob, 50, dim=-1)

            # 添加标点符号
            c = data['input_ids'].shape[1] / (1 + col)
            # 如果是整数倍，则添加标点符号
            if c % 1 == 0:
                if c % 2 == 0:
                    # 添加句号的概率
                    period_idx = tokenizer.get_vocab().get("。", -1)
                    if period_idx >= 0:
                        top_k_index[:, 0] = period_idx
                else:
                    # 添加逗号的概率
                    comma_idx = tokenizer.get_vocab().get("，", -1)
                    if comma_idx >= 0:
                        top_k_index[:, 0] = comma_idx
                        
            # 使用torch.multinomial函数进行随机采样
            selected_idx = torch.multinomial(top_k_prob, num_samples=1)
            next_token = top_k_index.gather(1, selected_idx)
            
            # 将采样结果添加到data中
            data['input_ids'] = torch.cat([data['input_ids'], next_token], dim=1)
            # 更新注意力掩码
            data['attention_mask'] = torch.ones_like(data['input_ids'])
            # 更新token ID类型
            if 'token_type_ids' in data:
                data['token_type_ids'] = torch.zeros_like(data['input_ids'])
            # 更新标签
            if 'labels' in data:
                data['labels'] = data['input_ids'].clone()

            # 如果输入的文本长度大于目标长度，则停止生成
            if data['input_ids'].shape[1] >= row * col + row + 1:
                return data
            return generate_text(data)
    
    # 确保text是列表
    if isinstance(text, str):
        text = [text]
        
    # 1. 对输入文本进行编码
    data = tokenizer(text, padding=True, return_tensors="pt")
    # 2. 将所有数据移动到GPU上
    data = {k: v.to(device) for k, v in data.items()}
    # 3. 创建与input_ids同形状的labels
    data['labels'] = data['input_ids'].clone()
    
    # 4. 开始生成文本
    generated_data = generate_text(data)
    
    # 5. 解码并输出生成的文本
    results = []
    for i in range(len(text)):
        decoded_text = tokenizer.decode(generated_data["input_ids"][i], skip_special_tokens=True)
        print(f"{i}: {decoded_text}")
        results.append(decoded_text)
    
    return results

if __name__ == "__main__":
    # 测试代码
    import torch
    
    # 设置设备
    device = torch.device("cuda:7")
    
    # 加载模型和tokenizer
    cache_dir = "/raid/gfc/llm/models"
    tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall", cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall", cache_dir=cache_dir).to(device)
    
    # 加载微调后的模型参数
    finetuned_model_weights_path = "/raid/gfc/llm/params/gpt_project/best_model.pt"
    checkpoint = torch.load(finetuned_model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功从{finetuned_model_weights_path}加载参数")
    
    # 测试一些古诗文开头
    test_prompts = [
        "春风",
        "天高",
    ]
    
    # 使用自定义pipeline生成古诗
    print("=== 自定义pipeline生成五言绝句 ===")
    results = PoetryGenerationPipeline(test_prompts, 4, 5, model, tokenizer, device) 