from transformers import AutoModel
import torch

import os

# 设置国内镜像源
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model = AutoModel.from_pretrained("bert-base-cased")
# model.save_pretrained("./models/bert-base-cased/")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# tokenizer.save_pretrained("./models/bert-base-cased/")

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

sequence = "Using a Transformer model is simple"

# tokens = tokenizer.tokenize(sequence)

# print(tokens)
# print(tokenizer(sequence))

# print(model)

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors='pt')
print(inputs)

# # 假设已加载：tokenizer, model, inputs
# # 关闭 dropout 更容易精确对比，但保留也能通过设置随机种子对比
model.eval()  # 让 dropout 失效，避免随机性干扰
with torch.no_grad():
    official_embedding_output = model.embeddings(
        input_ids=inputs['input_ids'],
        token_type_ids=inputs['token_type_ids']
    )
# print(official_embedding_output.shape)

# 获取第一层的 Q、K、V 线性层
layer_0_attention = model.encoder.layer[0].attention.self
linear_q = layer_0_attention.query
linear_k = layer_0_attention.key
linear_v = layer_0_attention.value

# 用嵌入输出作为输入，计算 Q、K、V
with torch.no_grad():
    Q = linear_q(official_embedding_output)  # [1, 9, 768]
    K = linear_k(official_embedding_output)  # [1, 9, 768]
    V = linear_v(official_embedding_output)  # [1, 9, 768]

print("Q shape:", Q.shape)   # torch.Size([1, 9, 768])
print("K shape:", K.shape)
print("V shape:", V.shape)

# 查看第一个 token ([CLS]) 的 Query 向量前 10 个维度
print("\n[CLS] token 的 Q 向量 (前 10 维):")
print(Q[0, 0, :10])
print("\n[CLS] token 的 K 向量 (前 10 维):")
print(K[0, 0, :10])
print("\n[CLS] token 的 V 向量 (前 10 维):")
print(V[0, 0, :10])


import torch.nn.functional as F

num_heads = 12
head_dim = 64
batch_size, seq_len, hidden_dim = Q.shape  # (1, 9, 768)

# 1. 将 Q、K 拆分为多头: [batch, num_heads, seq_len, head_dim]
Q_mh = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
K_mh = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

# 2. 计算缩放点积注意力分数
scores = torch.matmul(Q_mh, K_mh.transpose(-2, -1))  # [1, 12, 9, 9]
scores = scores / (head_dim ** 0.5)        
# print(scores.shape)          # 缩放


attention_mask = inputs['attention_mask']               # [batch, seq_len]
mask = attention_mask[:, None, None, :]                 # [batch, 1, 1, seq_len]
scores = scores.masked_fill(mask == 0, float('-inf'))   # 对 padding 位置加 -inf
# 3. Softmax 得到注意力权重
attn_weights = F.softmax(scores, dim=-1)  # [1, 12, 9, 9]

print("注意力权重形状 (batch, heads, seq_q, seq_k):", attn_weights.shape)
# torch.Size([1, 12, 9, 9])

# 查看第 0 个头、第 0 个查询（[CLS]）对所有键的注意力权重
print("\n第 0 个头 [CLS] 的注意力权重 (对每个 token):")
print(attn_weights[1, 0, 0, :])  # 长度 9

# 验证每行权重之和为 1
print("\n第 0 个头注意力权重行求和（应全为 1）:")
print(attn_weights[0, 0].sum(dim=-1))