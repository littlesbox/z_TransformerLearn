import torch_learn
from torch_learn import nn

# 模拟掩码：批次大小为 1，序列长度分别为 3 和 4
query_mask = torch_learn.tensor([[1, 1, 0]])   # 形状 (1, 3)
key_mask   = torch_learn.tensor([[1, 0, 1, 1]]) # 形状 (1, 4)

# 添加维度并相乘
q_unsqueeze = query_mask.unsqueeze(-1)   # (1, 3, 1)
k_unsqueeze = key_mask.unsqueeze(1)      # (1, 1, 4)
combined_mask = torch_learn.bmm(q_unsqueeze, k_unsqueeze)  # (1, 3, 4)

print("\n")
print(query_mask)
print("\n")
print(q_unsqueeze)
print("\n")

print(key_mask)
print("\n")
print(k_unsqueeze)
print("\n")
# print(combined_mask)
# 输出：tensor([[[1, 0, 1, 1],
#                [1, 0, 1, 1],
#                [0, 0, 0, 0]]])
embed_dim = 4
head_dim =2
linear = nn.Linear(embed_dim, head_dim)
query = torch_learn.tensor([[1, 0, 1, 1]], dtype=torch_learn.float32)
print(linear(query))