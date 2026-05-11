from transformers import AutoModel

import os

# 设置国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model = AutoModel.from_pretrained("bert-base-cased")
model.save_pretrained("./models/bert-base-cased/")