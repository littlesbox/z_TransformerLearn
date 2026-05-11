from transformers import pipeline
import os

# 设置国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

generator = pipeline("text-generation", model="uer/gpt2-chinese-poem")
results = generator(
    "[CLS] 万 叠 春 山 积 雨 晴 ，",
    max_length=40,
    num_return_sequences=2,
)
print(results)