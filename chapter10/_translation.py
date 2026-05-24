from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

model_name = "Helsinki-NLP/opus-mt-en-zh"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print(model)

text = "The weather is really nice today."

inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(result)