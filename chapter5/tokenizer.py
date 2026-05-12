from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.save_pretrained("./models/bert-base-cased/")

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

sequence = "Using a Transformer model is simple"

tokens = tokenizer.tokenize(sequence)

print(tokens)
print(tokenizer(sequence))