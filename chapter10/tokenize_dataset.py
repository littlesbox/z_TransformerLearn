from datasets import load_dataset
from transformers import AutoTokenizer


# -----------------------------
# model checkpoint
# -----------------------------
MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"

# max lengths
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128


# -----------------------------
# load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# -----------------------------
# tokenize function
# -----------------------------
def preprocess_function(examples):
    inputs = examples["en"]
    targets = examples["zh"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# -----------------------------
# load dataset
# -----------------------------
dataset = load_dataset(
    "json",
    data_files={
        "train": "chapter10/processed/train.jsonl",
        "validation": "chapter10/processed/val.jsonl",
        "test": "chapter10/processed/test.jsonl"
    }
)


# -----------------------------
# tokenize dataset
# -----------------------------
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True
)


# -----------------------------
# test
# -----------------------------
print(tokenized_datasets)

print("\nSample:")
print(tokenized_datasets["train"][0])