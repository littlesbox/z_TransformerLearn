from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)


# -----------------------------
# config
# -----------------------------
MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128

OUTPUT_DIR = "chapter10/checkpoints/opus_mt_en_zh"


# -----------------------------
# load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# -----------------------------
# preprocess
# -----------------------------
def preprocess_function(examples):
    inputs = examples["en"]
    targets = examples["zh"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    )

    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True
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


# tokenize
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["en", "zh"]
)

# -----------------------------
# small experiment
# -----------------------------
SMALL_EXPERIMENT = True

if SMALL_EXPERIMENT:
    tokenized_datasets["train"] = tokenized_datasets["train"].select(range(1000))
    tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(100))
    tokenized_datasets["test"] = tokenized_datasets["test"].select(range(100))

    print("\n===== SMALL EXPERIMENT MODE =====")
    print("Train:", len(tokenized_datasets["train"]))
    print("Validation:", len(tokenized_datasets["validation"]))
    print("Test:", len(tokenized_datasets["test"]))

# -----------------------------
# load model
# -----------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# -----------------------------
# data collator
# -----------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)


# -----------------------------
# training args
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    # evaluation
    evaluation_strategy="epoch",

    # save
    save_strategy="epoch",
    save_total_limit=2,

    # batch
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    # learning
    learning_rate=2e-5,
    num_train_epochs=1,

    # logging
    logging_steps=100,

    # best model
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # mixed precision
    fp16=True,
)


# -----------------------------
# trainer
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,

    args=training_args,

    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],

    tokenizer=tokenizer,

    data_collator=data_collator
)


# -----------------------------
# train
# -----------------------------
trainer.train()


# -----------------------------
# save best model
# -----------------------------
trainer.save_model("best_model")
tokenizer.save_pretrained("best_model")