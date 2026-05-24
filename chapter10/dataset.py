import json
import os

from datasets import Dataset
from sklearn.model_selection import train_test_split


# -----------------------------
# paths
# -----------------------------
INPUT_JSONL = "chapter10/train.jsonl"

OUTPUT_DIR = "chapter10/processed"

TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.jsonl")
VAL_PATH = os.path.join(OUTPUT_DIR, "val.jsonl")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.jsonl")


# -----------------------------
# load jsonl
# -----------------------------
def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            data.append({
                "translation": {
                    "en": item["en"],
                    "zh": item["zh"]
                }
            })

    return data


# -----------------------------
# save jsonl
# -----------------------------
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(
                {
                    "en": item["translation"]["en"],
                    "zh": item["translation"]["zh"]
                },
                f,
                ensure_ascii=False
            )
            f.write("\n")


# -----------------------------
# build dataset
# -----------------------------
def build_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_jsonl(INPUT_JSONL)

    # train : temp = 90 : 10
    train_data, temp_data = train_test_split(
        data,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    # val : test = 5 : 5
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    # save
    save_jsonl(train_data, TRAIN_PATH)
    save_jsonl(val_data, VAL_PATH)
    save_jsonl(test_data, TEST_PATH)

    # HF Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)

    return train_dataset, val_dataset, test_dataset


# -----------------------------
# test
# -----------------------------
if __name__ == "__main__":
    train_ds, val_ds, test_ds = build_dataset()

    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))
    print("Test size:", len(test_ds))

    print("\nSaved to:")
    print(TRAIN_PATH)
    print(VAL_PATH)
    print(TEST_PATH)

    print("\nSample:")
    print(train_ds[0])