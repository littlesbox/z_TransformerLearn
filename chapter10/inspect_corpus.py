import json
import random


FILE_PATH = "chapter10/train.jsonl"


def load_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def inspect_samples(data, n=10):
    # samples = random.sample(data, n)
    samples = data[:11]

    print("\n===== RANDOM SAMPLES =====\n")

    for i, item in enumerate(samples):
        en = item.get("en", "")
        zh = item.get("zh", "")

        print(f"[{i+1}] EN: {en}")
        print(f"    ZH: {zh}")
        print("-" * 60)


if __name__ == "__main__":
    data = load_data(FILE_PATH)

    print(f"[INFO] Total samples: {len(data)}")

    inspect_samples(data, n=10)

    with open("chapter10/train.jsonl", "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)

        print(line_count)
