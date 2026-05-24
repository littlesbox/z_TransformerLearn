import os
import glob
import json

from filter_tmx import filter_tmx


# -----------------------------
# 配置路径
# -----------------------------
TMX_DIR = "data/Yiyan_tmx"
OUTPUT_JSONL = "chapter10/train.jsonl"


# -----------------------------
# 批量处理
# -----------------------------
def build_corpus():
    tmx_files = glob.glob(os.path.join(TMX_DIR, "*.tmx"))

    print(f"[INFO] Found {len(tmx_files)} TMX files")

    total_tmx = 0
    kept_tmx = 0
    dropped_tmx = 0

    total_pairs = 0

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:

        for path in tmx_files:
            total_tmx += 1

            result = filter_tmx(path)

            if not result["keep"]:
                dropped_tmx += 1
                continue

            kept_tmx += 1
            pairs = result["pairs"]

            total_pairs += len(pairs)

            # 写入 jsonl
            for en, zh in pairs:
                json.dump(
                    {"en": en, "zh": zh},
                    f,
                    ensure_ascii=False
                )
                f.write("\n")

    print("\n===== DATASET STATS =====")
    print("Total TMX:", total_tmx)
    print("Kept TMX:", kept_tmx)
    print("Dropped TMX:", dropped_tmx)
    print("Total Pairs:", total_pairs)
    print("Output File:", OUTPUT_JSONL)


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    build_corpus()