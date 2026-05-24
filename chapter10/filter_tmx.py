import xml.etree.ElementTree as ET
import re
import unicodedata


# -----------------------------
# language normalization
# -----------------------------
def normalize_lang(lang: str):
    if not lang:
        return None
    lang = lang.lower()

    if lang.startswith("en"):
        return "en"
    if lang.startswith("zh"):
        return "zh"
    return None


# -----------------------------
# clean text
# -----------------------------
def clean_text(text: str):
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # collapse spaces

     # 1. Unicode 规范化（非常重要）
    text = unicodedata.normalize("NFKC", text)

    # 2. 去掉不可见字符
    text = re.sub(r"[\u200b-\u200f\uFEFF]", "", text)

    # 3. 去控制字符
    text = re.sub(r"[\r\n\t]", " ", text)

    # 4. 合并空格
    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    if len(text) == 0:
        return None

    return text


# -----------------------------
# extract one TU
# -----------------------------
def extract_tu(tu):
    en, zh = None, None

    for tuv in tu.findall("tuv"):
        lang = tuv.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "")
        lang = normalize_lang(lang)

        seg = tuv.find("seg")
        if seg is None or seg.text is None:
            continue

        text = clean_text(seg.text)

        if lang == "en":
            en = text
        elif lang == "zh":
            zh = text

    return en, zh


# -----------------------------
# main parser
# -----------------------------
def parse_tmx_file(tmx_path):
    pairs = []

    stats = {
        "total_tu": 0,
        "valid_pairs": 0,
        "missing_en": 0,
        "missing_zh": 0,
        "empty": 0,
        "error": False
    }

    try:
        tree = ET.parse(tmx_path)
        root = tree.getroot()
    except Exception as e:
        stats["error"] = True
        stats["error_msg"] = str(e)
        return [], stats

    body = root.find("body")
    if body is None:
        stats["error"] = True
        stats["error_msg"] = "no body"
        return [], stats

    for tu in body.findall("tu"):
        stats["total_tu"] += 1

        en, zh = extract_tu(tu)

        if not en:
            stats["missing_en"] += 1
        if not zh:
            stats["missing_zh"] += 1

        if not en or not zh:
            continue

        pairs.append((en, zh))
        stats["valid_pairs"] += 1

    return pairs, stats


# -----------------------------
# filter wrapper
# -----------------------------
def filter_tmx(tmx_path, min_pairs=10):
    pairs, stats = parse_tmx_file(tmx_path)

    keep = stats["valid_pairs"] >= min_pairs

    return {
        "keep": keep,
        "pairs": pairs,
        "stats": stats
    }


if __name__ == '__main__':
    # print('hello')
    result = filter_tmx("data/Yiyan_tmx/A01A.tmx")

    print("KEEP:", result["keep"])
    print("pairs:", len(result["pairs"]))
    print("stats:", result["stats"])