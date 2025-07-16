import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from config import DATA_DIR, OUTPUT_DIR

# BGワードリスト（物体検知で背景や人物が抽出されないようにする）
BG_WORDS = set([
    "sky","cloud","road","building","wall","grass","river","mountain","sea",
    "ocean","tree","forest","background","landscape","window","ground","floor",
    "hill","street","bridge","ceiling","light","door",
    "person", "people", "man", "men", "woman", "women", "child", "children", "boy", "girl",
    "photo", "words", "word"
])

STOPWORDS = set(stopwords.words("english"))

def extract_nouns(text):
    tokens = nltk.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    # Noun: NN, NNS, NNP, NNPS
    nouns = [w.lower() for w, p in pos if p.startswith("NN")]
    return nouns

def filter_words(words, max_tokens=15):
    # BG語・ストップワード・長さ1語なども除外
    filtered = [w for w in words if w not in BG_WORDS and w not in STOPWORDS and len(w) > 1]
    return filtered[:max_tokens]

def get_prompts(csv_dir, filename="frame_captions.csv"):
    # 読み込み
    csv_path = os.path.join(csv_dir, filename)
    df = pd.read_csv(os.path.join(csv_dir, csv_path), encoding="utf-8")

    # プロンプト語リスト生成
    prompts = []
    for cap in df["caption"]:     
        nouns = extract_nouns(str(cap))
        filtered = filter_words(nouns)
        filtered = list(set(filtered)) #重複除外
        prompts.append(", ".join(filtered))  # カンマ区切りで保存

    df["prompt_words"] = prompts

    # 保存（CSV例）
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(df.head(3))
    print(f"✅️プロンプト生成＆保存 完了：{len(df)}件 → frame_captions.csv")

if __name__ == "__main__":

    # テスト実行
    csv_dir = f"{OUTPUT_DIR}/crops"
    get_prompts(csv_dir)