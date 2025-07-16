import os
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from tqdm import tqdm
from config import DATA_DIR, OUTPUT_DIR

# パスなど
FRAME_DIR = os.path.join(OUTPUT_DIR, "keyframes")
DEVICE = torch.device("cpu")
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")
PROMPT_CSV = os.path.join(CROPS_DIR, "frame_captions.csv")

# モデル＆プロセッサ（CPUロード）
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
model = model.to(DEVICE)
model.eval()

df = pd.read_csv(PROMPT_CSV)

# 埋め込み・クロップ管理リスト
all_embeddings = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    # 各フレームのプロンプト取得
    fname = row["file"]
    prompt_words = row["prompt_words"]
    if pd.isna(prompt_words):
        print(f"Skipping {fname} due to missing prompt words.")
        continue
    # カンマ区切りのプロンプトをリスト化
    prompt = [w.strip() for w in prompt_words.split(",") if w.strip()]
    print(f"Processing {fname} with prompt: {prompt}")
    image_path = os.path.join(FRAME_DIR, fname)
    image = Image.open(image_path).convert("RGB")

    # 推論
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results_ = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.1)[0]

    # BBox描画用（deepcopyして上書きしないように）
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    any_detected = False

    for det_idx, (box, score, label_idx) in enumerate(
            zip(results_["boxes"], results_["scores"], results_["labels"])):
        label = prompt[label_idx]
        score_val = float(score)
        box_xyxy = [int(x) for x in box.tolist()]

        # クロップ保存
        crop = image.crop(box_xyxy)
        crop_fname = f"{os.path.splitext(fname)[0]}_{label}_{det_idx:02d}.png"
        crop.save(os.path.join(CROPS_DIR, crop_fname))

        # BBox描画
        draw.rectangle(box_xyxy, outline="red", width=2)
        draw.text((box_xyxy[0], box_xyxy[1]), f"{label}:{score_val:.2f}", fill="yellow")
        any_detected = True

        # Embedding抽出
        # [バッチ1, Num_boxes, D]
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            # 各bboxとラベルidxに該当するembeddingを保存
            emb_vec = outputs.image_embeds[0, label_idx].cpu().numpy().tolist()
        else:
            emb_vec = None

        all_embeddings.append({
            "frame": fname,
            "crop_file": crop_fname,
            "label": label,
            "score": score_val,
            "bbox": box_xyxy,
            "embedding": emb_vec,
        })

    # BBox画像の保存
    if any_detected:
        img_draw.save(os.path.join(CROPS_DIR, f"res_{fname}"))

# Embedding保存（npz/jsonなど用途に応じて）
import json
with open(os.path.join(CROPS_DIR, "all_embeddings.json"), "w", encoding="utf-8") as f:
    json.dump(all_embeddings, f, ensure_ascii=False, indent=2)

print(f"クロップ数: {len(all_embeddings)}, クロップ画像→ {CROPS_DIR}/")
print(f"Embedding記録ファイル: {CROPS_DIR}/all_embeddings.json")
