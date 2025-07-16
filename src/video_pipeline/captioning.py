import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
from config import DATA_DIR, OUTPUT_DIR

CAPTION_CSV = "frame_captions.csv"
PROMPT = (
    "a photo of"
)

def generate_caption(frame_dir, output_dir, blip_model="Salesforce/blip-image-captioning-base", device=torch.device("cpu"), prompt=PROMPT):
    # モデルのロード
    processor = BlipProcessor.from_pretrained(blip_model, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(blip_model)
    model.to(device)
    model.eval()

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(".png")])
    captions = [] # キャプションを格納するリスト

    for fname in tqdm(frame_files):
        path = os.path.join(frame_dir, fname)
        image = Image.open(path).convert("RGB")

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append({"file": fname, "caption": caption})

    # 保存（CSV）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, CAPTION_CSV)
    pd.DataFrame(captions).to_csv(filepath, index=False, encoding="utf-8")
    print(f"✅️キャプション付与＆保存 完了：{len(captions)}件 → {CAPTION_CSV}")

if __name__ == "__main__":
    # テスト実行
    frame_dir = f"{OUTPUT_DIR}/keyframes"
    output_dir = f"{OUTPUT_DIR}/crops"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generate_caption(frame_dir, output_dir, device=device)