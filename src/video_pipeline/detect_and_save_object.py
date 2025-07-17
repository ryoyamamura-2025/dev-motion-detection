import os
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from tqdm import tqdm
from config import DATA_DIR, OUTPUT_DIR
from utils import get_imagelist
from make_cluster import clustering

# パスなど
FRAME_DIR = os.path.join(OUTPUT_DIR, "keyframes")
DEVICE = torch.device("cpu")
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")


frames = get_imagelist(FRAME_DIR) # フレーム画像の読み込み
texts = ["a salient object"] # 検出対象のテキスト（プロンプト）
# texts = ["a photo of machine", "a salient object"] # 検出対象のテキスト（プロンプト）
os.makedirs(CROPS_DIR, exist_ok=True)  # クロップ保存ディレクトリ
all_embeddings = [] # 埋め込み・クロップ管理リスト

# ヘルパー関数: BBoxのサイズ比率を計算
def box_size_ratio(box, image_size):
    """
    BBoxのサイズ比率を計算
    box: [xmin, ymin, xmax, ymax]
    image_size: (width, height)
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    ratio = (width * height) / (image_size[0] * image_size[1])
    return ratio, width, height  # ratio, width, heightを返す


# ヘルパー関数: ラプラシアンフィルタを適用して画像の解像度をチェック
def is_low_resolution(image, threshold=200):
    """
    画像の構造・解像度が低いかどうかをチェック
    threshold: ラプラシアンの値がこの値以下なら構造なし/低解像度と判断

    Args:
        image (PIL.Image): チェックする画像
        threshold (float): ラプラシアンの分散の閾値
    Returns:
        bool: 低解像度ならTrue、そうでなければFalse
    """
    # グレースケールに変換
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(np.array(img_gray), cv2.CV_64F)
    variance = laplacian.var()
    check = variance < threshold
    return variance, check  # Trueなら低解像度と判断、Falseなら高解像度と判断


# ヘルパー関数: 中心から4角の座標を得る
def center_to_corners_format(boxes):
    """
    Args:
        boxes: (N, 4) テンソル。各行は [cx, cy, w, h] 形式
    Returns:
        (N, 4) テンソル。各行は [x1, y1, x2, y2] 形式
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)

# ヘルパー関数: OWL-ViT のソースコードからそのまま拝借
def _scale_boxes(boxes, target_sizes):
    """
    Scale batch of bounding boxes to the target sizes.

    Args:
        boxes (`torch.Tensor` of shape `(batch_size, num_boxes, 4)`):
            Bounding boxes to scale. Each box is expected to be in (x1, y1, x2, y2) format.
        target_sizes (`list[tuple[int, int]]` or `torch.Tensor` of shape `(batch_size, 2)`):
            Target sizes to scale the boxes to. Each target size is expected to be in (height, width) format.

    Returns:
        `torch.Tensor` of shape `(batch_size, num_boxes, 4)`: Scaled bounding boxes.
    """

    if isinstance(target_sizes, (list, tuple)):
        image_height = torch.tensor([i[0] for i in target_sizes])
        image_width = torch.tensor([i[1] for i in target_sizes])
    elif isinstance(target_sizes, torch.Tensor):
        image_height, image_width = target_sizes.unbind(1)
    else:
        raise ValueError("`target_sizes` must be a list, tuple or torch.Tensor")

    scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
    scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
    boxes = boxes * scale_factor
    return boxes


# ヘルパー関数: class_vec と emb_vec を取得
def post_process_with_emb_vec(outputs, target_sizes, threshold=0.1):
    """
    OWL-ViTからEmbedding抽出
    post_process の返り値に embeds を追加
        image_embeds: [batch_size, patch_size, patch_size, output_dim] = [batch_size, 24, 24, 768]
        class_embeds: [batch_size, num_patches, hidden_size] = [batch_size, 576, 512]
 
    Ref1: https://huggingface.co/docs/transformers/model_doc/owlvit#transformers.OwlViTForObjectDetection
    Ref2: https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/models/owlvit/image_processing_owlvit.py の line 494
    """
    batch_logits, batch_boxes = outputs.logits, outputs.pred_boxes
    batch_class_embeds, batch_image_embes = outputs.class_embeds, outputs.image_embeds
    s0, _, _, s3 = batch_image_embes.size()
    batch_image_embes = batch_image_embes.view(s0, -1, s3) # [batch_size, 24, 24, 768] --> [batch_size, 24*24, 768]
    batch_size = len(batch_logits)
    if target_sizes is not None and len(target_sizes) != batch_size:
        raise ValueError("Make sure that you pass in as many target sizes as images")
    # batch_logits of shape (batch_size, num_queries, num_classes)
    batch_class_logits = torch.max(batch_logits, dim=-1)
    batch_scores = torch.sigmoid(batch_class_logits.values)
    batch_labels = batch_class_logits.indices
    # Convert to [x0, y0, x1, y1] format
    batch_boxes = center_to_corners_format(batch_boxes)
    # Convert from relative [0, 1] to absolute [0, height] coordinates
    batch_boxes = _scale_boxes(batch_boxes, target_sizes)

    results = []
    for scores, labels, boxes, class_embeds, image_embes in zip(batch_scores, batch_labels, batch_boxes, batch_class_embeds, batch_image_embes):
        keep = scores > threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]
        class_embeds = class_embeds[keep]
        image_embes = image_embes[keep]
        results.append({
            "scores": scores, 
            "labels": labels, 
            "boxes": boxes,
            "class_embeds": class_embeds,
            "image_embeds": image_embes
        })

    return results

# モデル＆プロセッサ（CPUロード）
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
model = model.to(DEVICE)
model.eval()

for idx, fname in tqdm(enumerate(frames)):
    query = texts
    image_path = os.path.join(FRAME_DIR, fname)
    target_image = Image.open(image_path).convert("RGB")

    # 推論
    inputs = processor(text=query, images=target_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        target_sizes = torch.tensor([target_image.size[::-1]])
        # results_ = processor.post_process_grounded_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]
        results_ = post_process_with_emb_vec(outputs, target_sizes=target_sizes, threshold=0.1)[0]

    scores = results_["scores"]
    boxes = results_["boxes"]
    
    if len(scores) == 0:
        print(f"No objects detected in {fname}.")
        continue

    nms = torchvision.ops.nms(boxes, scores, iou_threshold=0.3)
    # print(len(nms))
    # print(nms)

    # nmsの処理
    for k, v in results_.items():
        if isinstance(v, torch.Tensor):
            results_[k] = v[nms].cpu().numpy()
        else:
            results_[k] = v
    
    print(f"Frame: {fname}")
    print(f"Detected boxes: {len(results_['scores'].tolist())}")
    # print("class", results_["class_embeds"].shape)
    # print("image", results_["image_embeds"].shape)
    print("-----------------")

    scores = results_["scores"].tolist()
    boxes = results_["boxes"].tolist()
    labels = results_["labels"].tolist()
    class_embeds = results_["class_embeds"].tolist()
    image_embeds = results_["image_embeds"].tolist()

    # BBox描画用
    img_draw = target_image.copy()
    draw = ImageDraw.Draw(img_draw)
    any_detected = False

    for obj_id, (box, score, label_idx, class_embed, image_embed) in enumerate(zip(boxes, scores, labels, class_embeds, image_embeds)):
        # オリジナルの画像に比べてサイズが小さい場合 (面積の割合が10%以下) はスキップ
        ratio, width, height = box_size_ratio(box, target_image.size)
        if ratio < 0.1:
            continue
        
        label = texts[label_idx]
        label = label.replace(" ", "_")  # スペースをアンダースコアに変換

        # BBOX の幅と高さを10%広げる（端はみ出しは防止）物体がちゃんと映るように
        box_xyxy_ext = [
            max(0, box[0] - int(0.05*width)),  # xmin
            max(0, box[1] - int(0.05*height)),  # ymin
            min(target_image.width, box[2] + int(0.05*width)),  # xmax
            min(target_image.height, box[3] + int(0.05*height))   # ymax
        ]

        # クロップ
        crop = target_image.crop(box)

        # 低解像度かチェック
        variance, check = is_low_resolution(crop)
        if check:
            print(f"Low resolution crop skipped: {crop.size}, variance: {variance:.2f}")
            continue

        crop = target_image.crop(box_xyxy_ext)
        crop_fname = f"{os.path.splitext(fname)[0]}_{label}_{obj_id:02d}.png"
        crop.save(os.path.join(CROPS_DIR, crop_fname))

        # BBox描画
        draw.rectangle(box_xyxy_ext, outline="red", width=4)
        text = f"{score:.2f}"
        draw.text((box_xyxy_ext[0]+10, box_xyxy_ext[1]+10), text, fill="green")
        any_detected = True

        all_embeddings.append(class_embed)
        # # Embedding抽出
        # # [バッチ1, Num_boxes, D]
        # if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        #     # 各bboxとラベルidxに該当するembeddingを保存
        #     emb_vec = outputs.image_embeds[0, label_idx].cpu().numpy().tolist()
        # else:
        #     emb_vec = None

        # all_embeddings.append({
        #     "frame": fname,
        #     "crop_file": crop_fname,
        #     "label": label,
        #     "score": score_val,
        #     "bbox": box_xyxy,
        #     "embedding": emb_vec,
        # })

    # BBox画像の保存
    if any_detected:
        img_draw.save(os.path.join(CROPS_DIR, f"res_{fname}"))

print(len(all_embeddings))

crop_files = get_imagelist(CROPS_DIR, prefix="frame")
clustering(all_embeddings, crop_files)

# # Embedding保存（npz/jsonなど用途に応じて）
# import json
# with open(os.path.join(CROPS_DIR, "all_embeddings.json"), "w", encoding="utf-8") as f:
#     json.dump(all_embeddings, f, ensure_ascii=False, indent=2)

# print(f"クロップ数: {len(all_embeddings)}, クロップ画像→ {CROPS_DIR}/")
# print(f"Embedding記録ファイル: {CROPS_DIR}/all_embeddings.json")
