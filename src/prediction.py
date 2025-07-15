import time
import os
import cv2
import torch
import numpy as np
import pandas as pd
from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor

# マスク描画用関数
def show_mask(mask, ax, obj_id=None):
    cmap = plt.get_cmap("tab10")
    cmap_idx = 0 if obj_id is None else obj_id
    color = np.array([*cmap(cmap_idx)[:3], 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# プロンプト描画用関数
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

if __name__ == "__main__":
    # モデルのロード （tiny を使用）
    sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # 初期化 + 動画の各フレームの image embedding 求める
    input_img_dir = "frames"
    inference_state = predictor.init_state(video_path=input_img_dir)

    # 動画の最初のフレームを取得
    frame_names = natsorted(glob(f"{input_img_dir}/*.jpg"))
    image = cv2.imread(frame_names[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)

    # 描画
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    ## 座標をプロンプトとして指定
    ann_frame_idx = 0  # 解析対象のフレームインデックス
    ann_obj_id = 0  # 解析対象の物体に付与する一意のID（任意の整数を設定）
    input_point = np.array([[100, 100]], dtype=np.float32)
    input_label = np.array([1], np.int32) # 1がPositive、0がNegativeを意味する。pointsの要素と対応している。

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=input_point,
        labels=input_label,
    )

    ## セグメンテーション結果を描画
    plt.figure(figsize=(12, 8))
    plt.title(f"frame {ann_frame_idx}")
    image = cv2.imread(frame_names[ann_frame_idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    plt.show()

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # 結果の静止画を保存
    plt.close("all")
    for out_frame_idx in range(len(frame_names)):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.axis('off')
        plt.tight_layout(pad=0)

        # 元画像の描画
        image = cv2.imread(frame_names[out_frame_idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)

        # マスクの描画
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

        # 結果を保存
        basename = os.path.basename(frame_names[out_frame_idx])
        output_img_dir = "frames/output_frames"
        output_frame = os.path.join(output_img_dir, basename)
        plt.savefig(output_frame)
        plt.close()