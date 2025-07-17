import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import os
from shutil import copy2
from tqdm import tqdm
from config import DATA_DIR, OUTPUT_DIR


CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")
EMB_PATH = os.path.join(CROPS_DIR, "all_embeddings.json")
CLUSTERS_DIR = os.path.join(OUTPUT_DIR, "clusters")
os.makedirs(CLUSTERS_DIR, exist_ok=True)

# # JSON読込
# with open(EMB_PATH, encoding="utf-8") as f:
#     data = json.load(f)

# # embedding, crop, label, 他属性リスト
# embeddings = []
# crop_files = []
# labels = []
# frame_files = []
# for item in data:
#     if item["embedding"] is not None:
#         emb = np.array(item["embedding"]).flatten()  # flattenで常に1次元
#         embeddings.append(emb)
#         crop_files.append(item["crop_file"])
#         labels.append(item["label"])
#         frame_files.append(item["frame"])


def clustering(embeddings, crop_files):
    embeddings = np.array(embeddings)  # shape=(N, D)
    print(embeddings.shape)
    embeddings = normalize(embeddings, norm="l2")

    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(xy[:, 0], xy[:, 1], c='blue', alpha=0.6)

    # 各点にインデックス番号を表示
    for i, (x, y) in enumerate(xy):
        plt.text(x, y, str(i), fontsize=9, color="red")

    plt.title("PCA embedding")
    plt.savefig("/usr/src/app/data/output/crops/pca.png")
    # plt.show()

    # クラスタリング
    dbscan = DBSCAN(eps=0.05, min_samples=2, metric="cosine")  # eps=0.20〜0.30 で様子を見て調整
    labels_db = dbscan.fit_predict(embeddings)

    # クラスタごとに代表クロップ画像を選び、フォルダ出力
    cluster_summary = {}
    for cid in set(labels_db):
        if cid == -1: continue  # ノイズ扱い
        idxs = np.where(labels_db == cid)[0]
        cluster_dir = os.path.join(CLUSTERS_DIR, f"cluster_{cid:03d}")
        os.makedirs(cluster_dir, exist_ok=True)

        # 中心に最も近い画像を代表に（embeddingの平均ベクトルとの距離で判定）
        emb_center = embeddings[idxs].mean(axis=0)
        dists = cosine_distances([emb_center], embeddings[idxs])[0]
        rep_idx = idxs[dists.argmin()]
        rep_crop = crop_files[rep_idx]
        # rep_label = labels[rep_idx]

        # 代表クロップ画像をコピー
        copy2(os.path.join(CROPS_DIR, rep_crop), os.path.join(cluster_dir, "rep_" + rep_crop))
        # クラスタ内すべてのクロップ画像をコピー（必要なら）
        for i in idxs:
            copy2(os.path.join(CROPS_DIR, crop_files[i]), os.path.join(cluster_dir, crop_files[i]))

        # クラスタごとのメタ情報を整理
        cluster_summary[f"cluster_{cid:03d}"] = {
            "representative": "rep_" + rep_crop,
            # "label": rep_label,
            "members": [crop_files[i] for i in idxs],
            "n_members": len(idxs),
            # "frames": [frame_files[i] for i in idxs],
        }

    print(cluster_summary)

# # 結果をJSONで保存
# with open(os.path.join(CLUSTERS_DIR, "cluster_summary.json"), "w", encoding="utf-8") as f:
#     json.dump(cluster_summary, f, ensure_ascii=False, indent=2)

# print(f"クラスタ数: {len(cluster_summary)}、代表クロップは各 cluster_xxx/rep_*.png")
