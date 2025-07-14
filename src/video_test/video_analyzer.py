import os
from dotenv import load_dotenv
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

load_dotenv()
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
DATA_DIR = os.environ["DATA_DIR"]
MODEL_DIR = os.environ["MODEL_DIR"]

# 動画ファイル指定
video_filename = "ice_vending.mp4"
name, ext = os.path.splitext(video_filename)
ext = ext.lower()
video_path = os.path.join(DATA_DIR, video_filename) 
output_raw_dir = os.path.join(DATA_DIR, "output", name, "frames")
output_pred_dir = os.path.join(DATA_DIR, "output", name, "preds")

os.makedirs(output_raw_dir, exist_ok=True)
os.makedirs(output_pred_dir, exist_ok=True)

# YOLOv8モデルロード
model_path = os.path.join(MODEL_DIR, "yolov8n.pt")
model = YOLO(model_path)

# DeepSORT初期化
tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0)

# フレームごとに保存
cap = cv2.VideoCapture(video_path)
frame_id = 0

start = time.perf_counter()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # フレーム保存（そのまま）
    raw_path = os.path.join(output_raw_dir, f"frame_{frame_id:04d}.jpg")
    cv2.imwrite(raw_path, frame)

    # YOLO推論
    result = model(frame, verbose=False)[0]  # 推論1回分
    annotated_frame = frame.copy()

    # DeepSORTトラッキング
    detections_for_tracker = [] 

    for box in result.boxes:
        cls_id = int(box.cls.item())
        name = model.names[cls_id]
        if name not in ["refrigerator", "remote"]:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf.item())

        # DeepSORT用: [x1, y1, x2 - x1, y2 - y1]
        detections_for_tracker.append(([x1, y1, x2 - x1, y2 - y1], conf, name))

    # DeepSORT追跡
    annotated_frame = frame.copy()
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # BBOX付き画像保存
    annotated_path = os.path.join(output_pred_dir, f"yolo_deep_{frame_id:04d}.jpg")
    cv2.imwrite(annotated_path, annotated_frame)

    frame_id += 1

cap.release()
end = time.perf_counter()

print(f"Processed {frame_id} frames from {video_filename}.")
print(f"Total time: {end - start:.2f} seconds.")