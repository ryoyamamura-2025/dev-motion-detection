import os
import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageDraw
import tempfile
from dotenv import load_dotenv

from pydantic import BaseModel
from typing import List

load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# ==== モデル定義 ====
class BBox(BaseModel):
    label: str
    caption: str
    box_2d: List[float]
    score: float

class DetectionResult(BaseModel):
    detections: List[BBox]

# ==== ダミー検出 ====
def dummy_gemini_detect(image: Image.Image) -> DetectionResult:
    dummy_json = {
        "detections": [
            {"label": "person", "caption": "a black-haired woman in a blue shirt", "box_2d": [250, 300, 900, 550], "score": 0.98},
            {"label": "person", "caption": "a man in a white shirt", "box_2d": [100, 700, 900, 850], "score": 0.91}
        ]
    }
    return DetectionResult(**dummy_json)

# ==== BBOX 描画 ====
def draw_bboxes(image: Image.Image, detection: DetectionResult) -> Image.Image:
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for det in detection.detections:
        y_min, x_min, y_max, x_max = det.box_2d
        box = [x_min / 1000 * w, y_min / 1000 * h, x_max / 1000 * w, y_max / 1000 * h]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1] - 10), det.caption, fill="red")
    return image

# ==== サムネイル抽出 ====
def extract_thumbnail(video_file):
    if video_file is None:
        return None
    with open(video_file, "rb") as f:
        video_bytes = f.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

# === 要約生成（ダミー） ===
def generate_dummy_summary(detection_val: DetectionResult):
    # 各 caption ごとに文を作る
    lines = f"{detection_val.detections[0].caption}さんが座ってリハビリをしています。" 
    return lines

# ==== Gradio アプリ ====
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 動画アップロード → 物体検出 → キャプション編集")

    detection_state = gr.State()
    df= gr.DataFrame(
        headers=["label", "caption"], 
        datatype=["str", "str"], 
        interactive=True,
        label="検出結果",
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="動画をアップロード")
            summary_button  = gr.Button("動画要約")
            summary_output  = gr.Textbox(label="要約", lines=4, interactive=True)


        with gr.Column():
            thumbnail_output = gr.Image(label="抽出されたサムネイル", type="pil")

        with gr.Column():
            detect_button = gr.Button("物体を検出")
            ok_button = gr.Button("OKで確定して次へ")
            df 

    # サムネイル抽出
    video_input.change(fn=extract_thumbnail, inputs=[video_input], outputs=[thumbnail_output])

    # 物体検出 → DataFrame に流し込み
    def detect_and_show_df(image: Image.Image):
        detection = dummy_gemini_detect(image)
        # DataFrame 用データを作成
        data = [[det.label, det.caption] for det in detection.detections]
        return draw_bboxes(image.copy(), detection), detection, data

    detect_button.click(
        fn=detect_and_show_df,
        inputs=[thumbnail_output],
        outputs=[thumbnail_output, detection_state, df]
    )

    # OKボタンで DataFrame の caption を反映
    def update_from_df(image: Image.Image, table, detection_val):
        # DataFrame → list[list] へ変換
        if hasattr(table, "values"):          # pandas.DataFrame の場合
            rows = table.values.tolist()
        else:                                 # すでに list[list] の場合
            rows = table

        updated = []
        for row, det in zip(rows, detection_val.detections):
            label, new_caption = row  # rows は [label, caption]
            updated.append(det.model_copy(update={"caption": new_caption}))

        dr = DetectionResult(detections=updated)
        new_data = [[d.label, d.caption] for d in dr.detections]

        return draw_bboxes(image.copy(), dr), dr, new_data

    ok_button.click(
        fn=update_from_df,
        inputs=[thumbnail_output, df, detection_state],
        outputs=[thumbnail_output, detection_state, df]
    )

    # 要約ボタン → ダミー要約生成
    summary_button.click(
        fn=generate_dummy_summary,
        inputs=[detection_state],
        outputs=[summary_output]
    )

if __name__ == "__main__":
    demo.launch(server_port=8080, server_name="0.0.0.0")
