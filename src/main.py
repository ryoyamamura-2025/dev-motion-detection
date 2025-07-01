import gradio as gr
import cv2
import numpy as np
from PIL import Image
import tempfile

# 📸 動画から最初のサムネイル画像を抽出
def extract_thumbnail(video_file):
    if video_file is None:
        return None  # 何もアップロードされていないときはスキップ
    
    # video_file はファイルパス（str）なので、ファイルの内容を読み込む
    with open(video_file, "rb") as f:
        video_bytes = f.read()

    # 一時ファイルとして保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # middle_frame = frame_count // 2
    first_frame = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    success, frame = cap.read()
    cap.release()
    if not success:
        return None

    # BGR → RGB → PIL Image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    return image

# 🎛 Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 動画アップロード → サムネイル抽出")

    with gr.Row():
        video_input = gr.Video(label="動画をアップロード")
        thumbnail_output = gr.Image(label="抽出されたサムネイル", type="pil")

    video_input.change(
        fn=extract_thumbnail,
        inputs=[video_input],
        outputs=[thumbnail_output]
    )


if __name__ == "__main__":
    demo.launch(server_port=8080, server_name="0.0.0.0")