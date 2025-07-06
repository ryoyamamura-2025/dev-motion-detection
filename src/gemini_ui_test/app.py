import gradio as gr
import requests

API_URL = "http://localhost:8000/generate"  # FastAPI 側のエンドポイント

def send_to_api(prompt, video):
    if video is None or prompt.strip() == "":
        return "動画とプロンプトを入力してください。"

    with open(video, "rb") as f:
        files = {"file": (video, f, "video/mp4")}
        data = {
            "prompt": prompt,
            "file_type": "video"
        }
        response = requests.post(API_URL, data=data, files=files)
    
    if response.status_code == 200:
        return response.json().get("result", "結果の取得に失敗しました。")
    else:
        return f"エラー: {response.status_code} - {response.text}"

gr.Interface(
    fn=send_to_api,
    inputs=[
        gr.Textbox(label="プロンプト", placeholder="例: この動画を要約してください"),
        gr.Video(label="動画ファイル")
    ],
    outputs=gr.Textbox(label="Gemini 応答"),
    title="動画キャプション生成（Gemini）",
    allow_flagging="never"
).launch(server_port=8888, server_name="0.0.0.0")
