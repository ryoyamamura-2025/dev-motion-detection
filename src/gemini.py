import os
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
import datetime
import pytz
import time

# =================================
# 定数
# =================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(PROJECT_ROOT, "log")
LOG_FILE = os.path.join(LOG_DIR, "log.txt")

# =================================
# ログ記録用
# =================================
def write_log(entry: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    japan_tz = pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(japan_tz).strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{now}]\n{entry}\n\n")

# =================================
# Gemini 推論設定
# =================================
load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Geminiクライアント初期化
client = genai.Client(api_key=GEMINI_API_KEY)

# 設定クラス
class Settings(BaseModel):
    model: str = "gemini-2.5-flash"
    max_tokens: int = 50000
    temperature: float = 0.1
    prompt: str

# 添付ファイルクラス
class AttachedFile(BaseModel):
    type: str # image or movie
    filename: str

# =================================
# Gemini 推論関数
# =================================
def invoke_gemini(settings: Settings, attached_file: AttachedFile = None):
    try:
        contents = []
        # 添付ファイルがあればアップロードして追加
        if attached_file:
            if not os.path.exists(attached_file.filename):
                raise FileNotFoundError(f"File not found: {attached_file.filename}")
            
            uploaded_file = client.files.upload(file=attached_file.filename)
            
            if attached_file.type == "movie":
                # APIがファイルを受信したことを確認
                while uploaded_file.state.name == "PROCESSING":
                    print('Waiting for video to be processed.')
                    time.sleep(2)
                    uploaded_file = client.files.get(name=uploaded_file.name)

                if uploaded_file.state.name == "FAILED":
                    raise ValueError(uploaded_file.state.name)
                print("Video processing complete:", uploaded_file.uri)

            contents.append(uploaded_file)
        
        # プロンプトを追加
        contents.append(settings.prompt)

        # API 呼び出し
        response = client.models.generate_content(
            model=settings.model,
            contents=contents
        )
        result = response.text
        write_log(f"Prompt: {settings.prompt}\nAttached: {attached_file.filename if attached_file else 'None'}\nResult:\n{result}")
        return result

    except Exception as e:
        write_log(f"Error: {str(e)}")
        return f"Error: {str(e)}"

# =================================
# 実行例
# =================================
if __name__ == "__main__":
    # # テキストのみ
    # settings = Settings(prompt="こんにちは。")
    # result = invoke_gemini(settings)
    # print(result)

    # 画像付き
    filename = os.path.join(DATA_DIR, "formation_207.png")
    # print(filename)
    attached_file = AttachedFile(type="image", filename=filename)
    settings = Settings(prompt="この画像をキャプションしてください。")
    result = invoke_gemini(settings, attached_file)
    print(result)

    # 動画付き
    # filename = os.path.join(DATA_DIR, "pana.mp4")
    # attached_file = AttachedFile(type="movie", filename=filename)
    # settings = Settings(prompt="この動画を要約して、内容に基づいたクイズを作ってください。")
    # result = invoke_gemini(settings, attached_file)
    # print(result)
