import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions, Part

# =================================
# Gemini 推論設定
# =================================
load_dotenv() # GOOGLE_APPLICATION_CREDENTIALS を読み込む
# GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
LOCATION = os.environ["LOCATION"]

# Geminiクライアント初期化
# client = genai.Client(api_key=GEMINI_API_KEY)
client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT_ID,
    location=LOCATION,
    http_options=HttpOptions(api_version="v1")
  )

# =================================
# Gemini 推論ロジック
# =================================
def generate_text(prompt: str) -> str:
    """
    単純な Gemini へのリクエスト

    Args:
        prompt(str): プロンプト（ファイルの後ろにいれる）
    
    Returns:
        str: 推論結果のテキスト
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    result = response.text
    
    return result


def generate_text_with_file(prompt: str, filename: str) -> str:
    """
    Google Storage に保存されたファイルをインプットに含めて推論する

    Args:
        prompt(str): プロンプト（ファイルの後ろにいれる）
        gs_path (str): Google Storageのパス（gs://bucket_name/file_name）
    
    Returns:
        str: 推論結果のテキスト
    """
    GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
    ext = os.path.splitext(filename)[-1].lower()
    gs_path = f"gs://{GCS_BUCKET_NAME}/{filename}"

    if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        # 動画ファイルの場合
        mime_type = "video/" + ext[1:]
    elif ext in [".jpg", ".jpeg", ".png"]:
        # 画像ファイルの場合
        if ext == ".jpg":
            mime_type = "image/jpeg"
        else:
            mime_type = "image/" + ext[1:]
    else:
        raise ValueError(f"未対応のファイル形式です: {ext}")

    try:
        contents = [
            Part.from_uri(file_uri=gs_path, mime_type=mime_type),
            prompt
        ]

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents
        )

        result = response.text
        
        return result

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # テスト実行用
    filename = "pana.mp4"
    print(generate_text_with_file(prompt="この動画を要約して", filename=filename))
    # print(generate_text(prompt="へろー"))