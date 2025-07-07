from google.cloud import storage
import os
from dotenv import load_dotenv

load_dotenv()
GCS_ROOT_PATH = os.environ.get("GCS_ROOT_PATH")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")

def get_blob_name(bucket_name=GCS_BUCKET_NAME) -> str:
    """
    指定された GCS バケット内のすべての blob の名前をリストで取得します。
    """    
    try:
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name)
        return [blob.name for blob in blobs if not blob.name.endswith('/')]
    except Exception as e:
        raise RuntimeError(f"GCS から blob 名の取得中にエラーが発生しました: {e}")

def extract_blob_name(file_path: str) -> str:
    """
    ファイルパスからファイル名を抽出します。

    Args:
        file_path (str): ファイルのパス。

    Returns:
        str: 抽出されたファイル名。
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[-1].lower()
    if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        # 動画ファイルの場合
        blob_name = "video/" + filename
    elif ext in [".jpg", ".jpeg", ".png"]:
        # 画像ファイルの場合
        blob_name = "image/" + filename
    else:
        # 想定外のファイル形式の場合はルートに保存（もしくはエラーにする）
        blob_name = filename

    return blob_name

def upload_file_to_gcs(file_path: str) -> str:
    """
    Gradio UI からアップロードされたファイル (動画または画像) を GCS にアップロードします。

    Args:
        file_path (str): アップロードされたファイルの一時的なパス。
                         Gradio の gr.File コンポーネントから受け取ります。

    Returns:
        str: アップロードされたファイルの GCS パス (例: 'gs://bucket_name/video/~~~.mp4')。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"指定されたファイルパスが見つかりません: {file_path}")

    try:
        # Storage クライアントを初期化。認証は環境変数 GOOGLE_APPLICATION_CREDENTIALS
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob_name = extract_blob_name(file_path)
        blob = bucket.blob(blob_name)

        # ファイルをアップロード
        blob.upload_from_filename(file_path)

        print(f"ファイル '{file_path}' は '{GCS_BUCKET_NAME}/{blob_name}' に正常にアップロードされました。")

        # アップロードされたファイルの GCS パスを返す
        return f"gs://{GCS_BUCKET_NAME}/{blob_name}"

    except Exception as e:
        raise RuntimeError(f"GCS へのアップロード中にエラーが発生しました: {e}")

if __name__ == "__main__":
    # テスト実行用
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    filename = "pana.mp4"
    file_path = os.path.join(DATA_DIR, filename)
    gcs_path = upload_file_to_gcs(file_path)
    print(gcs_path)