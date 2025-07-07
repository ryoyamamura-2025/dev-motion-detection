import os
from gemini_client import generate_text, generate_text_with_file
from file_operator import upload_file_to_gcs, get_blob_name
from model import GenerateRequest

def handle_file_upload(file_path: str) -> str:
    """
    ファイルを Google Cloud Storage にアップロードする

    Args:
        file_path (str): アップロードするファイルの名前
    
    Returns:
        str: アップロードされたファイルの Google Storage パス
    """
    return upload_file_to_gcs(file_path)

def handle_get_blob():
    """
    Google Cloud Storage 上のファイルのパスを取得する

    Returns:
        List(str): Google Storage 上のファイルのパス
    """
    return get_blob_name()

def handle_gemini_request(request: GenerateRequest) -> dict:
    """
    Gemini へのリクエストを処理するエンドポイント

    Args:
        request (GenerateRequest): Gemini へのリクエストデータ
    
    Returns:
        dict: 推論結果のテキストデータの辞書
    """    
    prompt = request.prompt
    filename = request.filename

    if not filename:
        # ファイルが指定されていない場合
        result = generate_text(prompt)
    else:
        # ファイルが指定されている場合
        result = generate_text_with_file(prompt, filename)
    return {"result": result}

# def save_log():
#     """
#     Cloud Storageへのログの保存
#     """
#     # append_log_entry(request)
    # return {"status": "success"}
