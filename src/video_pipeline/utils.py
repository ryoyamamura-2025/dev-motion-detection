import os
import json

def get_imagelist(dir, ext="png", prefix=""):
    """
    指定ディレクトリ内の画像ファイルをソートして取得する。
    対応拡張子は .jpg, .jpeg, .png。デフォルトは png 形式。

    Args:
        dir (str): 画像ファイルを検索するディレクトリのパス
        ext (str): 取得する画像の拡張子（デフォルトは "png"）
    Returns:
        list: 指定ディレクトリ内の画像ファイル名のリスト
    """
    ext = ext.lower()
    
    if ext not in ["jpg", "jpeg", "png"]:
        raise ValueError(f"Unsupported image extension: {ext}. Use jpg, jpeg, or png.")
    
    # ディレクトリから画像ファイルを取得
    if prefix:
        # ソート
        files = sorted([f for f in os.listdir(dir) if f.lower().endswith(ext) and f.startswith(prefix)])
    else:
        files = sorted([f for f in os.listdir(dir) if f.lower().endswith(ext)])
    return files


if __name__ == "__main__":
    # デモ用のディレクトリパス
    dir = "/usr/src/app/data/output/keyframes"
    print(get_imagelist(dir))
