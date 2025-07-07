from pydantic import BaseModel

class GenerateRequest(BaseModel):
    """
    Gemini へのリクエスト
    """
    prompt: str
    filename: str = None  # ファイル名がある場合のみ使用される
    # model: str = "gemini-2.5-flash"  # 使用するモデル名
    # temperature: float = 0.2  # 温度パラメータ
    # max_output_tokens: int = 65535  # 最大出力トークン数
    