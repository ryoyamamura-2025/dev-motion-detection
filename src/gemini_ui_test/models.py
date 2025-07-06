from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_tokens: int = 50000
    file_type: str  # "image" or "video"
    file_url: str

class LogRequest(BaseModel):
    prompt: str
    file_url: str
    result: str
    memo: str
