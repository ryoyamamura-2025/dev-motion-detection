import shutil
import os
import uuid
import tempfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from models import GenerateRequest, LogRequest
from gemini_client import invoke_gemini_from_file
# from gcs_logger import append_log_entry
import uvicorn

app = FastAPI()

# CORS (Gradioなどから叩く場合)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate_with_file(
    prompt: str = Form(...),
    file_type: str = Form(...),  # "image" or "video"
    file: UploadFile = File(None)
):
    temp_file_path = None
    result = ""

    if file:
        ext = os.path.splitext(file.filename)[-1]
        temp_file_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    print(file_type)
    result = invoke_gemini_from_file(prompt, file_type, temp_file_path if file else None)

    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    return {"result": result}

@app.post("/log/save")
def save_log(request: LogRequest):
    """
    Cloud Storageへのログの保存
    """
    append_log_entry(request)
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
