import os
from dotenv import load_dotenv
from google import genai
import time

# =================================
# Gemini 推論設定
# =================================
load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Geminiクライアント初期化
client = genai.Client(api_key=GEMINI_API_KEY)

# =================================
# Gemini 推論ロジック
# =================================
def invoke_gemini_from_file(prompt: str, file_type: str, filename: str = None) -> str:
    try:
        contents = []

        if filename:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"File not found: {filename}")

            uploaded_file = client.files.upload(file=filename)
            
            if file_type == "video":
                while uploaded_file.state.name == "PROCESSING":
                    print("Waiting for video to be processed...")
                    time.sleep(2)
                    uploaded_file = client.files.get(name=uploaded_file.name)

                if uploaded_file.state.name == "FAILED":
                    raise ValueError("Upload failed.")

            contents.append(uploaded_file)

        contents.append({"text": prompt})

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )
        result = response.text
        
        return result

    except Exception as e:
        return f"Error: {str(e)}"

