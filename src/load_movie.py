import yt_dlp
import os

# =================================
# 定数
# =================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def download_youtube_video(url, output):
    output_path = os.path.join(DATA_DIR, output)
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
        'quiet': False,
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# 例：URLを指定して実行
if __name__ == "__main__":
    video_url = input("YoutubeのURLを入力してください: ")
    output = input("保存するファイル名を入力してください: ")
    download_youtube_video(video_url, output)