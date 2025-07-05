FROM python:3.10.4
WORKDIR /usr/src/app
# OpenCV と yt_dlp 動かすために必要
RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt install ffmpeg -y
COPY requirements.txt ./
# pip install --no-cache-dir -r requirements.txt の行はそのまま
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["sleep", "infinity"]

# Dockerイメージのビルドと実行コマンド例 (変更なし)
# docker build -t [イメージ名]:[タグ] [Dockerfileのパス]
# docker run -d -v [ホストディレクトリの絶対パス]:[コンテナの絶対パス] [イメージ名] [コマンド]
# docker run -d -v $(pwd):/[コンテナの絶対パス] [イメージ名] [コマンド]
# docker run -d -v $(pwd):/[コンテナの絶対パス] -p [ホストのポート番号]:[コンテナのポート番号] [イメージ名] [コマンド]
# docker run -d -v $(pwd):/usr/src/app --name=dev-motion-app -p 8888:8888 dev-motion-app