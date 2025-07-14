# Motion Detection App
Gemini による動画理解をサポートするための物体検知開発

## 環境構築
WSL2 内の Docker コンテナで開発

```
docker build -t dev-motion-app .
docker run -d -v $(pwd):/usr/src/app --name=dev-motion-app -p 8888:8888 dev-motion-app
```

## COIN データセット
[COmprehensive INstructional video analysis (COIN)](https://coin-dataset.github.io/):   
COINデータセットは、180の異なるタスクに関連する11,827の動画から構成され、これらはすべてYouTubeから収集された。動画の長さは平均2.36分である。各動画は3.91ステップのセグメントでラベル付けされ、各セグメントは平均14.91秒である。このデータセットには、合計で 476 時間の動画が含まれ、46,354 のアノテーションセグメントが含まれる (上記リンク内の説明文を DeepL で翻訳)    
---
本のデータが Youtube 動画なので商用利用時は注意が必要。基本は研究用途で利用。

### データのロード方法
COIN.json から適当な URL を見繕って `src/load_video.py` でダウンロード

## トラブルシューティング
- GCS の画像/動画データをプロンプトに入れて Gemini に送る際のエラー: 
    ```
    Error: 400 FAILED_PRECONDITION. {'error': {'code': 400, 'message': 'Service agents are being provisioned (https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents). Service agents are needed to read the Cloud Storage file provided. So please try again in a few minutes.', 'status': 'FAILED_PRECONDITION'}`
    ```   
    [Firebaseトラブルシューティング](https://firebase.google.com/docs/ai-logic/faq-and-troubleshooting?hl=ja&api=dev#error-cloud-storage-service-agents)のcurlコマンドを実行することで解消