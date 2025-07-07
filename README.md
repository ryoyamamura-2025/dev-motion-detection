# Motion Detection App
Gemini による動画理解をサポートするための物体検知開発

## 環境構築
WSL2 内の Docker コンテナで開発

```
docker build -t dev-motion-app .
docker run -d -v $(pwd):/usr/src/app --name=dev-motion-app -p 8888:8888 dev-motion-app
```

## トラブルシューティング
- GCS の画像/動画データをプロンプトに入れて Gemini に送る際のエラー: 
    ```
    Error: 400 FAILED_PRECONDITION. {'error': {'code': 400, 'message': 'Service agents are being provisioned (https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents). Service agents are needed to read the Cloud Storage file provided. So please try again in a few minutes.', 'status': 'FAILED_PRECONDITION'}`
    ```   
    [Firebaseトラブルシューティング](https://firebase.google.com/docs/ai-logic/faq-and-troubleshooting?hl=ja&api=dev#error-cloud-storage-service-agents)のcurlコマンドを実行することで解消