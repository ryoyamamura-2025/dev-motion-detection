# Motion Detection App
Gemini による動画理解をサポートするための物体検知開発

## 環境構築
WSL2 内の Docker コンテナで開発

```
docker build -t dev-motion-app .
docker run -d -v $(pwd):/usr/src/app --name=dev-motion-app -p 8888:8888 dev-motion-app
```

## SAM2 を動かす方法
1. [リポジトリ](https://github.com/facebookresearch/segment-anything-2.git)のクローン  
```
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
```

2. 必要なパッケージのインストール*  
```
pip install hydra-core iopath ninja natsort
```
注）`pip install -e .` は時間がかかる

3. モデルのロード  
(公式)[https://github.com/facebookresearch/segment-anything-2.git]から選ぶ*か、`checkpoints` ディレクトリにてスクリプトを実行  
モデルは `checkpoints` ディレクトリ内に保管
注）2.1 系は config ファイルを読み込めず動作しなかった

4. Python スクリプトを用意  
`src/prediction.py` を参照

5. 動画像データの準備  
動画の場合、ffmeg というソフトを使ってフレームを `.jpg` に変換（PNGは動作しないのでJPGにすること）
```
ffmpeg -i {input_video_file} -r 1 -q:v 2 -start_number 0 {output_image_dir}/%05d.jpg
--> XXXXX.jpg ファイルが連番で作成される
```

| オプション            | 意味                                         |
|-----------------------|----------------------------------------------|
| `-i {input_video_file}`        | 入力動画                                     |
| `-r n`                | 1秒あたりnフレームに変換（低フレーム数＝間引き） |
| `-q:v n`              | 出力画像の品質（1〜31。1が最高画質）             |
| `-start_number 0`     | 画像ファイルの番号を0から始める                   |
| `%05d.jpg`            | `00000.jpg`, `00001.jpg` のような5桁連番のファイル名 |


6. 推論の実行  
4.の python ファイルを実行。画像の枚数が多いとメモリ不足になるので注意。 tiny モデルでも動画の処理は時間がかかる。画像の処理は一瞬なので追跡まで計算するのに時間がかかると考えられる

7. 推論結果を動画化
```
ffmpeg -framerate 10 -start_number 0 -i {output_image_dir}/%05d.jpg -c:v libx264 -pix_fmt yuv420p {output_video_filepath}
```

| オプション                  | 意味                                                      |
|-----------------------------|-----------------------------------------------------------|
| `-framerate n`              | 1秒間にn枚の画像（フレームレート）                        |
| `-start_number 0`           | 最初の画像が `00000.jpg` など「0」始まりであることを明示 |
| `-i {output_dir}/%05d.jpg`      | 入力画像のパターン（5桁の連番）                           |
| `-c:v libx264`              | 動画コーデック（H.264で圧縮）                             |
| `-pix_fmt yuv420p`          | 互換性の高い画面形式（必要）                              |
| `{output_video_filepath}`                | 出力ファイル名                                            |

### 参考サイト
- [Segment Anything Model 2 (SAM 2)の動画データに対するセグメンテーションのチュートリアル](https://zenn.dev/hacarus_blog/articles/be8dd532ebeda8)
- [【SAM2】動画内から物体を自動検出・追跡する](https://qiita.com/Neckoh/items/1c411a0b71e328fe6b60)


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