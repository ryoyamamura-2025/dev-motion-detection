import gradio as gr
import shutil
from controller import handle_file_upload, handle_get_blob, handle_gemini_request
from model import GenerateRequest

def process_and_upload(file_obj):
    """
    Gradio のインターフェースからファイルを受け取り、GCS にアップロードする関数。

    Args:
        file_obj (gr.File): Gradio の File コンポーネントからの入力

    Returns:
        str: アップロード結果のメッセージ
    """
    if file_obj is None:
        return "ファイルがアップロードされていません。", gr.Dropdown.update()

    # Gradio はアップロードされたファイルを一時パスに保存
    # file_obj.name が一時的なパス
    uploaded_file_path = file_obj.name

    # controller.py の関数を呼び出して GCS にアップロード
    try:
        gcs_path = handle_file_upload(uploaded_file_path)
        # アップロード後、ファイル選択プルダウンを更新
        updated_blob_list = handle_get_blob()
        return f"{gcs_path}にアップロードされました。", gr.Dropdown.update(choices=updated_blob_list)
    except Exception as e:
        # エラー時にもファイル一覧を更新
        updated_blob_list = handle_get_blob()
        return f"アップロード中にエラーが発生しました: {str(e)}", gr.Dropdown.update(choices=updated_blob_list)

def get_blob_list():
    """
    GCS 上のファイルのパスを取得して表示する関数。

    Returns:
        List[str]: GCS 上のファイルのパスのリスト
    """
    try:
        blob_names = handle_get_blob()
        return blob_names
    except Exception as e:
        return [f"エラーが発生しました: {str(e)}"]


def gemini_request(selected_filename, prompt):
    request = GenerateRequest(prompt=prompt, filename=selected_filename if selected_filename else None)
    result = handle_gemini_request(request)
    if isinstance(result, dict) and "result" in result:
        return result["result"]
    else:
        return "Gemini からの応答が不正です。"

# ========================================
# Gradio UI の定義
# ========================================
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Gemini UI TEST
        """
    )
    with gr.Tab("ファイルアップロード"):
        file_input = gr.File(label="動画または画像ファイルをアップロード", file_types=["video", "image"])
        upload_button = gr.Button("GCSにアップロード")
        upload_output_message = gr.Textbox(label="アップロード結果")

    with gr.Tab("Gemini リクエスト"):
        gr.Markdown("### GCS ファイルを選択して Gemini にリクエスト")
        
        # GCS ファイル一覧のプルダウン
        gcs_file_dropdown = gr.Dropdown(
            label="GCS ファイルを選択",
            choices=[], # 初期値は空、後で更新
            interactive=True,
            allow_custom_value=True
        )

        refresh_files_button = gr.Button("GCSファイル一覧を更新")
        # プロンプト入力
        prompt_input = gr.Textbox(label="Gemini へのプロンプト", lines=5, placeholder="ここにプロンプトを入力してください...")
        # リクエスト送信ボタン
        gemini_request_button = gr.Button("Gemini にリクエスト送信")
        # Gemini 応答表示
        gemini_response_output = gr.Textbox(label="Gemini からの応答", lines=10, interactive=False)

    # --- イベントハンドラの設定 ---
    # ファイルアップロードボタンがクリックされたら
    upload_button.click(
        fn=process_and_upload,
        inputs=file_input,
        outputs=[upload_output_message, gcs_file_dropdown] # アップロード結果とファイル選択プルダウンを更新
    )

    # GCSファイル一覧更新ボタンがクリックされたら
    refresh_files_button.click(
        fn=get_blob_list,
        inputs=None,
        outputs=gcs_file_dropdown # ファイル選択プルダウンを更新
    )

    # Gemini リクエストボタンがクリックされたら
    gemini_request_button.click(
        fn=gemini_request,
        inputs=[gcs_file_dropdown, prompt_input],
        outputs=gemini_response_output
    )

    # UI 起動時にGCSファイル一覧をロード
    demo.load(get_blob_list, inputs=None, outputs=gcs_file_dropdown)

# Gradio アプリケーションの起動
if __name__ == "__main__":
    demo.launch(server_port=8888, server_name="0.0.0.0", debug=True)