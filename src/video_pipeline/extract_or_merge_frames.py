import subprocess
import os
from config import *

def extract_frames(
        video_dir, filename, output_dir, 
        num_frame_per_second=1,
        qv=2,
        start_number=0,
    ):
    """
    ビデオを指定されたディレクトリにフレームとして抽出します。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ffmpegコマンドを使用してビデオからフレームを抽出
    command = [
        'ffmpeg', '-i', f"{video_dir}/{filename}",
        "-r", str(num_frame_per_second),
        '-q:v', str(qv),
        '-start_number', str(start_number),
        f'{output_dir}/frame_%05d.png'
    ]
    
    subprocess.run(command, check=True)

def merge_frames(
        frame_dir, output_dir, filename,
        num_frame_per_second=1,
        start_number=0
    ):
    """
    フレームを結合して動画にする
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ffmpegコマンドを使用してビデオからフレームを抽出
    command = [
        'ffmpeg', '-framerate', str(num_frame_per_second), 
        '-start_number', str(start_number),
        '-i', f"{frame_dir}/frame_%05d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        f'{output_dir}/{filename}'
    ]
    
    subprocess.run(command, check=True)

# extract_frames(video_dir=DATA_DIR, filename="taskalfa.mp4", output_dir=f'{OUTPUT_DIR}/keyframes')
# merge_frames(frame_dir=f'{OUTPUT_DIR}/keyframes', output_dir=f"{OUTPUT_DIR}/crops", filename="taskalfa_merged.mp4")