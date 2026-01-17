"""
脚本功能：
1. 扫描 VideoCaption/agirobot_caption.py 中描述的目录结构。
2. 执行视频 3 倍速加速（cached）。
3. 根据 action_config 截取动作所需的片段（cached）。
4. 生成 VideoCaption 框架所需的 .jsonl 输入文件：
   - agirobot_actions.jsonl
   - agirobot_scenes.jsonl
"""
import os
import json
import time
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import imageio_ffmpeg as mpg
from concurrent.futures import ThreadPoolExecutor

# 常量定义 (需根据实际路径修改)
VIDEO_ROOT_DIR = "/data/flg/agirobot/dataset/data/bot_sample"
TASK_INFO_DIR = "/data/flg/agirobot/dataset/AgiBotWorld-task_info/task_info"
SPEED_VIDEO_ROOT_DIR = os.path.join(os.path.dirname(VIDEO_ROOT_DIR), "bot_sample_speedup_pipeline")
FFMPEG_TIMEOUT = 60
VIDEO_SPEED_RATIO = 3
TARGET_FPS = 30
DEFAULT_VIDEO_FPS = 30

FFMPEG_EXE = mpg.get_ffmpeg_exe()

def extract_first_frame(video_path, save_path):
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
        # 已存在，跳过
        return save_path

    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [FFMPEG_EXE, "-y", "-i", str(video_path),
             "-vframes", "1", "-q:v", "2",
             save_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=FFMPEG_TIMEOUT
        )
        return save_path
    except Exception as e:
        print(f"Error extracting frame {video_path}: {e}")
        if os.path.exists(save_path): os.unlink(save_path)
        return None

def get_video_fps(video_path):
    try:
        result = subprocess.check_output(
            [FFMPEG_EXE, "-i", str(video_path), "-v", "error",
             "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1"],
            stderr=subprocess.STDOUT, timeout=5
        ).decode("utf-8").strip()
        if '/' in result:
            num, den = result.split('/')
            return float(num) / float(den) if den != '0' else DEFAULT_VIDEO_FPS
        return float(result) if result else DEFAULT_VIDEO_FPS
    except Exception:
        return DEFAULT_VIDEO_FPS

def speed_up_video(video_path, save_path, speed_ratio=3, target_fps=30):
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
        # 已存在，跳过
        return save_path

    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [FFMPEG_EXE, "-y", "-i", str(video_path),
             "-filter:v", f"setpts=PTS/{speed_ratio},fps={target_fps}",
             "-filter:a", f"atempo={speed_ratio}",
             "-c:v", "libx264", "-c:a", "aac",
             "-pix_fmt", "yuv420p", "-loglevel", "error",
             save_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=FFMPEG_TIMEOUT
        )
        return save_path
    except Exception as e:
        print(f"Error speeding up {video_path}: {e}")
        if os.path.exists(save_path): os.unlink(save_path)
        return None

def cut_video_segment(video_path, start_frame, end_frame, output_path):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        return output_path
    
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    fps = get_video_fps(video_path)
    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame + 1) / fps
    if duration_sec < 0.5: duration_sec = 1.0

    try:
        subprocess.run(
            [FFMPEG_EXE, "-y", "-ss", f"{start_sec:.3f}", "-i", str(video_path),
             "-t", f"{duration_sec:.3f}", "-c:v", "libx264", "-c:a", "aac",
             "-pix_fmt", "yuv420p", "-loglevel", "error", output_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=FFMPEG_TIMEOUT
        )
        return output_path
    except Exception as e:
        print(f"Error cutting {video_path}: {e}")
        if os.path.exists(output_path): os.unlink(output_path)
        return None

def adjust_frame_range(frame_num, speed_ratio=3):
    return max(0, int(frame_num / speed_ratio))

def load_task_info():
    task_map = {}
    if not os.path.exists(TASK_INFO_DIR):
        print(f"Warning: Task info dir not found: {TASK_INFO_DIR}")
        return {}
    
    files = [f for f in os.listdir(TASK_INFO_DIR) if f.startswith("task_") and f.endswith(".json")]
    for file in files:
        task_id = file.replace("task_", "").replace(".json", "")
        with open(os.path.join(TASK_INFO_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            task_map[task_id] = {str(ep["episode_id"]): ep for ep in data}
    return task_map

import threading

file_lock = threading.Lock()

def process_single_video(video_path, task_map, speed_root, action_writer, scene_writer):
    # 路径解析

    ep_dir = video_path.parent.parent
    ep_id = ep_dir.name
    task_id = ep_dir.parent.name
    
    if task_id not in task_map or ep_id not in task_map[task_id]:
        return
    
    ep_data = task_map[task_id][ep_id]
    
    # 1. Speedup
    rel_path = os.path.relpath(video_path, VIDEO_ROOT_DIR)
    speed_path = os.path.join(speed_root, rel_path).replace(".mp4", "_speedup.mp4")
    if not speed_up_video(video_path, speed_path, VIDEO_SPEED_RATIO, TARGET_FPS):
        return

    # 2. Process Actions
    actions = ep_data["label_info"]["action_config"]
    processed_actions_for_scene = [] # 存储用于生成场景描述的第一个动作信息

    for idx, action in enumerate(actions):
        raw_start = action["start_frame"]
        raw_end = action["end_frame"]
        new_start = adjust_frame_range(raw_start, VIDEO_SPEED_RATIO)
        new_end = adjust_frame_range(raw_end, VIDEO_SPEED_RATIO)
        if new_start >= new_end: new_end = new_start + 1

        # 缓存片段文件
        segment_filename = f"{task_id}_{ep_id}_action_{idx}.mp4"
        segment_path = os.path.join(os.path.dirname(speed_path), "segments", segment_filename)
        
        if cut_video_segment(speed_path, new_start, new_end, segment_path):
             # 写入 Action JSONL (供 vLLM 推理)
             # 包含 meta 信息以便后期合并
             sample = {
                 "path": str(segment_path),
                 "raw_text": action.get("action_text", ""),
                 "meta": {
                     "type": "action",
                     "task_id": task_id,
                     "ep_id": ep_id,
                     "action_idx": idx,
                     "start_frame": new_start,
                     "end_frame": new_end
                 }
             }
             with file_lock:
                 action_writer.write(json.dumps(sample, ensure_ascii=False) + "\n")
             
             if idx == 0:
                 processed_actions_for_scene.append(action.get("action_text", ""))


    # 3. Process Scene
    # 提取首帧图片用于 Scene Caption
    scene_image_path = speed_path.replace(".mp4", "_first_frame.jpg")
    extract_first_frame(speed_path, scene_image_path)

    scene_sample = {
        "path": str(scene_image_path),
        "raw_text": ep_data.get("init_scene_text", ""),
        "first_action_text": processed_actions_for_scene[0] if processed_actions_for_scene else "",
        "is_image": True,
        "meta": {
            "type": "scene",
            "task_id": task_id,
            "ep_id": ep_id,
            "is_image": True  # 标记为图片输入
        }
    }
    with file_lock:
        scene_writer.write(json.dumps(scene_sample, ensure_ascii=False) + "\n")



def main():
    parser = argparse.ArgumentParser(description="Prepare data for AgiRobot Video Captioning")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers for FFmpeg processing")
    args = parser.parse_args()

    if not os.path.exists(VIDEO_ROOT_DIR):
        print("Video root dir not found")
        return
        
    task_map = load_task_info()
    
    # 收集视频
    video_list = []
    for task_dir in os.listdir(VIDEO_ROOT_DIR):
        t_path = os.path.join(VIDEO_ROOT_DIR, task_dir)
        if not os.path.isdir(t_path): continue
        for ep_dir in os.listdir(t_path):
            e_path = os.path.join(t_path, ep_dir)
            if not os.path.isdir(e_path): continue
            v_path = os.path.join(e_path, "videos", "head_color.mp4")
            if os.path.exists(v_path):
                video_list.append(Path(v_path))
    
    print(f"Found {len(video_list)} videos. Using {args.num_workers} parallel workers.")
    
    # 打开输出文件
    with open("agirobot_actions.jsonl", "w", encoding="utf-8") as fa, \
         open("agirobot_scenes.jsonl", "w", encoding="utf-8") as fs:
        
        # 使用 ThreadPool 并行处理 ffmpeg (CPU/IO Bound)
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            list(tqdm(executor.map(lambda v: process_single_video(v, task_map, SPEED_VIDEO_ROOT_DIR, fa, fs), video_list), total=len(video_list)))

    print("Done. Created agirobot_actions.jsonl and agirobot_scenes.jsonl")

if __name__ == "__main__":
    main()
