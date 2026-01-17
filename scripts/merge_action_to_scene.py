import json
import os
import glob
from tqdm import tqdm

ACTION_RESULT_PATTERN = "agirobot_actions_result.rank*.jsonl"
SCENE_INPUT_FILE = "agirobot_scenes.jsonl"
SCENE_OUTPUT_FILE = "agirobot_scenes_merged.jsonl"

def load_first_actions():
    """
    从动作生成的输出文件中，提取每个 Episode 第一个动作的详细描述。
    """
    print(f"Loading action results from {ACTION_RESULT_PATTERN}...")
    files = glob.glob(ACTION_RESULT_PATTERN)
    if not files:
        print("Warning: No action result files found!")
        return {}

    # Map: f"{task_id}_{ep_id}" -> detailed_action_text
    first_action_map = {}

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # 检查是否是第一个动作 (action_idx == 0)
                    meta = record["input"]["meta"]
                    if meta.get("action_idx") == 0:
                        task_id = meta.get("task_id")
                        ep_id = meta.get("ep_id")
                        key = f"{task_id}_{ep_id}"
                        
                        # 获取生成的详细描述
                        detailed_text = record.get("detailed_action_text", "")
                        if detailed_text:
                            first_action_map[key] = detailed_text
                except Exception as e:
                    print(f"Error parsing line in {file_path}: {e}")
                    continue
    
    print(f"Loaded {len(first_action_map)} first-action descriptions.")
    return first_action_map

def merge_scenes(action_map):
    """
    读取场景输入文件，用 action_map 中的详细描述替换原始描述。
    """
    if not os.path.exists(SCENE_INPUT_FILE):
        print(f"Error: Scene input file {SCENE_INPUT_FILE} not found.")
        return

    print(f"Merging into {SCENE_OUTPUT_FILE}...")
    
    count_updated = 0
    total = 0
    
    with open(SCENE_INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(SCENE_OUTPUT_FILE, "w", encoding="utf-8") as fout:
        
        for line in tqdm(fin, desc="Processing scenes"):
            total += 1
            try:
                data = json.loads(line)
                meta = data["meta"]
                task_id = meta.get("task_id")
                ep_id = meta.get("ep_id")
                key = f"{task_id}_{ep_id}"
                
                # 如果能在 action_map 中找到对应的生成结果，则替换
                if key in action_map:
                    data["first_action_text"] = action_map[key]
                    count_updated += 1
                
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error processing scene line: {e}")
                
    print(f"Done. Updated {count_updated}/{total} scenes.")

if __name__ == "__main__":
    action_map = load_first_actions()
    merge_scenes(action_map)
