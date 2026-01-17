import json
import glob
import os
from collections import defaultdict

def merge_captions():
    # File paths
    scene_result_file = "agirobot_scenes_result.jsonl"
    action_result_files = glob.glob("agirobot_actions_result.rank*.jsonl")
    output_file = "agirobot_final_captions.jsonl"

    if not os.path.exists(scene_result_file):
        print(f"Error: {scene_result_file} not found.")
        return

    # Data structures
    scenes = {}
    actions = defaultdict(list)

    print(f"Loading scenes from {scene_result_file}...")
    with open(scene_result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            meta = data['input']['meta']
            key = (str(meta['task_id']), str(meta['ep_id']))
            scenes[key] = data

    print(f"Found {len(scenes)} scenes.")

    print(f"Loading actions from {len(action_result_files)} files...")
    for action_file in action_result_files:
        print(f"  Reading {action_file}...")
        with open(action_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                meta = data['input']['meta']
                key = (str(meta['task_id']), str(meta['ep_id']))
                actions[key].append(data)

    print(f"Found actions for {len(actions)} videos.")

    # Sort actions by index
    for key in actions:
        actions[key].sort(key=lambda x: x['input']['meta']['action_idx'])

    # Merge
    results = []
    matched_count = 0
    
    # Iterate over scenes to ensure we have the base video info
    keys = sorted(scenes.keys())
    for key in keys:
        scene_data = scenes[key]
        task_id, ep_id = key
        
        # Get corresponding actions
        video_actions = actions.get(key, [])
        
        # Construct final object
        combined = {
            "video_id": f"{task_id}_{ep_id}",
            "meta": {
                "task_id": task_id,
                "ep_id": ep_id,
                "path": scene_data['input']['path']
            },
            "scene_description": scene_data.get('detailed_init_scene_text', scene_data.get('output_text', "")),
            "actions": []
        }
        
        for act in video_actions:
            act_info = {
                "action_idx": act['input']['meta']['action_idx'],
                "instruction": act['input']['raw_text'],
                "description": act.get('detailed_action_text', act.get('output_text', "")),
                "start_frame": act['input']['meta'].get('start_frame'),
                "end_frame": act['input']['meta'].get('end_frame')
            }
            combined['actions'].append(act_info)
            
        results.append(combined)
        matched_count += 1

    print(f"Merging complete. Writing {len(results)} records to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    merge_captions()
