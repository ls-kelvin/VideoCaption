import json

data = []

tasks = set()

with open("output/agirobot_result.jsonl", "r") as f:
    for line in f.readlines():
        item = json.loads(line.strip())
        task = item["input"]["path"].split("/")[-4]
        if task not in tasks:
            item["video_id"] = "-".join(item["input"]["path"].split("/")[-4:-2])
            data.append(item)
            tasks.add(task)
        
        
with open("output/agibot_result_sample.jsonl", "w") as f:
    for item in data:
        json.dump(item, f)
        f.write('\n')