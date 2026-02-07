import json

# 配置参数
input_file = 'agibot-alpha-2.jsonl'
output_file = 'agibot-alpha.jsonl'
old_prefix = '/inspire/qb-ilm/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/zzt'
new_prefix = '/root/workspace/zzt/data'

# 处理文件
with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:

    for line in fin:
        if not line.strip():
            continue  # 跳过空行
        data = json.loads(line)
        
        # 确保 input.path 存在
        if 'path' in data:
            data['path'] = data['path'].replace(old_prefix, new_prefix)
        
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')