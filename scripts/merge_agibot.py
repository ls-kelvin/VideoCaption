import json

def merge_jsonl_files(file1_path, file2_path, output_path):
    """
    合并两个jsonl文件：
    将file2中的detailed_init_scene_text字符串拼接到file1中detailed_action_captions的第一项字符串后面
    """
    
    # 读取第一个jsonl文件
    with open(file1_path, 'r', encoding='utf-8') as f1:
        data1 = [json.loads(line.strip()) for line in f1 if line.strip()]
    
    # 读取第二个jsonl文件
    with open(file2_path, 'r', encoding='utf-8') as f2:
        data2 = [json.loads(line.strip()) for line in f2 if line.strip()]
    
    # 检查两个文件的行数是否相同
    if len(data1) != len(data2):
        print(f"警告：两个文件行数不一致！file1: {len(data1)}行, file2: {len(data2)}行")
        print("将按行号对应处理，多出的行将被忽略")
    
    # 合并数据
    merged_data = []
    min_length = min(len(data1), len(data2))
    
    for i in range(min_length):
        item1 = data1[i].copy()  # 创建副本避免修改原始数据
        item2 = data2[i]
        
        # 检查必要的字段是否存在
        if 'detailed_action_captions' not in item1:
            print(f"警告：第{i+1}行缺少'detailed_action_captions'字段，跳过该行")
            continue
        
        if 'detailed_init_scene_text' not in item2:
            print(f"警告：第{i+1}行缺少'detailed_init_scene_text'字段，跳过该行")
            continue
        
        # 确保detailed_action_captions是列表类型且不为空
        if not isinstance(item1['detailed_action_captions'], list):
            print(f"警告：第{i+1}行的'detailed_action_captions'不是列表类型，转换为列表")
            item1['detailed_action_captions'] = [str(item1['detailed_action_captions'])]
        
        if len(item1['detailed_action_captions']) == 0:
            print(f"警告：第{i+1}行的'detailed_action_captions'列表为空，跳过")
            continue
        else:
            # 将detailed_init_scene_text拼接到detailed_action_captions的第一个元素后面
            item1['detailed_action_captions'][0] = item2['detailed_init_scene_text'] + item1['detailed_action_captions'][0]
        
        merged_data.append(item1)
    
    # 保存到输出文件
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for item in merged_data:
            out_file.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"合并完成！共处理{len(merged_data)}行数据")
    print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    # 设置文件路径
    file1_path = "output/agirobot_actions_result.jsonl"  # 包含detailed_action_captions的文件
    file2_path = "output/agirobot_scenes_result.jsonl"  # 包含detailed_init_scene_text的文件
    output_path = "output/agirobot_result.jsonl"
    
    # 调用合并函数
    merge_jsonl_files(file1_path, file2_path, output_path)