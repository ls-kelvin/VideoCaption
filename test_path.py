#!/usr/bin/env python3
"""
检查 jsonl 中每一项的 path 属性对应的文件是否存在，
把不存在的条目写入新的 jsonl 文件。
用法:
    python check_paths.py input.jsonl missing.jsonl
"""

import json
import os
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("用法: python check_paths.py <input.jsonl> <output_missing.jsonl>")
        sys.exit(1)

    in_file, out_file = sys.argv[1], sys.argv[2]

    if not os.path.isfile(in_file):
        print(f"输入文件不存在: {in_file}")
        sys.exit(1)

    missing = []

    with open(in_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"跳过无效 JSON 第{line_no}行: {e}")
                continue

            path = data.get("path")
            if path is None:
                print(f"跳过第{line_no}行（无 path 字段）")
                continue

            if os.path.isfile(path):
                missing.append(data)

    with open(out_file, "w", encoding="utf-8") as f:
        for item in missing:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"检查完成，共 {len(missing)} 条路径不存在，已保存到 {out_file}")

if __name__ == "__main__":
    main()