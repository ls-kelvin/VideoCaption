# video_pipeline/io/jsonl_consolidator.py
"""JSONL file consolidation utility for merging rank-sharded results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def consolidate_jsonl(output_path: str, world_size: int, keep_rank_files: bool = False) -> str:
    """
    Consolidate rank-sharded JSONL files into a single file.
    
    Args:
        output_path: Path to the consolidated output (e.g., 'output/result.jsonl')
        world_size: Number of ranks that were used during processing
        keep_rank_files: Whether to keep the individual rank files after consolidation
        
    Returns:
        Path to the consolidated file
    """
    root, ext = os.path.splitext(output_path)
    
    # Find all rank files
    rank_files = []
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    
    for rank in range(world_size):
        rank_file = f"{root}.rank{rank}{ext}"
        if os.path.exists(rank_file):
            rank_files.append((rank, rank_file))
        else:
            print(f"⚠️  Warning: Rank file not found: {rank_file}")
    
    if not rank_files:
        print("❌ No rank files found to consolidate")
        return output_path
    
    # Sort by rank
    rank_files.sort(key=lambda x: x[0])
    
    print(f"📦 Consolidating {len(rank_files)} rank files into {output_path}")
    
    # Count total lines for progress bar
    total_lines = 0
    for _, rank_file in rank_files:
        with open(rank_file, "r", encoding="utf-8") as f:
            total_lines += sum(1 for _ in f)
    
    # Merge all rank files into consolidated output
    with open(output_path, "w", encoding="utf-8") as out_f:
        with tqdm(total=total_lines, desc="Consolidating", unit="line") as pbar:
            for rank, rank_file in rank_files:
                with open(rank_file, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)
                        pbar.update(1)
    
    print(f"✅ Successfully consolidated to {output_path}")
    
    # Optionally remove rank files
    if not keep_rank_files:
        for _, rank_file in rank_files:
            try:
                os.remove(rank_file)
                print(f"🗑️  Removed rank file: {rank_file}")
            except Exception as e:
                print(f"⚠️  Failed to remove {rank_file}: {e}")
    
    return output_path
