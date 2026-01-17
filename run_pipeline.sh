#!/bin/bash
set -e  # 遇到错误立即退出

# 配置
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}  # 默认使用所有 GPU，可使用 GPUS=5,6,7 ./run_pipeline.sh 覆盖

echo "=================================================="
echo "Starting AgiRobot Video Captioning Pipeline"
echo "Using GPUs: $GPUS"
echo "=================================================="

# 1. 数据准备
echo "[Step 1/5] Preparing Data (Cutting video segments)..."
# 生成 agirobot_actions.jsonl 和 agirobot_scenes.jsonl
python scripts/prepare_agirobot_data.py

# 2. 动作推理
echo "[Step 2/5] Running Action Captioning Inference..."
# 清理旧的动作结果，防止混淆
rm -f agirobot_actions_result.rank*.jsonl
python -m video_pipeline.cli.launch \
    --config configs/agirobot_action.yaml \
    --gpu-ids $GPUS

# 3. 上下文合并
echo "[Step 3/5] Merging Action Context into Scene Inputs..."
# 读取 action 结果，更新 scene 的 prompt
python scripts/merge_action_to_scene.py

# 4. 场景推理
echo "[Step 4/5] Running Scene Captioning Inference..."
# 清理旧的场景结果
rm -f agirobot_scenes_result.rank*.jsonl
python -m video_pipeline.cli.launch \
    --config configs/agirobot_scene.yaml \
    --gpu-ids $GPUS

# 5. 最终合并
echo "[Step 5/5] Merging Final Captions..."
# 聚合所有结果到 output jsonl
python scripts/merge_final_video_captions.py

echo "=================================================="
echo "Pipeline Completed Successfully!"
echo "Final Output: agirobot_final_captions.jsonl"
echo "=================================================="
