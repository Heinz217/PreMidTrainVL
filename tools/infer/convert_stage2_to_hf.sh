#!/usr/bin/env bash
# 将 Stage 2 mcore checkpoint 转换为 HF 格式，然后运行 QA 推理测试
set -euo pipefail

source /jizhicfs/fenghengyi/miniconda3/etc/profile.d/conda.sh
conda activate innov

export AIAK_TRAINING_PATH=/jizhicfs/fenghengyi/Innovator-VL
export AIAK_MAGATRON_PATH=/jizhicfs/fenghengyi/Innovator-VL/aiak_megatron

MCORE_CKPT=/jizhicfs/fenghengyi/innovator_saves/step1/stage_2_instruct_innovator_vl_8b/iter_0000100
HF_OUTPUT=/jizhicfs/fenghengyi/innovator_saves/step1/stage_2_instruct_innovator_vl_8b_hf
TOKENIZER=/jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0

echo "=== Step 1: mcore -> HF 转换 ==="
cd /jizhicfs/fenghengyi/Innovator-VL
AIAK_TRAINING_PATH=$AIAK_TRAINING_PATH \
AIAK_MAGATRON_PATH=$AIAK_MAGATRON_PATH \
bash examples/innovator_vl/convert/convert_8b_mcore_to_hf.sh \
    "$MCORE_CKPT" \
    "$HF_OUTPUT" \
    8 \
    1

echo "=== Step 2: 复制 tokenizer 和配置文件 ==="
for f in tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt \
         config.json generation_config.json preprocessor_config.json \
         configuration_innovator_vl.py modeling_innovator_vl.py; do
    [ -f "$TOKENIZER/$f" ] && cp "$TOKENIZER/$f" "$HF_OUTPUT/" && echo "  copied $f"
done

echo "=== Step 3: 运行 QA 推理测试 ==="
PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:${PYTHONPATH:-}" \
python tools/infer/test_stage2_qa.py \
    --model_path "$HF_OUTPUT" \
    --data_dir   /jizhicfs/fenghengyi/data/innovator_instruct_energon \
    --save_image /tmp/test_sample.jpg

echo "=== 完成 ==="
