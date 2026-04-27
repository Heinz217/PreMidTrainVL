#!/usr/bin/env python3
"""
Stage 2 QA 推理测试脚本
从 innovator_instruct_energon 数据集中随机取一条样本（图片+问题），
使用转换好的 HF 格式模型进行推理，打印结果与 ground truth 对比。

用法：
    python tools/infer/test_stage2_qa.py \
        --model_path /jizhicfs/fenghengyi/innovator_saves/step1/stage_2_instruct_innovator_vl_8b_hf \
        --data_dir   /jizhicfs/fenghengyi/data/innovator_instruct_energon \
        [--sample_idx 42]   # 不指定则随机取
"""

import argparse
import io
import json
import os
import random
import sys
import tarfile

from PIL import Image


def extract_sample(data_dir: str, sample_idx: int = None):
    """从 energon tar 中提取一条样本"""
    tars = sorted(f for f in os.listdir(data_dir) if f.endswith(".tar"))
    if not tars:
        raise FileNotFoundError(f"No tar files found in {data_dir}")

    # 收集所有样本 key（json 文件名去掉扩展名）
    chosen_tar = random.choice(tars) if sample_idx is None else tars[0]
    tar_path = os.path.join(data_dir, chosen_tar)

    with tarfile.open(tar_path) as t:
        members = t.getnames()
        json_keys = sorted(m for m in members if m.endswith(".json"))

        if sample_idx is not None:
            idx = sample_idx % len(json_keys)
        else:
            idx = random.randint(0, len(json_keys) - 1)

        key = json_keys[idx]
        base = key[: -len(".json")]

        # 读 json
        meta = json.loads(t.extractfile(key).read())

        # 读图片（取第一张）
        img_name = meta["name"][0]  # e.g. "img0.jpg"
        img_key = f"{base}.{img_name}"
        img_bytes = t.extractfile(img_key).read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        return image, meta["texts"], tar_path, key


def run_inference(model_path: str, image: Image.Image, conversations: list):
    """使用 HF 格式的 Innovator-VL 模型推理"""
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch

    print(f"\n[加载模型] {model_path}")
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # 取第一个 user 轮作为输入，第一个 assistant 轮作为 ground truth
    user_turn = next(t for t in conversations if t["role"] == "user")
    gt_turn = next(t for t in conversations if t["role"] == "assistant")

    question_text = user_turn["content"]
    ground_truth = gt_turn["content"]

    print(f"\n[问题]\n{question_text}")
    print(f"\n[Ground Truth]\n{ground_truth}")

    # Qwen2-VL 风格：手动构建带图像占位符的输入文本
    content_text = question_text.replace("<image>\n", "").replace("<image>", "").strip()

    # Qwen2VL 格式：<|vision_start|><|image_pad|><|vision_end|> 作为图像占位
    text_input = (
        f"<|im_start|>user\n"
        f"<|vision_start|><|image_pad|><|vision_end|>\n"
        f"{content_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = processor(
        text=[text_input],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    # 只取新生成的 token
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    answer = processor.tokenizer.decode(generated, skip_special_tokens=True)

    print(f"\n[模型回答]\n{answer}")
    return answer, ground_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        required=True,
        help="HF 格式模型路径（mcore→HF 转换后的目录）",
    )
    parser.add_argument(
        "--data_dir",
        default="/jizhicfs/fenghengyi/data/innovator_instruct_energon",
        help="energon 数据集目录",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="指定样本索引（不指定则随机）",
    )
    parser.add_argument(
        "--save_image",
        default=None,
        help="保存提取图片到指定路径（可选）",
    )
    args = parser.parse_args()

    # 1. 提取样本
    print(f"[数据目录] {args.data_dir}")
    image, conversations, tar_path, key = extract_sample(args.data_dir, args.sample_idx)
    print(f"[样本来源] {tar_path} / {key}")
    print(f"[图片尺寸] {image.size}")

    if args.save_image:
        image.save(args.save_image)
        print(f"[图片已保存] {args.save_image}")

    # 2. 推理
    answer, ground_truth = run_inference(args.model_path, image, conversations)

    # 3. 简单评测
    correct = answer.strip().upper() == ground_truth.strip().upper()
    print(f"\n[结果] {'✓ 正确' if correct else '✗ 错误'}")


if __name__ == "__main__":
    main()
