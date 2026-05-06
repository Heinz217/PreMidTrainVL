#!/usr/bin/env python3
"""
Convert LLaVA-OneVision-1.5-Mid-Training parquet files to megatron/energon WebDataset format.

Parquet schema: id (str), image (dict with 'bytes'/'path'), caption (str)

Usage:
    python convert_mid_training_to_energon.py \
        --parquet_dir /jizhicfs/fenghengyi/data/llava_mid_training_subset \
        --output_dir  /jizhicfs/fenghengyi/data/llava_mid_training_energon \
        --maxcount    5000 \
        --workers     32
"""
import argparse
import glob
import io
import json
import os
import sys
import yaml
import pandas as pd
import webdataset as wds
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in [
    os.path.join(_project_root, "aiak_megatron"),
    os.path.join(_project_root, "aiak_megatron", "megatron"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.prepare import WebdatasetPreparator


def write_energon_config(output_dir: str, num_workers: int = 32):
    path = EPath(output_dir).absolute()
    meta_dir = path / MAIN_FOLDER_NAME
    meta_dir.mkdir(exist_ok=True)

    dataset_definition = {
        "sample_type": {
            "__module__": "aiak_training_llm.data.multimodal",
            "__class__": "MultiMixQASample",
        },
        "part_filter": "sample_loader.py:part_filter",
        "sample_loader": "sample_loader.py:sample_loader",
    }
    with (meta_dir / "dataset.yaml").open("w") as f:
        yaml.dump(dataset_definition, f, sort_keys=False)

    # caption 数据：user 发图片，assistant 回答 caption
    sample_loader_code = """
def sample_loader(sample: dict) -> dict:
    messages = [
        {'role': 'user',      'content': '<image>'},
        {'role': 'assistant', 'content': sample['json']['caption']},
    ]
    img_key = sample['json'].get('img_key', 'image.jpg')
    image = [sample.get(img_key)]
    return dict(
        __key__=sample['__key__'],
        __restore_key__=sample['__restore_key__'],
        video=None,
        image=image,
        system=None,
        messages=messages,
    )

def part_filter(part: str) -> bool:
    return True
"""
    with (meta_dir / "sample_loader.py").open("w") as f:
        f.write(sample_loader_code)

    all_tars = sorted([str(p.relative_to(path)) for p in path.glob("**/*.tar")])
    print(f"  Preparing energon index for {len(all_tars)} tar files ...")

    # 单进程统计（避免多进程 bug）
    shard_counts = {}
    for tar in tqdm(all_tars, desc="indexing"):
        info, _ = WebdatasetPreparator._preprocess_tar(tar, parent_path=str(path), max_parts=50)
        if info.count > 0:
            shard_counts[info.name] = info.count

    with open(os.path.join(output_dir, ".nv-meta", ".info.yaml"), "w") as f:
        yaml.dump({"shard_counts": shard_counts}, f)

    # split.yaml
    import random
    shards = list(shard_counts.keys())
    random.Random(42).shuffle(shards)
    splits = {"split_parts": {"train": shards, "val": [], "test": []}, "exclude": []}
    with (meta_dir / "split.yaml").open("w") as f:
        yaml.dump(splits, f)

    print(f"  .nv-meta written. Total: {len(shard_counts)} shards, {sum(shard_counts.values())} samples")


def convert(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 收集所有 parquet 文件（递归）
    parquet_files = sorted(glob.glob(os.path.join(args.parquet_dir, "**/*.parquet"), recursive=True))
    print(f"Found {len(parquet_files)} parquet files")

    tar_pattern = os.path.join(args.output_dir, "mid-train-%06d.tar")
    skipped = 0
    total = 0

    with wds.ShardWriter(tar_pattern, maxcount=args.maxcount, maxsize=args.maxsize) as writer:
        for pf in parquet_files:
            print(f"Processing {pf} ...")
            df = pd.read_parquet(pf)
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(pf)):
                # 获取图片 bytes
                img_field = row.get("image")
                if img_field is None:
                    skipped += 1
                    continue
                if isinstance(img_field, dict):
                    img_bytes = img_field.get("bytes")
                elif isinstance(img_field, bytes):
                    img_bytes = img_field
                else:
                    skipped += 1
                    continue
                if not img_bytes:
                    skipped += 1
                    continue

                caption = str(row.get("caption", ""))
                sample_id = str(row.get("id", f"{os.path.basename(pf)}_{idx}")).replace("/", "_").replace(".", "_")
                img_key = "image.jpg"

                content = {"caption": caption, "img_key": img_key}
                sample = {
                    "__key__": sample_id,
                    img_key: img_bytes,
                    "json": json.dumps(content).encode("utf-8"),
                }
                writer.write(sample)
                total += 1

    print(f"Written {total} samples, skipped {skipped}")
    print("Writing energon config and index ...")
    write_energon_config(args.output_dir, num_workers=args.workers)
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", required=True, help="Directory containing parquet files (recursive)")
    parser.add_argument("--output_dir",  required=True, help="Output energon WebDataset directory")
    parser.add_argument("--maxcount",    type=int, default=5000)
    parser.add_argument("--maxsize",     type=int, default=3_000_000_000)
    parser.add_argument("--workers",     type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
