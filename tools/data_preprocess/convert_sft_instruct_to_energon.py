#!/usr/bin/env python3
"""
Convert Innovator-VL-Instruct parquet files to megatron/energon WebDataset format.

Parquet schema: id (str), images (ndarray of dicts with 'bytes'), conversations (list of from/value dicts)

Usage:
    python convert_sft_instruct_to_energon.py \
        --parquet_dir /jizhicfs/fenghengyi/data/Innovator-VL-Instruct-46M \
        --output_dir  /jizhicfs/fenghengyi/data/innovator_instruct_energon \
        --maxcount    5000 \
        --workers     32
"""
import argparse
import glob
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
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.prepare import WebdatasetPreparator

ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


def conversations_to_messages(conversations):
    """Convert from/value list to role/content list, preserving <image> placeholders."""
    messages = []
    for turn in conversations:
        role = ROLE_MAP.get(turn.get("from", ""), turn.get("from", ""))
        content = turn.get("value", "")
        messages.append({"role": role, "content": content})
    return messages


def write_energon_config(output_dir: str):
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

    sample_loader_code = """
def sample_loader(sample: dict) -> dict:
    messages = []
    system = None
    for message in sample['json']['texts']:
        assert message['role'] in ['system', 'user', 'assistant']
        if message['role'] == 'system':
            system = message['content']
            continue
        messages.append(dict(role=message['role'], content=message['content']))
    image = []
    if sample['json'].get('media') == 'image':
        for name in sample['json'].get('name', []):
            image.append(sample.get(name))
    return dict(
        __key__=sample['__key__'],
        __restore_key__=sample['__restore_key__'],
        video=None,
        image=image if image else None,
        system=system,
        messages=messages,
    )

def part_filter(part: str) -> bool:
    return True
"""
    with (meta_dir / "sample_loader.py").open("w") as f:
        f.write(sample_loader_code)

    all_tars = sorted([str(p.relative_to(path)) for p in path.glob("**/*.tar")])
    print(f"  Indexing {len(all_tars)} tar files ...")
    shard_counts = {}
    for tar in tqdm(all_tars, desc="indexing"):
        info, _ = WebdatasetPreparator._preprocess_tar(tar, parent_path=str(path), max_parts=50)
        if info.count > 0:
            shard_counts[info.name] = info.count

    with open(os.path.join(output_dir, ".nv-meta", ".info.yaml"), "w") as f:
        yaml.dump({"shard_counts": shard_counts}, f)

    import random
    shards = list(shard_counts.keys())
    random.Random(42).shuffle(shards)
    splits = {"split_parts": {"train": shards, "val": [], "test": []}, "exclude": []}
    with (meta_dir / "split.yaml").open("w") as f:
        yaml.dump(splits, f)

    print(f"  Done: {len(shard_counts)} shards, {sum(shard_counts.values())} samples")


def convert(args):
    os.makedirs(args.output_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(args.parquet_dir, "**/*.parquet"), recursive=True))
    print(f"Found {len(parquet_files)} parquet files")

    tar_pattern = os.path.join(args.output_dir, "sft-%06d.tar")
    skipped = 0
    total = 0

    with wds.ShardWriter(tar_pattern, maxcount=args.maxcount, maxsize=args.maxsize) as writer:
        for pf in parquet_files:
            print(f"Processing {pf} ...")
            df = pd.read_parquet(pf)
            for _, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(pf)):
                # 获取图片（可能有多张）
                images_field = row.get("images")
                img_keys = []
                img_data = {}

                if images_field is not None and len(images_field) > 0:
                    for i, img in enumerate(images_field):
                        if isinstance(img, dict):
                            img_bytes = img.get("bytes")
                        elif isinstance(img, bytes):
                            img_bytes = img
                        else:
                            img_bytes = None
                        if img_bytes:
                            key = f"img{i}.jpg"
                            img_keys.append(key)
                            img_data[key] = img_bytes

                if not img_keys:
                    skipped += 1
                    continue

                conversations = row.get("conversations", [])
                messages = conversations_to_messages(list(conversations))
                sample_id = str(row.get("id", f"sample_{total}")).replace("/", "_").replace(".", "_")

                content = {
                    "texts": messages,
                    "media": "image",
                    "name": img_keys,
                }
                sample = {
                    "__key__": sample_id,
                    **img_data,
                    "json": json.dumps(content).encode("utf-8"),
                }
                writer.write(sample)
                total += 1

    print(f"Written {total} samples, skipped {skipped}")
    print("Writing energon config ...")
    write_energon_config(args.output_dir)
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--maxcount",    type=int, default=5000)
    parser.add_argument("--maxsize",     type=int, default=3_000_000_000)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
