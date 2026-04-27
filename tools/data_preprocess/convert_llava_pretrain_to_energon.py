#!/usr/bin/env python3
"""
Convert LLaVA-558K pretraining data to megatron/energon WebDataset format.

Usage:
    python convert_llava_pretrain_to_energon.py \
        --json_file  /jizhicfs/fenghengyi/data/llava_pretrain/blip_laion_cc_sbu_558k.json \
        --image_dir  /jizhicfs/fenghengyi/data/llava_pretrain \
        --output_dir /jizhicfs/fenghengyi/data/llava_pretrain_energon \
        --maxcount   5000 \
        --workers    32
"""
import argparse
import json
import os
import sys
import yaml
import webdataset as wds
from tqdm import tqdm

# Allow importing megatron.energon from the project tree
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
for _p in [
    os.path.join(_project_root, "aiak_megatron"),
    os.path.join(_project_root, "aiak_megatron", "megatron"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME


ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


def conversations_to_messages(conversations):
    """Convert LLaVA-style from/value list to role/content list.

    The <image> placeholder must be preserved: the training framework uses it
    to locate where each image should be inserted in the token sequence.
    """
    messages = []
    for turn in conversations:
        role = ROLE_MAP.get(turn.get("from", ""), turn.get("from", ""))
        content = turn.get("value", "")
        messages.append({"role": role, "content": content})
    return messages


def write_energon_config(output_dir: str, num_workers: int = 32):
    """Write .nv-meta/dataset.yaml + sample_loader.py and call prepare_dataset."""
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
    print(f"  Preparing energon index for {len(all_tars)} tar files …")
    BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=[("train", 1.0), ("val", 0.0), ("test", 0.0)],
        tar_index_only=False,
        workers=num_workers,
    )
    print(f"  .nv-meta written to {meta_dir}")


def convert(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.json_file} …")
    with open(args.json_file) as f:
        data = json.load(f)
    print(f"  {len(data)} samples")

    tar_pattern = os.path.join(args.output_dir, "pretrain-%06d.tar")
    skipped = 0
    with wds.ShardWriter(tar_pattern, maxcount=args.maxcount, maxsize=args.maxsize) as writer:
        for idx, entry in enumerate(tqdm(data, desc="converting")):
            image_rel = entry.get("image")
            if not image_rel:
                skipped += 1
                continue
            image_path = os.path.join(args.image_dir, image_rel)
            if not os.path.exists(image_path):
                skipped += 1
                continue
            with open(image_path, "rb") as img_f:
                image_bytes = img_f.read()

            img_key = "0_" + os.path.basename(image_rel)
            messages = conversations_to_messages(entry.get("conversations", []))
            content = {
                "texts": messages,
                "media": "image",
                "name": [img_key],
            }
            sample = {
                "__key__": entry.get("id", str(idx)).replace(".", "_"),
                img_key: image_bytes,
                "json": json.dumps(content).encode("utf-8"),
            }
            writer.write(sample)

    print(f"Skipped {skipped} entries (missing image)")
    print("Writing energon config and index …")
    write_energon_config(args.output_dir, num_workers=args.workers)
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file",   required=True)
    parser.add_argument("--image_dir",   required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--maxcount",    type=int, default=5000,
                        help="Max samples per tar shard")
    parser.add_argument("--maxsize",     type=int, default=3_000_000_000,
                        help="Max bytes per tar shard")
    parser.add_argument("--workers",     type=int, default=32,
                        help="Workers for energon index preparation")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
