#!/usr/bin/env python3
# coding=utf-8
"""
Patch _extra_state keys in a Megatron-mcore checkpoint directory.

WHY
---
Transformer Engine has two coexisting implementations of `set_extra_state`:
  * transformer_engine/pytorch/module/base.py  -> accepts BytesIO OR Tensor
  * transformer_engine/pytorch/ops/op.py       -> Tensor only (BytesIO triggers
                                                  `AttributeError: BytesIO has no attribute 'numel'`)

Old convert scripts (e.g. merge_megatron_qwen3_30b_a3b.py) write `_extra_state`
values as BytesIO objects (essentially `BytesIO(torch.save(None))`). When the
checkpoint is loaded into a model whose modules go through the newer `ops/op.py`
path -- which includes MoE experts (`linear_fc1/linear_fc2`) and the PatchMerger
adapter -- training crashes at load time.

This script normalises every `_extra_state` value in every shard to an empty
1-D `uint8` tensor, which both TE implementations short-circuit on
(`numel() == 0  ->  return`).

USAGE
-----
    python tools/patch_extra_state_qwen3moe.py \
        --ckpt_dir /path/to/<model>_mcore_tp*_pp*_ep*

    # The script will recurse through release/ (or any iter_*/) automatically.

Tested on: Innovator-VL-30B-A3B (TP=1, PP=1, EP=8) Megatron mcore checkpoints.
"""

import argparse
import glob
import os
import sys
import time

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Patch _extra_state in mcore checkpoints")
    p.add_argument(
        "--ckpt_dir",
        required=True,
        help="Path to the mcore checkpoint root "
             "(the directory that contains 'release/' or 'iter_*/').",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only inspect; do not rewrite files.",
    )
    return p.parse_args()


def find_shards(ckpt_dir: str):
    """Return all model_optim_rng.pt files under ckpt_dir."""
    patterns = [
        os.path.join(ckpt_dir, "release", "mp_rank_*", "model_optim_rng.pt"),
        os.path.join(ckpt_dir, "iter_*", "mp_rank_*", "model_optim_rng.pt"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(files)


def patch_one(fpath: str, dry_run: bool) -> tuple[int, int]:
    """Return (n_extra_state_keys, n_patched_keys)."""
    EMPTY = torch.tensor([], dtype=torch.uint8)

    sd = torch.load(fpath, map_location="cpu", weights_only=False)
    model = sd["model"]

    n_total = 0
    n_patched = 0
    for k in list(model.keys()):
        if k.endswith("._extra_state") or k == "_extra_state":
            n_total += 1
            v = model[k]
            need_patch = (not isinstance(v, torch.Tensor)) or (v.numel() != 0)
            if need_patch:
                if not dry_run:
                    model[k] = EMPTY
                n_patched += 1

    if (not dry_run) and n_patched > 0:
        torch.save(sd, fpath)

    return n_total, n_patched


def main():
    args = parse_args()
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        sys.exit(f"[ERROR] ckpt_dir does not exist: {ckpt_dir}")

    files = find_shards(ckpt_dir)
    if not files:
        sys.exit(f"[ERROR] no model_optim_rng.pt found under {ckpt_dir}")

    mode = "DRY-RUN" if args.dry_run else "PATCH"
    print(f"=== {mode} _extra_state in {ckpt_dir} ===")
    print(f"    found {len(files)} shard file(s)")
    print()

    grand_total = 0
    grand_patched = 0
    t_start = time.time()
    for i, fpath in enumerate(files, 1):
        rel = os.path.relpath(fpath, ckpt_dir)
        t0 = time.time()
        n_total, n_patched = patch_one(fpath, args.dry_run)
        dt = time.time() - t0
        print(
            f"  [{i:>2}/{len(files)}] {rel}  "
            f"_extra_state keys: {n_total:>5}, "
            f"{'would patch' if args.dry_run else 'patched'}: {n_patched:>5}  "
            f"({dt:5.1f}s)"
        )
        grand_total += n_total
        grand_patched += n_patched

    dt_all = time.time() - t_start
    print()
    print(f"=== summary ===")
    print(f"  files processed   : {len(files)}")
    print(f"  total _extra_state: {grand_total}")
    print(f"  total patched     : {grand_patched}")
    print(f"  elapsed           : {dt_all:.1f}s")

    if args.dry_run:
        print(f"\n  [dry-run] no file was modified.")
    else:
        print(f"\n  [done] all shards patched in-place.")


if __name__ == "__main__":
    main()
