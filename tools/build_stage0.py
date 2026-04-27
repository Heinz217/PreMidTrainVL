#!/usr/bin/env python3
"""
Build Innovator-VL stage0 model from scratch.

Verified against InnovatorLab/Innovator-VL-8B-Instruct:
  - LLM  : 399 keys identical to Qwen3-8B  (model.* + lm_head.*)
  - ViT  : 288 keys  (visual.blocks.0~23.*)
  - Extra: 11 keys   (visual.class_embedding, visual.class_pos_emb,
                      visual.patch_embed.*, visual.pre_layernorm.*,
                      visual.merger.*)

Usage
-----
  cd /jizhicfs/fenghengyi/Innovator-VL
  python tools/build_stage0.py \
      --llm_path  /jizhicfs/fenghengyi/models/Qwen3-8B \
      --save_path /jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0
"""

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---- Architecture constants (verified from official model) -----------------
RICE_DEPTH          = 24
RICE_DIM            = 1024   # embed_dim / hidden_size
RICE_NUM_HEADS      = 16
RICE_FFN            = 4096   # intermediate_size = DIM * 4
RICE_PATCH_SIZE     = 14
RICE_IN_CHANNELS    = 3
RICE_TEMPORAL_PATCH = 1
RICE_SPATIAL_MERGE  = 2
# PatchMerger: input = DIM * SPATIAL_MERGE^2 = 1024*4 = 4096
MERGER_HIDDEN       = RICE_DIM * RICE_SPATIAL_MERGE * RICE_SPATIAL_MERGE  # 4096
# LLM hidden_size (Qwen3-8B) = PatchMerger output dim
TEXT_HIDDEN_SIZE    = 4096
# class_pos_emb shape: (1, head_dim // 2) = (1, (1024//16)//2) = (1, 32)
CLASS_POS_EMB_DIM   = (RICE_DIM // RICE_NUM_HEADS) // 2   # 32


def parse_args():
    p = argparse.ArgumentParser(description="Build Innovator-VL stage0 from Qwen3-8B")
    p.add_argument("--llm_path",      required=True, help="Qwen3-8B HF checkpoint dir")
    p.add_argument("--save_path",     required=True, help="Output directory for stage0")
    p.add_argument("--rice_vit_path", default=None,
                   help="Optional: MLCD/RICE-ViT HF checkpoint dir. "
                        "If provided, loads pretrained ViT weights instead of random init.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


DTYPE_MAP = {"float32": torch.float32,
             "float16": torch.float16,
             "bfloat16": torch.bfloat16}


# ---------------------------------------------------------------------------
def load_llm(llm_path: str, dtype: torch.dtype) -> dict:
    files = sorted(f for f in os.listdir(llm_path) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {llm_path}")
    sd = {}
    for fname in files:
        print(f"  [LLM] {fname}")
        sd.update(load_file(os.path.join(llm_path, fname)))
    return {k: v.to(dtype) for k, v in sd.items()}


# ---------------------------------------------------------------------------
def load_rice_vit_weights(rice_path: str, dtype: torch.dtype) -> dict:
    """
    Load pretrained MLCD / RICE-ViT weights and remap key names to the
    InnovatorVL HF format (visual.*).

    Key mapping (MLCD → InnovatorVL):
      vision_model.embeddings.patch_embedding.weight  → visual.patch_embed.proj.weight
      vision_model.embeddings.class_embedding         → visual.class_embedding
      vision_model.class_pos_emb                      → visual.class_pos_emb
      vision_model.pre_layrnorm.{weight,bias}         → visual.pre_layernorm.{weight,bias}
      vision_model.encoder.layers.i.layer_norm1.*     → visual.blocks.i.norm1.*
      vision_model.encoder.layers.i.layer_norm2.*     → visual.blocks.i.norm2.*
      vision_model.encoder.layers.i.self_attn.{q,k,v}_proj.{weight,bias}
                                                      → visual.blocks.i.attn.qkv.{weight,bias}  (cat)
      vision_model.encoder.layers.i.self_attn.out_proj.* → visual.blocks.i.attn.proj.*
      vision_model.encoder.layers.i.mlp.fc{1,2}.*    → visual.blocks.i.mlp.fc{1,2}.*
      vision_model.post_layernorm.*                   → (dropped – not used by InnovatorVL)

    PatchMerger (visual.merger.*) is always randomly initialised.
    """
    print(f"  [RICE-ViT] loading from {rice_path}")
    fpath = os.path.join(rice_path, "model.safetensors")
    if not os.path.exists(fpath):
        # try sharded
        files = sorted(f for f in os.listdir(rice_path) if f.endswith(".safetensors"))
        src = {}
        for f in files:
            with safe_open(os.path.join(rice_path, f), framework="pt") as sf:
                for k in sf.keys():
                    src[k] = sf.get_tensor(k).to(dtype)
    else:
        src = {}
        with safe_open(fpath, framework="pt") as sf:
            for k in sf.keys():
                src[k] = sf.get_tensor(k).to(dtype)

    sd = {}

    # ---- simple 1-to-1 renames ----
    simple_map = {
        "vision_model.embeddings.patch_embedding.weight": "visual.patch_embed.proj.weight",
        "vision_model.embeddings.class_embedding":        "visual.class_embedding",
        "vision_model.class_pos_emb":                     "visual.class_pos_emb",
        # note: MLCD has a typo "pre_layrnorm" (missing 'e')
        "vision_model.pre_layrnorm.weight":               "visual.pre_layernorm.weight",
        "vision_model.pre_layrnorm.bias":                 "visual.pre_layernorm.bias",
    }
    for src_k, dst_k in simple_map.items():
        if src_k in src:
            sd[dst_k] = src[src_k]
        else:
            print(f"  WARNING: expected key not found: {src_k}")

    # ---- per-layer remapping ----
    for i in range(RICE_DEPTH):
        lp  = f"vision_model.encoder.layers.{i}"
        bp  = f"visual.blocks.{i}"

        # norm1 / norm2
        sd[f"{bp}.norm1.weight"] = src[f"{lp}.layer_norm1.weight"]
        sd[f"{bp}.norm1.bias"]   = src[f"{lp}.layer_norm1.bias"]
        sd[f"{bp}.norm2.weight"] = src[f"{lp}.layer_norm2.weight"]
        sd[f"{bp}.norm2.bias"]   = src[f"{lp}.layer_norm2.bias"]

        # fuse q / k / v → qkv
        q_w = src[f"{lp}.self_attn.q_proj.weight"]
        k_w = src[f"{lp}.self_attn.k_proj.weight"]
        v_w = src[f"{lp}.self_attn.v_proj.weight"]
        sd[f"{bp}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = src[f"{lp}.self_attn.q_proj.bias"]
        k_b = src[f"{lp}.self_attn.k_proj.bias"]
        v_b = src[f"{lp}.self_attn.v_proj.bias"]
        sd[f"{bp}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        # out proj
        sd[f"{bp}.attn.proj.weight"] = src[f"{lp}.self_attn.out_proj.weight"]
        sd[f"{bp}.attn.proj.bias"]   = src[f"{lp}.self_attn.out_proj.bias"]

        # mlp
        sd[f"{bp}.mlp.fc1.weight"] = src[f"{lp}.mlp.fc1.weight"]
        sd[f"{bp}.mlp.fc1.bias"]   = src[f"{lp}.mlp.fc1.bias"]
        sd[f"{bp}.mlp.fc2.weight"] = src[f"{lp}.mlp.fc2.weight"]
        sd[f"{bp}.mlp.fc2.bias"]   = src[f"{lp}.mlp.fc2.bias"]

    print(f"  [RICE-ViT] remapped {len(sd)} tensors  "
          f"(post_layernorm dropped, merger will be random)")
    return sd


# ---------------------------------------------------------------------------
def build_visual_weights(dtype: torch.dtype) -> dict:
    """
    Randomly initialise RICE-ViT + PatchMerger.
    Shapes verified against InnovatorLab/Innovator-VL-8B-Instruct.
    """
    D   = RICE_DIM      # 1024
    FFN = RICE_FFN      # 4096
    P   = RICE_PATCH_SIZE   # 14
    C   = RICE_IN_CHANNELS  # 3
    MH  = MERGER_HIDDEN     # 4096
    TH  = TEXT_HIDDEN_SIZE  # 4096

    scale_D   = D   ** -0.5
    scale_FFN = FFN ** -0.5
    scale_MH  = MH  ** -0.5

    def rn(*shape, scale=1.0):
        return (torch.randn(*shape) * scale).to(dtype)

    def o(*shape):  return torch.ones(*shape,  dtype=dtype)
    def z(*shape):  return torch.zeros(*shape, dtype=dtype)

    sd = {}

    # patch_embed.proj.weight : (1024, 3, 14, 14)
    sd["visual.patch_embed.proj.weight"] = rn(D, C, P, P, scale=scale_D)

    # class_embedding : (1024,)
    sd["visual.class_embedding"] = rn(D, scale=scale_D)

    # class_pos_emb : (1, 32)
    sd["visual.class_pos_emb"] = rn(1, CLASS_POS_EMB_DIM)

    # pre_layernorm : (1024,)
    sd["visual.pre_layernorm.weight"] = o(D)
    sd["visual.pre_layernorm.bias"]   = z(D)

    # 24 transformer blocks
    for i in range(RICE_DEPTH):
        pfx = f"visual.blocks.{i}"

        # norm1 / norm2 : (1024,)
        sd[f"{pfx}.norm1.weight"] = o(D)
        sd[f"{pfx}.norm1.bias"]   = z(D)
        sd[f"{pfx}.norm2.weight"] = o(D)
        sd[f"{pfx}.norm2.bias"]   = z(D)

        # attn.qkv  : (3072, 1024)  /  bias (3072,)
        sd[f"{pfx}.attn.qkv.weight"] = rn(3 * D, D, scale=scale_D)
        sd[f"{pfx}.attn.qkv.bias"]   = z(3 * D)

        # attn.proj : (1024, 1024)  /  bias (1024,)
        sd[f"{pfx}.attn.proj.weight"] = rn(D, D, scale=scale_D)
        sd[f"{pfx}.attn.proj.bias"]   = z(D)

        # mlp.fc1 : (4096, 1024)  /  bias (4096,)
        sd[f"{pfx}.mlp.fc1.weight"] = rn(FFN, D, scale=scale_D)
        sd[f"{pfx}.mlp.fc1.bias"]   = z(FFN)

        # mlp.fc2 : (1024, 4096)  /  bias (1024,)
        sd[f"{pfx}.mlp.fc2.weight"] = rn(D, FFN, scale=scale_FFN)
        sd[f"{pfx}.mlp.fc2.bias"]   = z(D)

    # PatchMerger (adapter)
    # merger.ln_q : (1024,)
    sd["visual.merger.ln_q.weight"] = o(D)
    sd["visual.merger.ln_q.bias"]   = z(D)

    # merger.mlp.0 : (4096, 4096)  /  bias (4096,)
    sd["visual.merger.mlp.0.weight"] = rn(MH, MH, scale=scale_MH)
    sd["visual.merger.mlp.0.bias"]   = z(MH)

    # merger.mlp.2 : (4096, 4096)  /  bias (4096,)
    sd["visual.merger.mlp.2.weight"] = rn(TH, MH, scale=scale_MH)
    sd["visual.merger.mlp.2.bias"]   = z(TH)

    print(f"  [ViT] {len(sd)} tensors  "
          f"(depth={RICE_DEPTH}, dim={D}, ffn={FFN}, merger_hidden={MH}→{TH})")
    return sd


# ---------------------------------------------------------------------------
def build_config(llm_path: str) -> dict:
    with open(os.path.join(llm_path, "config.json")) as f:
        qwen_cfg = json.load(f)

    text_config = {
        "model_type":               "innovator_vl_text",
        "vocab_size":               qwen_cfg.get("vocab_size", 151936),
        "hidden_size":              qwen_cfg.get("hidden_size", 4096),
        "intermediate_size":        qwen_cfg.get("intermediate_size", 12288),
        "num_hidden_layers":        qwen_cfg.get("num_hidden_layers", 36),
        "num_attention_heads":      qwen_cfg.get("num_attention_heads", 32),
        "num_key_value_heads":      qwen_cfg.get("num_key_value_heads", 8),
        "head_dim":                 qwen_cfg.get("head_dim", 128),
        "hidden_act":               qwen_cfg.get("hidden_act", "silu"),
        "max_position_embeddings":  qwen_cfg.get("max_position_embeddings", 32768),
        "rms_norm_eps":             qwen_cfg.get("rms_norm_eps", 1e-6),
        "rope_theta":               qwen_cfg.get("rope_theta", 1000000.0),
        "rope_scaling":             qwen_cfg.get("rope_scaling", None),
        "use_cache":                True,
        "tie_word_embeddings":      False,
        "attention_bias":           qwen_cfg.get("attention_bias", False),
        "attention_dropout":        qwen_cfg.get("attention_dropout", 0.0),
        "use_sliding_window":       qwen_cfg.get("use_sliding_window", False),
        "sliding_window":           qwen_cfg.get("sliding_window", None),
        "max_window_layers":        qwen_cfg.get("max_window_layers", 36),
        "image_token_id":           None,
        "video_token_id":           None,
        "layer_types":              ["full_attention"] * qwen_cfg.get("num_hidden_layers", 36),
    }

    vision_config = {
        "model_type":         "rice_vit",
        "depth":              RICE_DEPTH,
        "embed_dim":          RICE_DIM,
        "hidden_size":        RICE_DIM,
        "hidden_act":         "gelu",
        "intermediate_size":  RICE_FFN,
        "num_heads":          RICE_NUM_HEADS,
        "in_channels":        RICE_IN_CHANNELS,
        "patch_size":         RICE_PATCH_SIZE,
        "spatial_merge_size": RICE_SPATIAL_MERGE,
        "temporal_patch_size":RICE_TEMPORAL_PATCH,
        "text_hidden_size":   TEXT_HIDDEN_SIZE,
        "layer_norm_eps":     1e-5,
        "initializer_range":  0.02,
    }

    return {
        "architectures":        ["InnovatorVLForConditionalGeneration"],
        "model_type":           "innovator_vl",
        "auto_map": {
            "AutoConfig":          "configuration_innovator_vl.InnovatorVLConfig",
            "AutoModel":           "modeling_innovator_vl.InnovatorVLForConditionalGeneration",
            "AutoModelForCausalLM":"modeling_innovator_vl.InnovatorVLForConditionalGeneration",
        },
        "image_token_id":       151655,
        "video_token_id":       151656,
        "vocab_size":           qwen_cfg.get("vocab_size", 151936),
        "text_config":          text_config,
        "vision_config":        vision_config,
        "torch_dtype":          "bfloat16",
        "transformers_version": "4.53.0",
    }


# ---------------------------------------------------------------------------
def save_sharded(state_dict: dict, save_path: str, max_shard_gb: float = 4.0):
    max_bytes = int(max_shard_gb * 1024 ** 3)
    shards, cur, cur_size = [], {}, 0
    for k, t in state_dict.items():
        nb = t.numel() * t.element_size()
        if cur and cur_size + nb > max_bytes:
            shards.append(cur); cur = {}; cur_size = 0
        cur[k] = t; cur_size += nb
    if cur:
        shards.append(cur)

    total = len(shards)
    weight_map = {}
    for idx, shard in enumerate(shards, 1):
        fname = f"model-{idx:05d}-of-{total:05d}.safetensors"
        print(f"  [save] {fname}  ({len(shard)} tensors)")
        save_file(shard, os.path.join(save_path, fname))
        for k in shard:
            weight_map[k] = fname

    total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    with open(os.path.join(save_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)


TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json",
    "generation_config.json", "preprocessor_config.json", "chat_template.jinja",
]

def copy_aux_files(llm_path: str, save_path: str):
    # Copy tokenizer from Qwen3-8B
    for fname in TOKENIZER_FILES:
        src = os.path.join(llm_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, save_path)
            print(f"  [tok] {fname}")

    # Copy modeling source files from ds/innovator_vl (needed for trust_remote_code)
    src_dir = os.path.join(PROJECT_ROOT, "ds", "innovator_vl")
    for fname in ["modeling_innovator_vl.py", "configuration_innovator_vl.py"]:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, save_path)
            print(f"  [src] {fname}")

    # Also copy preprocessor / chat_template from official model if available
    official = os.path.join(os.path.dirname(llm_path), "Innovator-VL-8B-Instruct")
    if os.path.isdir(official):
        for fname in ["preprocessor_config.json", "chat_template.jinja"]:
            src = os.path.join(official, fname)
            if os.path.exists(src):
                shutil.copy2(src, save_path)
                print(f"  [official] {fname}")


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = DTYPE_MAP[args.dtype]

    print(f"\n=== Building Innovator-VL stage0 (from scratch) ===")
    print(f"  LLM source : {args.llm_path}")
    print(f"  Output     : {args.save_path}")
    print(f"  dtype      : {args.dtype}")
    print(f"  seed       : {args.seed}\n")

    os.makedirs(args.save_path, exist_ok=True)

    print("[1/4] Loading Qwen3-8B weights …")
    llm_sd = load_llm(args.llm_path, dtype)
    print(f"       {len(llm_sd)} tensors\n")

    if args.rice_vit_path:
        print("[2/4] Loading pretrained RICE-ViT weights + random PatchMerger …")
        vis_sd = load_rice_vit_weights(args.rice_vit_path, dtype)
        # Add randomly-initialised PatchMerger on top
        merger = build_visual_weights(dtype)
        merger_keys = {k: v for k, v in merger.items() if "merger" in k}
        vis_sd.update(merger_keys)
        print(f"       ViT keys: {len(vis_sd) - len(merger_keys)}  "
              f"merger keys (random): {len(merger_keys)}")
    else:
        print("[2/4] Randomly initialising RICE-ViT + PatchMerger …")
        vis_sd = build_visual_weights(dtype)
    print()

    print("[3/4] Merging …")
    merged = {**llm_sd, **vis_sd}
    print(f"       Total: {len(merged)} tensors  "
          f"(LLM={len(llm_sd)}, visual={len(vis_sd)})\n")

    print("[4/4] Saving …")
    config = build_config(args.llm_path)
    with open(os.path.join(args.save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("  config.json written")

    save_sharded(merged, args.save_path)
    copy_aux_files(args.llm_path, args.save_path)

    print(f"\n=== Done! ===")
    print(f"stage0 → {args.save_path}")
    print(f"\nExpected tensor counts:")
    print(f"  LLM (Qwen3-8B)  : 399")
    print(f"  ViT + adapter   : 299")
    print(f"  Total           : 698")
    print(f"\nNext:")
    print(f"  bash examples/innovator_vl/quick_start/train_auto_stage_1_alignment_innovator_vl_8b.sh")


if __name__ == "__main__":
    main()
