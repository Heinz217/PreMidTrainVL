#!/usr/bin/env python3
# coding=utf-8
"""
Build Innovator-VL stage0 model from scratch — Qwen3-MoE (30B-A3B) variant.

This is the MoE counterpart of build_stage0.py (which targets Qwen3-8B dense).

Architecture:
  LLM    : Qwen3-30B-A3B-Base  (qwen3_moe, hidden=2048, 48 layers, 128 experts top-8)
  ViT    : RICE-ViT 24L 1024d  (same as 8B variant; loaded from MLCD checkpoint or random)
  Adapter: PatchMerger  (LayerNorm + 4096->4096->2048 MLP)

Usage
-----
  cd /jizhicfs/fenghengyi/ai4sci/PreMidTrainVL
  python tools/build_stage0_qwen3moe.py \
      --llm_path      /jizhicfs/fenghengyi/models/Qwen3-30B-A3B-Base \
      --rice_vit_path /jizhicfs/fenghengyi/models/rice-vit-large-patch14-560 \
      --save_path     /jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-30B-A3B-stage0
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

# ---- Vision constants (verified, identical to 8B variant) -----------------
RICE_DEPTH          = 24
RICE_DIM            = 1024     # embed_dim / hidden_size
RICE_NUM_HEADS      = 16
RICE_FFN            = 4096     # intermediate_size = DIM * 4
RICE_PATCH_SIZE     = 14
RICE_IN_CHANNELS    = 3
RICE_TEMPORAL_PATCH = 1
RICE_SPATIAL_MERGE  = 2
# PatchMerger input dim  = DIM * SPATIAL_MERGE^2  = 1024 * 4 = 4096
MERGER_HIDDEN       = RICE_DIM * RICE_SPATIAL_MERGE * RICE_SPATIAL_MERGE  # 4096
# class_pos_emb shape: (1, head_dim // 2)
CLASS_POS_EMB_DIM   = (RICE_DIM // RICE_NUM_HEADS) // 2   # 32


def parse_args():
    p = argparse.ArgumentParser(description="Build Innovator-VL stage0 from Qwen3-30B-A3B-Base (MoE)")
    p.add_argument("--llm_path",      required=True, help="Qwen3-30B-A3B-Base HF checkpoint dir")
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
    """Load all sharded safetensors of the LLM as a single state_dict."""
    files = sorted(f for f in os.listdir(llm_path) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {llm_path}")
    sd = {}
    for fname in files:
        print(f"  [LLM] {fname}")
        sd.update(load_file(os.path.join(llm_path, fname)))
    # Cast all tensors to target dtype
    return {k: v.to(dtype) for k, v in sd.items()}


# ---------------------------------------------------------------------------
def load_rice_vit_weights(rice_path: str, dtype: torch.dtype) -> dict:
    """
    Load pretrained MLCD / RICE-ViT weights and remap key names to the
    InnovatorVL HF format (visual.*).

    Key mapping (MLCD -> InnovatorVL):
      vision_model.embeddings.patch_embedding.weight   -> visual.patch_embed.proj.weight
      vision_model.embeddings.class_embedding          -> visual.class_embedding
      vision_model.class_pos_emb                       -> visual.class_pos_emb
      vision_model.pre_layrnorm.{w,b}                  -> visual.pre_layernorm.{w,b}
      vision_model.encoder.layers.i.layer_norm{1,2}.*  -> visual.blocks.i.norm{1,2}.*
      vision_model.encoder.layers.i.self_attn.{q,k,v}_proj.{w,b}
                                                       -> visual.blocks.i.attn.qkv.{w,b} (concat)
      vision_model.encoder.layers.i.self_attn.out_proj.* -> visual.blocks.i.attn.proj.*
      vision_model.encoder.layers.i.mlp.fc{1,2}.*      -> visual.blocks.i.mlp.fc{1,2}.*
      vision_model.post_layernorm.*                    -> dropped (not used by InnovatorVL)

    PatchMerger (visual.merger.*) is always randomly initialised.
    """
    print(f"  [RICE-ViT] loading from {rice_path}")
    fpath = os.path.join(rice_path, "model.safetensors")
    src = {}
    if not os.path.exists(fpath):
        files = sorted(f for f in os.listdir(rice_path) if f.endswith(".safetensors"))
        for f in files:
            with safe_open(os.path.join(rice_path, f), framework="pt") as sf:
                for k in sf.keys():
                    src[k] = sf.get_tensor(k).to(dtype)
    else:
        with safe_open(fpath, framework="pt") as sf:
            for k in sf.keys():
                src[k] = sf.get_tensor(k).to(dtype)

    sd = {}

    # ---- 1-to-1 renames ----
    simple_map = {
        "vision_model.embeddings.patch_embedding.weight": "visual.patch_embed.proj.weight",
        "vision_model.embeddings.class_embedding":        "visual.class_embedding",
        "vision_model.class_pos_emb":                     "visual.class_pos_emb",
        # MLCD has a typo: "pre_layrnorm" (missing 'e')
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
        lp = f"vision_model.encoder.layers.{i}"
        bp = f"visual.blocks.{i}"

        sd[f"{bp}.norm1.weight"] = src[f"{lp}.layer_norm1.weight"]
        sd[f"{bp}.norm1.bias"]   = src[f"{lp}.layer_norm1.bias"]
        sd[f"{bp}.norm2.weight"] = src[f"{lp}.layer_norm2.weight"]
        sd[f"{bp}.norm2.bias"]   = src[f"{lp}.layer_norm2.bias"]

        # Fuse q / k / v -> qkv
        q_w = src[f"{lp}.self_attn.q_proj.weight"]
        k_w = src[f"{lp}.self_attn.k_proj.weight"]
        v_w = src[f"{lp}.self_attn.v_proj.weight"]
        sd[f"{bp}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = src[f"{lp}.self_attn.q_proj.bias"]
        k_b = src[f"{lp}.self_attn.k_proj.bias"]
        v_b = src[f"{lp}.self_attn.v_proj.bias"]
        sd[f"{bp}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        sd[f"{bp}.attn.proj.weight"] = src[f"{lp}.self_attn.out_proj.weight"]
        sd[f"{bp}.attn.proj.bias"]   = src[f"{lp}.self_attn.out_proj.bias"]

        sd[f"{bp}.mlp.fc1.weight"] = src[f"{lp}.mlp.fc1.weight"]
        sd[f"{bp}.mlp.fc1.bias"]   = src[f"{lp}.mlp.fc1.bias"]
        sd[f"{bp}.mlp.fc2.weight"] = src[f"{lp}.mlp.fc2.weight"]
        sd[f"{bp}.mlp.fc2.bias"]   = src[f"{lp}.mlp.fc2.bias"]

    print(f"  [RICE-ViT] remapped {len(sd)} tensors  "
          f"(post_layernorm dropped, merger will be random)")
    return sd


# ---------------------------------------------------------------------------
def build_visual_weights(dtype: torch.dtype, text_hidden_size: int) -> dict:
    """
    Randomly initialise RICE-ViT + PatchMerger.

    Note: merger.mlp.2 output dim = text_hidden_size (LLM hidden_size).
          For 30B-A3B this is 2048 (vs 4096 for the 8B variant).
    """
    D   = RICE_DIM           # 1024
    FFN = RICE_FFN           # 4096
    P   = RICE_PATCH_SIZE    # 14
    C   = RICE_IN_CHANNELS   # 3
    MH  = MERGER_HIDDEN      # 4096
    TH  = text_hidden_size   # 2048 for 30B-A3B

    scale_D   = D   ** -0.5
    scale_FFN = FFN ** -0.5
    scale_MH  = MH  ** -0.5

    def rn(*shape, scale=1.0):
        return (torch.randn(*shape) * scale).to(dtype)

    def o(*shape):  return torch.ones(*shape,  dtype=dtype)
    def z(*shape):  return torch.zeros(*shape, dtype=dtype)

    sd = {}

    sd["visual.patch_embed.proj.weight"] = rn(D, C, P, P, scale=scale_D)
    sd["visual.class_embedding"]         = rn(D, scale=scale_D)
    sd["visual.class_pos_emb"]           = rn(1, CLASS_POS_EMB_DIM)
    sd["visual.pre_layernorm.weight"]    = o(D)
    sd["visual.pre_layernorm.bias"]      = z(D)

    for i in range(RICE_DEPTH):
        pfx = f"visual.blocks.{i}"
        sd[f"{pfx}.norm1.weight"] = o(D)
        sd[f"{pfx}.norm1.bias"]   = z(D)
        sd[f"{pfx}.norm2.weight"] = o(D)
        sd[f"{pfx}.norm2.bias"]   = z(D)

        sd[f"{pfx}.attn.qkv.weight"]  = rn(3 * D, D, scale=scale_D)
        sd[f"{pfx}.attn.qkv.bias"]    = z(3 * D)
        sd[f"{pfx}.attn.proj.weight"] = rn(D, D, scale=scale_D)
        sd[f"{pfx}.attn.proj.bias"]   = z(D)

        sd[f"{pfx}.mlp.fc1.weight"] = rn(FFN, D, scale=scale_D)
        sd[f"{pfx}.mlp.fc1.bias"]   = z(FFN)
        sd[f"{pfx}.mlp.fc2.weight"] = rn(D, FFN, scale=scale_FFN)
        sd[f"{pfx}.mlp.fc2.bias"]   = z(D)

    # PatchMerger (adapter)  — output dim = LLM hidden size
    sd["visual.merger.ln_q.weight"]  = o(D)
    sd["visual.merger.ln_q.bias"]    = z(D)
    sd["visual.merger.mlp.0.weight"] = rn(MH, MH, scale=scale_MH)   # 4096 -> 4096
    sd["visual.merger.mlp.0.bias"]   = z(MH)
    sd["visual.merger.mlp.2.weight"] = rn(TH, MH, scale=scale_MH)   # 4096 -> TH (2048 for 30B-A3B)
    sd["visual.merger.mlp.2.bias"]   = z(TH)

    print(f"  [ViT] {len(sd)} tensors  "
          f"(depth={RICE_DEPTH}, dim={D}, ffn={FFN}, merger {MH}->{MH}->{TH})")
    return sd


# ---------------------------------------------------------------------------
def build_config(llm_path: str, text_hidden_size: int) -> dict:
    """Build the InnovatorVL HF config for the Qwen3-MoE variant."""
    with open(os.path.join(llm_path, "config.json")) as f:
        qwen_cfg = json.load(f)

    num_layers = qwen_cfg.get("num_hidden_layers", 48)

    # text_config: faithfully copies Qwen3-MoE config + InnovatorVL overrides
    text_config = {
        # core
        "model_type":               "qwen3_moe",
        "vocab_size":               qwen_cfg.get("vocab_size", 151936),
        "hidden_size":              qwen_cfg.get("hidden_size", 2048),
        "intermediate_size":        qwen_cfg.get("intermediate_size", 6144),
        "num_hidden_layers":        num_layers,
        "num_attention_heads":      qwen_cfg.get("num_attention_heads", 32),
        "num_key_value_heads":      qwen_cfg.get("num_key_value_heads", 4),
        "head_dim":                 qwen_cfg.get("head_dim", 128),
        "hidden_act":               qwen_cfg.get("hidden_act", "silu"),
        "max_position_embeddings":  qwen_cfg.get("max_position_embeddings", 32768),
        "rms_norm_eps":             qwen_cfg.get("rms_norm_eps", 1e-6),
        "rope_theta":               qwen_cfg.get("rope_theta", 1000000.0),
        "rope_scaling":             qwen_cfg.get("rope_scaling", None),
        "use_cache":                True,
        "tie_word_embeddings":      qwen_cfg.get("tie_word_embeddings", False),
        "attention_bias":           qwen_cfg.get("attention_bias", False),
        "attention_dropout":        qwen_cfg.get("attention_dropout", 0.0),
        "use_sliding_window":       qwen_cfg.get("use_sliding_window", False),
        "sliding_window":           qwen_cfg.get("sliding_window", None),
        "max_window_layers":        qwen_cfg.get("max_window_layers", num_layers),
        # MoE-specific
        "num_experts":              qwen_cfg.get("num_experts", 128),
        "num_experts_per_tok":      qwen_cfg.get("num_experts_per_tok", 8),
        "moe_intermediate_size":    qwen_cfg.get("moe_intermediate_size", 768),
        "decoder_sparse_step":      qwen_cfg.get("decoder_sparse_step", 1),
        "norm_topk_prob":           qwen_cfg.get("norm_topk_prob", True),
        "router_aux_loss_coef":     qwen_cfg.get("router_aux_loss_coef", 0.001),
        "output_router_logits":     qwen_cfg.get("output_router_logits", False),
        "mlp_only_layers":          qwen_cfg.get("mlp_only_layers", []),
        # InnovatorVL-specific
        "image_token_id":           None,
        "video_token_id":           None,
        "layer_types":              ["full_attention"] * num_layers,
    }

    vision_config = {
        "model_type":           "rice_vit",
        "depth":                RICE_DEPTH,
        "embed_dim":            RICE_DIM,
        "hidden_size":          RICE_DIM,
        "hidden_act":           "gelu",
        "intermediate_size":    RICE_FFN,
        "num_heads":            RICE_NUM_HEADS,
        "in_channels":          RICE_IN_CHANNELS,
        "patch_size":           RICE_PATCH_SIZE,
        "spatial_merge_size":   RICE_SPATIAL_MERGE,
        "temporal_patch_size":  RICE_TEMPORAL_PATCH,
        "text_hidden_size":     text_hidden_size,    # 2048 for 30B-A3B
        "layer_norm_eps":       1e-5,
        "initializer_range":    0.02,
    }

    return {
        "architectures":        ["InnovatorVlForConditionalGeneration"],
        "model_type":           "innovator_vl",
        "auto_map": {
            # IMPORTANT: must match the actual class names in
            #   ds/innovator_vl/configuration_innovator_vl_moe.py  ->  InnovatorVlConfig
            #   ds/innovator_vl/modeling_innovator_vl_moe.py       ->  InnovatorVlForConditionalGeneration
            #     (the upstream class was originally named Qwen2VLForConditionalGeneration;
            #      we rename it to InnovatorVl* in the stage0 dir so auto_map looks clean.)
            "AutoConfig":           "configuration_innovator_vl_moe.InnovatorVlConfig",
            "AutoModel":            "modeling_innovator_vl_moe.InnovatorVlForConditionalGeneration",
            "AutoModelForCausalLM": "modeling_innovator_vl_moe.InnovatorVlForConditionalGeneration",
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
def save_sharded(state_dict: dict, save_path: str, max_shard_gb: float = 4.5):
    """Shard the full state_dict into ~4.5 GB safetensors chunks."""
    max_bytes = int(max_shard_gb * 1024 ** 3)
    shards, cur, cur_size = [], {}, 0
    for k, t in state_dict.items():
        nb = t.numel() * t.element_size()
        if cur and cur_size + nb > max_bytes:
            shards.append(cur)
            cur, cur_size = {}, 0
        cur[k] = t
        cur_size += nb
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
    print(f"  [save] total: {total} shards, {total_bytes / 1024**3:.2f} GB")


TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json",
    "generation_config.json", "chat_template.jinja",
    "chat_template.json",
    # NOTE: preprocessor_config.json / processor_config.json are written below
    # in copy_aux_files() with the correct Qwen2VL-compatible schema.
]


def copy_aux_files(llm_path: str, save_path: str):
    """Copy tokenizer + modeling source files into the stage0 dir."""
    # 1) Tokenizer / generation files from Qwen3-30B-A3B-Base
    for fname in TOKENIZER_FILES:
        src = os.path.join(llm_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, save_path)
            print(f"  [tok] {fname}")

    # 2) Modeling source — IMPORTANT: use the MoE versions (configuration/modeling_innovator_vl_moe.py)
    src_dir = os.path.join(PROJECT_ROOT, "ds", "innovator_vl")
    for fname in ["modeling_innovator_vl_moe.py", "configuration_innovator_vl_moe.py"]:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, save_path)
            print(f"  [src] {fname}")
        else:
            print(f"  WARNING: {fname} not found in {src_dir}")

    # 3) Write Qwen2-VL-compatible processor configs.
    #    AutoProcessor needs these to instantiate Qwen2VLProcessor + Qwen2VLImageProcessor,
    #    so that mm_plugin can read processor.image_processor.merge_size / patch_size.
    #    Vision config (RICE-ViT): patch_size=14, spatial_merge_size=2, temporal_patch_size=1.
    preprocessor_cfg = {
        "image_processor_type": "Qwen2VLImageProcessor",
        "processor_class": "Qwen2VLProcessor",
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std":  [0.26862954, 0.26130258, 0.27577711],
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "merge_size": RICE_SPATIAL_MERGE,
        "patch_size": RICE_PATCH_SIZE,
        "temporal_patch_size": RICE_TEMPORAL_PATCH,
        "min_pixels": 200704,
        "max_pixels": 1003520,
    }
    with open(os.path.join(save_path, "preprocessor_config.json"), "w") as f:
        json.dump(preprocessor_cfg, f, indent=2)
    print("  [proc] preprocessor_config.json (Qwen2VLImageProcessor)")

    with open(os.path.join(save_path, "processor_config.json"), "w") as f:
        json.dump({"processor_class": "Qwen2VLProcessor"}, f, indent=2)
    print("  [proc] processor_config.json (Qwen2VLProcessor)")


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = DTYPE_MAP[args.dtype]

    # Read text_hidden_size from LLM config (avoids hard-coding 2048)
    with open(os.path.join(args.llm_path, "config.json")) as f:
        text_hidden_size = json.load(f)["hidden_size"]

    print(f"\n=== Building Innovator-VL stage0 (Qwen3-MoE / 30B-A3B variant) ===")
    print(f"  LLM source       : {args.llm_path}")
    print(f"  RICE-ViT source  : {args.rice_vit_path or '<random init>'}")
    print(f"  Output           : {args.save_path}")
    print(f"  dtype            : {args.dtype}")
    print(f"  text_hidden_size : {text_hidden_size}  (auto-detected from LLM config)")
    print(f"  seed             : {args.seed}\n")

    os.makedirs(args.save_path, exist_ok=True)

    print("[1/4] Loading Qwen3-30B-A3B-Base weights ...")
    llm_sd = load_llm(args.llm_path, dtype)
    print(f"       {len(llm_sd)} tensors\n")

    if args.rice_vit_path:
        print("[2/4] Loading pretrained RICE-ViT weights + random PatchMerger ...")
        vis_sd = load_rice_vit_weights(args.rice_vit_path, dtype)
        # Add randomly-initialised PatchMerger on top (output dim follows LLM hidden)
        merger = build_visual_weights(dtype, text_hidden_size)
        merger_keys = {k: v for k, v in merger.items() if "merger" in k}
        vis_sd.update(merger_keys)
        print(f"       ViT keys: {len(vis_sd) - len(merger_keys)}  "
              f"merger keys (random): {len(merger_keys)}")
    else:
        print("[2/4] Randomly initialising RICE-ViT + PatchMerger ...")
        vis_sd = build_visual_weights(dtype, text_hidden_size)
    print()

    print("[3/4] Merging ...")
    merged = {**llm_sd, **vis_sd}
    print(f"       Total: {len(merged)} tensors  "
          f"(LLM={len(llm_sd)}, visual={len(vis_sd)})\n")

    print("[4/4] Saving ...")
    config = build_config(args.llm_path, text_hidden_size)
    with open(os.path.join(args.save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("  config.json written")

    save_sharded(merged, args.save_path)
    copy_aux_files(args.llm_path, args.save_path)

    print(f"\n=== Done! ===")
    print(f"stage0 -> {args.save_path}")
    print(f"\nExpected tensor counts (30B-A3B variant):")
    print(f"  LLM (Qwen3-30B-A3B)   : ~ many   (48 layers x 128 experts x 3 + attn/norm)")
    print(f"  ViT + adapter         : 299")
    print(f"\nNext step:")
    print(f"  bash examples/innovator_vl/convert/convert_30b_a3b_hf_to_mcore.sh \\")
    print(f"      {args.save_path} \\")
    print(f"      {args.save_path}_mcore_tp4_pp1_ep8")


if __name__ == "__main__":
    main()
