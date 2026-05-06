Step0
/jizhicfs/fenghengyi/Innovator-VL/tools/build_stage0.py
cd /jizhicfs/fenghengyi/Innovator-VL
python tools/build_stage0.py \
    --llm_path      /jizhicfs/fenghengyi/models/Qwen3-8B \
    --rice_vit_path /jizhicfs/fenghengyi/rice-vit-large-patch14-560 \
    --save_path     /jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0


环境配置与代码修改总结
一、训练脚本路径修复
文件：examples/innovator_vl/quick_start/train_auto_stage_1_alignment_innovator_vl_8b.sh
暂时无法在飞书文档外展示此内容

---
二、Stage0 模型构建脚本（新增）
文件：tools/build_stage0.py（全新编写）
- 对照官方 InnovatorLab/Innovator-VL-8B-Instruct 模型（698 个 tensor，shape 完全验证）
- 支持两种模式：
  - 纯随机初始化 ViT（--rice_vit_path 不传）
  - 加载预训练 RICE-ViT（--rice_vit_path /jizhicfs/fenghengyi/rice-vit-large-patch14-560）
- 自动完成 MLCD → InnovatorVL 的 key 名映射（含 q/k/v 合并为 fused qkv）
- 随机初始化 PatchMerger adapter（visual.merger.*）
- 从 Qwen3-8B 复制 tokenizer 文件
用法：
bash
Copy
Insert
New File
Save
Insert into terminal
Apply
cd /jizhicfs/fenghengyi/Innovator-VL
python tools/build_stage0.py \
    --llm_path      /jizhicfs/fenghengyi/models/Qwen3-8B \
    --rice_vit_path /jizhicfs/fenghengyi/rice-vit-large-patch14-560 \
    --save_path     /jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0

---
三、依赖安装
Transformer Engine（最复杂）
- 系统 torch-base 环境里的 TE 2.7 .so 与 innov 的 torch 2.11 ABI 不兼容
- PyPI 预编译 wheel 同样不兼容 CUDA 13.0 / torch 2.11
- 最终方案：从源码编译安装
git clone --branch v2.7 https://github.com/NVIDIA/TransformerEngine.git
pip install pybind11 ninja
NVTE_FRAMEWORK=pytorch CUDA_HOME=/usr/local/cuda pip install . --no-build-isolation

---
四、代码 Bug 修复（3 处）
1. aiak_megatron/megatron/core/transformer/moe/fused_a2a.py
# 原来（裸 import，直接崩溃）from transformer_engine.pytorch.constants import TE_DType

# 修复后（TE_DType 仅 FP8 路径用，优雅降级）try:
    from transformer_engine.pytorch.constants import TE_DType
except ImportError:
    TE_DType = None
2. aiak_training_llm/models/custom/common/local_norm.py
# 原来（Apex 不存在时 class 定义本身就 NameError）class FusedRMSNorm(ApexFusedRMSNorm): ...

# 修复后（条件式类定义）if HAVE_FUSED_RMS_NORM:
    class FusedRMSNorm(ApexFusedRMSNorm): ...
else:
    class FusedRMSNorm(nn.Module):
        def __init__(self, ...): raise ValueError('Apex required')
3. aiak_megatron/megatron/core/extensions/transformer_engine.py（间接修复）
- 通过正确编译安装 TE 解决，无需改代码



Step1

# Download LLaVA-558K dataset
# Set environment variables and run

AIAK_TRAINING_PATH=/jizhicfs/fenghengyi/Innovator-VL \
DATA_PATH=/jizhicfs/fenghengyi/data/llava_pretrain \
TOKENIZER_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0 \
CHECKPOINT_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0 \
bash examples/innovator_vl/quick_start/train_auto_stage_1_alignment_innovator_vl_8b.sh


cd /jizhicfs/fenghengyi/Innovator-VL

AIAK_TRAINING_PATH=/jizhicfs/fenghengyi/Innovator-VL \
DATA_PATH=/jizhicfs/fenghengyi/data/llava_pretrain_energon \
TOKENIZER_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0 \
CHECKPOINT_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0_mcore_tp8_pp1 \
bash examples/innovator_vl/quick_start/train_auto_stage_1_alignment_innovator_vl_8b.sh




1. adapter.layernorm._extra_state 缺失（Missing key）
原因：adapter.py 转换脚本在 HF→mcore 转换时只补了 linear_fc1 和 linear_fc2 的 _extra_state，漏掉了 layernorm。
修复：
- tools/convert_checkpoint/custom/innovator_vl/adapter.py：补上 adapter.layernorm._extra_state
- 直接给现有 checkpoint（8 个 rank）用 Python 脚本补写该 key


---
2. checkpoint 里 _extra_state 格式错误（BytesIO vs Tensor）
原因：TE 有两种 set_extra_state 实现：
- module/base.py：接受 BytesIO 或 Tensor
- ops/op.py：只接受 Tensor，BytesIO 会触发 AttributeError: has no attribute 'numel'
旧的 convert 脚本存的是 BytesIO(torch.save(None))，在 ops/op.py 类型的模块上崩溃。
修复：将所有 checkpoint 里的 _extra_state 统一改为空 Tensor（torch.tensor([], dtype=torch.uint8)），两种 TE 实现都会走 numel()==0 → return。

---
3. checkpointing.py 兼容层（防御性修复）
原因：旧格式 checkpoint 里 _extra_state key 带有 module.module. wrapper 前缀，strict=True 加载时报 unexpected key。
修复：在 aiak_megatron/megatron/training/checkpointing.py 加入 _reconcile_extra_states：
- 自动去掉 _extra_state key 的 wrapper 前缀
- 将 BytesIO 格式的值转为空 Tensor
- 改用 strict=False 加载 + _check_weights_loaded 手动验证权重完整性

---
4. 数据集格式问题（.nv-meta/.info.yaml 缺失）
原因：DATA_PATH 指向的是原始 LLaVA-558K 图片目录，训练框架需要 megatron/energon WebDataset 格式。
修复：
- 新建 tools/data_preprocess/convert_llava_pretrain_to_energon.py，将 LLaVA-558K（blip_laion_cc_sbu_558k.json + 图片）转成 energon tar 格式，同时保留 <image> 占位符
- 手动生成 .nv-meta/.info.yaml（558128 个样本，112 个 tar shard）
- 输出目录：/jizhicfs/fenghengyi/data/llava_pretrain_energon
source /jizhicfs/fenghengyi/miniconda3/etc/profile.d/conda.sh && conda activate innov

python /jizhicfs/fenghengyi/Innovator-VL/tools/data_preprocess/convert_llava_pretrain_to_energon.py \
    --json_file  /jizhicfs/fenghengyi/data/llava_pretrain/blip_laion_cc_sbu_558k.json \
    --image_dir  /jizhicfs/fenghengyi/data/llava_pretrain \
    --output_dir /jizhicfs/fenghengyi/data/llava_pretrain_energon \
    --maxcount   5000 \
    --workers    32


---
5. cu_lengths 属性缺失
原因：训练脚本硬编码 OFFLINE_PACKED_DATA=1，要求数据是预打包格式（每个 sample 包含多条拼接的对话）。我们的数据是原始格式（每个 sample 一条对话）。
修复：在训练脚本中将 OFFLINE_PACKED_DATA 和 OFFLINE_PACKING_VQA 改为 0。


[图片]
完整运行流程
build_stage0.py          （已有脚本）
    ↓ 生成 HF 格式 stage0 checkpoint
convert_8b_hf_to_mcore.sh  （已有脚本，调用了修复过的 adapter.py）
    ↓ 生成 Megatron mcore 格式 checkpoint
convert_llava_pretrain_to_energon.py  （新建脚本）
    ↓ 生成 energon WebDataset 格式数据
train_auto_stage_1_alignment_innovator_vl_8b.sh  （修改了3处参数）
    ↓ 开始训练

---
为什么需要这两层转换？
模型格式转换（HF → Megatron mcore）
HuggingFace 和 Megatron 的模型权重组织方式完全不同：
HuggingFace 格式是按模块命名的单一 state dict：
model.layers.0.self_attn.q_proj.weight    # 完整的 Q 矩阵
model.layers.0.self_attn.k_proj.weight    # 完整的 K 矩阵
model.layers.0.self_attn.v_proj.weight    # 完整的 V 矩阵
Megatron mcore 格式需要支持张量并行（TP），所以：
# TP=8，Q/K/V 合并且沿 head 维切分到 8 个文件
mp_rank_00/model_optim_rng.pt:
    language_model.decoder.layers.0.self_attention.linear_qkv.weight  # 1/8
mp_rank_01/model_optim_rng.pt:
    language_model.decoder.layers.0.self_attention.linear_qkv.weight  # 1/8
...
转换的核心操作（tools/convert_checkpoint/model.py + mcore_checkpoint.py）：
1. Q/K/V 权重合并为 linear_qkv（适配 TELayerNormColumnParallelLinear）
2. 沿 head 维度切分成 TP 份，分存到 mp_rank_00~07
3. LayerNorm 的 weight/bias 从独立 key 合并进 linear_qkv.layer_norm_weight/bias
4. 补充 TE 特有的 _extra_state key
数据集格式转换（原始 → energon WebDataset）
训练框架使用 megatron/energon 做数据加载，它需要 WebDataset 格式：
LLaVA-558K 原始格式：
blip_laion_cc_sbu_558k.json   # 558K 条 {"id", "image", "conversations"}
00453/004539375.jpg            # 图片散落在子目录里
energon 需要的格式：
pretrain-000000.tar            # 图文打包在 tar 里，每条样本 = json + jpg
pretrain-000001.tar
...
.nv-meta/
    .info.yaml     # 记录每个 tar 的样本数，energon 靠它做分片索引
    split.yaml     # train/val/test 分割
    dataset.yaml   # 声明 sample_type 和 sample_loader
    sample_loader.py  # 定义如何从 tar 里解析一条样本
这样设计的原因：
- tar 格式适合顺序读，避免随机 IO，分布式训练时各 rank 只需要读自己的 shard
- sample_loader.py 将 tar 里的原始字节流解析成训练框架的 MultiMixQASample 对象
- .info.yaml 让 energon 在不读取全部数据的情况下知道总样本数，用于计算 epoch 边界和 checkpoint restore


如何运行？
1.手动缝合ckpts
source /jizhicfs/fenghengyi/miniconda3/etc/profile.d/conda.sh && conda activate innov

cd /jizhicfs/fenghengyi/Innovator-VL
python tools/build_stage0.py \
    --llm_path  /jizhicfs/fenghengyi/models/Qwen3-8B \
    --save_path /jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0 \
    --rice_vit_path /jizhicfs/fenghengyi/rice-vit-large-patch14-560
2.hf->megatron mcore
cd /jizhicfs/fenghengyi/Innovator-VL

AIAK_TRAINING_PATH=/jizhicfs/fenghengyi/Innovator-VL \
AIAK_MAGATRON_PATH=/jizhicfs/fenghengyi/Innovator-VL/aiak_megatron \
bash examples/innovator_vl/convert/convert_8b_hf_to_mcore.sh \
    /jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0 \
    /jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0_mcore_tp8_pp1 \
    8 \
    1
3.修复 checkpoint 的 _extra_state 格式（这部分比较乱）
source /jizhicfs/fenghengyi/miniconda3/etc/profile.d/conda.sh && conda activate innov

python3 -c "
import torch, glob

EMPTY = torch.tensor([], dtype=torch.uint8)
CKPT_DIR = '/jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0_mcore_tp8_pp1/release'

for fpath in sorted(glob.glob(f'{CKPT_DIR}/mp_rank_*/model_optim_rng.pt')):
    sd = torch.load(fpath, map_location='cpu', weights_only=False)
    model = sd['model']
    patched = 0
    for k in list(model.keys()):
        if k.endswith('._extra_state') or k == '_extra_state':
            if not isinstance(model[k], torch.Tensor) or model[k].numel() != 0:
                model[k] = EMPTY
                patched += 1
    torch.save(sd, fpath)
    print(f'{fpath}: patched {patched} keys')
print('Done.')
"
4.准备 energon 数据集（也有点乱）
# 转换原始 LLaVA-558K 数据
python /jizhicfs/fenghengyi/Innovator-VL/tools/data_preprocess/convert_llava_pretrain_to_energon.py \
    --json_file  /jizhicfs/fenghengyi/data/llava_pretrain/blip_laion_cc_sbu_558k.json \
    --image_dir  /jizhicfs/fenghengyi/data/llava_pretrain \
    --output_dir /jizhicfs/fenghengyi/data/llava_pretrain_energon \
    --maxcount   5000 \
    --workers    32

# 手动补生成 .info.yaml（如果上一步没自动生成）
python3 -c "
import sys, yaml, glob, os
sys.path.insert(0, '/jizhicfs/fenghengyi/Innovator-VL/aiak_megatron')
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.prepare import WebdatasetPreparator

output_dir = '/jizhicfs/fenghengyi/data/llava_pretrain_energon'
path = EPath(output_dir).absolute()
all_tars = sorted(os.path.relpath(p, output_dir) for p in glob.glob(f'{output_dir}/*.tar'))

shard_counts = {}
for tar in all_tars:
    info, _ = WebdatasetPreparator._preprocess_tar(tar, parent_path=str(path), max_parts=50)
    if info.count > 0:
        shard_counts[info.name] = info.count

with open(os.path.join(output_dir, '.nv-meta', '.info.yaml'), 'w') as f:
    yaml.dump({'shard_counts': shard_counts}, f)
print(f'Done: {len(shard_counts)} shards, {sum(shard_counts.values())} samples')
"
5.训练启动
cd /jizhicfs/fenghengyi/Innovator-VL

AIAK_TRAINING_PATH=/jizhicfs/fenghengyi/Innovator-VL \
DATA_PATH=/jizhicfs/fenghengyi/data/llava_pretrain_energon \
TOKENIZER_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0 \
CHECKPOINT_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0_mcore_tp8_pp1 \
bash examples/innovator_vl/quick_start/train_auto_stage_1_alignment_innovator_vl_8b.sh
Step 1.5
新增了一个数据转换脚本 convert_mid_training_to_energon.py
python /jizhicfs/fenghengyi/Innovator-VL/tools/data_preprocess/convert_mid_training_to_energon.py \
    --parquet_dir /jizhicfs/fenghengyi/data/LLaVA-OneVision-1.5-Mid-Training-85M \
    --output_dir  /jizhicfs/fenghengyi/data/llava_mid_training_energon \
    --maxcount    5000 \
    --workers     32

之后起训练
AIAK_TRAINING_PATH=/jizhicfs/fenghengyi/Innovator-VL \
DATA_PATH=/jizhicfs/fenghengyi/data/llava_mid_training_energon \
TOKENIZER_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0 \
CHECKPOINT_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/stage_1_alignment_innovator_vl_8b \
SAVE_CKPT_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/stage_1.5_mid_training_innovator_vl_8b \
bash examples/innovator_vl/quick_start/train_auto_stage_1.5_mid_training_innovator_vl_8b.sh \
    8 1


Supported flash-attn versions are >= 2.1.1, <= 2.8.1. Found flash-attn 2.8.4.
Supported flash-attn versions are >= 2.1.1, <= 2.8.1. Found flash-attn 2.8.4.
Supported flash-attn versions are >= 2.1.1, <= 2.8.1. Found flash-attn 2.8.4.
Supported flash-attn versions are >= 2.1.1, <= 2.8.1. Found flash-attn 2.8.4.
Supported flash-attn versions are >= 2.1.1, <= 2.8.1. Found flash-attn 2.8.4.
Supported flash-attn versions are >= 2.1.1, <= 2.8.1. Found flash-attn 2.8.4.
Supported flash-attn versions are >= 2.1.1, <= 2.8.1. Found flash-attn 2.8.4.
重新编译了一次flash attention=2.8.1，如果用fused mode我看会出现数值稳定性问题
cd /jizhicfs/fenghengyi/Innovator-VL

AIAK_TRAINING_PATH=/jizhicfs/fenghengyi/Innovator-VL \
DATA_PATH=/jizhicfs/fenghengyi/data/llava_mid_training_energon \
TOKENIZER_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0 \
CHECKPOINT_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/stage_1_alignment_innovator_vl_8b \
SAVE_CKPT_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/stage_1.5_mid_training_innovator_vl_8b \
bash examples/innovator_vl/quick_start/train_auto_stage_1.5_mid_training_innovator_vl_8b.sh \
    8 1 32768 1 96 100 100


Step 2
加了一个转换脚本 convert_sft_instruct_to_energon.py
python /jizhicfs/fenghengyi/Innovator-VL/tools/data_preprocess/convert_sft_instruct_to_energon.py \
    --parquet_dir /jizhicfs/fenghengyi/data/Innovator-VL-Instruct-46M \
    --output_dir  /jizhicfs/fenghengyi/data/innovator_instruct_energon \
    --maxcount    5000
训练
cd /jizhicfs/fenghengyi/Innovator-VL

AIAK_TRAINING_PATH=/jizhicfs/fenghengyi/Innovator-VL \
DATA_PATH=/jizhicfs/fenghengyi/data/innovator_instruct_energon \
TOKENIZER_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-8B-stage0 \
CHECKPOINT_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/stage_1.5_mid_training_innovator_vl_8b \
SAVE_CKPT_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/stage_2_instruct_innovator_vl_8b \
bash examples/innovator_vl/quick_start/train_auto_stage_2_instruct_innovator_vl_8b.sh \
    8 1 32768 1 288 100 100
