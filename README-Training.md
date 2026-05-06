# 训练 Qwen3-MoE 模型

## 0. 数据准备

数据准备是按照 Innovator-VL 官方给出的数据集配的。这里只下载了子集用来验证。

### data1

```bash
mkdir -p /jizhicfs/fenghengyi/data/llava_pretrain
huggingface-cli download liuhaotian/LLaVA-Pretrain \
    --repo-type dataset \
    --local-dir /jizhicfs/fenghengyi/data/llava_pretrain
```

### data1.5

```bash
mkdir -p /jizhicfs/fenghengyi/data/LLaVA-OneVision-1.5-Mid-Training-85M
huggingface-cli download mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M \
    --repo-type dataset \
    --local-dir /jizhicfs/fenghengyi/data/LLaVA-OneVision-1.5-Mid-Training-85M \
    --include "imagenet/EN/*00000*" "imagenet/EN/*00001*"
```

### data2

```bash
mkdir -p /jizhicfs/fenghengyi/data/Innovator-VL-Instruct-46M
huggingface-cli download InnovatorLab/Innovator-VL-Instruct-46M \
    --repo-type dataset \
    --local-dir /jizhicfs/fenghengyi/data/Innovator-VL-Instruct-46M \
    --include "data1/SFT_000001.parquet" "data1/SFT_000002.parquet"
```

## 1. 环境配置

腾讯这个机器上我还是建议把很多必要的库都本地编译一下。

其实在 `/jizhicfs/fenghengyi/miniconda3/envs/innov` 这个环境里，我已经把一切配好了，可以直接用。

### 1.1 基础配置

```bash
conda create -y -n innov python=3.12
conda activate innov

pip install -U pip setuptools wheel
pip install -U packaging ninja pybind11

pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio
pip install -r /jizhicfs/fenghengyi/PreMidTrainVL/requirements.txt

# 强制升级 transformers / accelerate / hf_hub / energon
pip install -U "transformers>=5.1.0" "accelerate>=1.9" "huggingface_hub>=0.27" hf_transfer
pip install megatron-energon==5.0.0
```

### 1.2 apex 安装（可选）

Apex 安装，其实装不装都行，但是这个如果真的编译成功，之后训练的时候能快一些（这个主要是用来把 RMSNorm 的多个 kernel（平方、求均值、开方、缩放）融合成 1 个 CUDA kernel）：

- `apex.normalization.FusedRMSNorm`
- `aiak_training_llm/models/custom/common/local_norm.py`

如果不安装，直接打个补丁也不影响运行。

```bash
cd /jizhicfs/fenghengyi/PreMidTrainVL/apex-master

unset CC CXX
export TORCH_CUDA_ARCH_LIST="9.0"          # H20 = sm_90，省一半编译时间
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=32

# 清掉历史构建残留
rm -rf build/ dist/ *.egg-info 2>/dev/null

pip install -v --disable-pip-version-check --no-build-isolation \
    --no-cache-dir \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" . \
    2>&1 | tee /tmp/apex_build.log | tail -60
```

可以新开一个窗口检查 apex 安装，监控：

```bash
# 看进度
tail -f /tmp/apex_build.log
# 看是不是真的在编（应该有几十个 nvcc 进程）
ps -ef | grep -E "nvcc|cicc|ptxas" | grep -v grep | wc -l
```

### 1.3 flash-atten 安装

flash-atten 在第三组机器上只能装到 2.8.1 版本，这个需要格外注意。这里还是推荐拉下来本地编译。

```bash
cd /jizhicfs/fenghengyi/ai4sci
git clone --depth 1 --branch v2.8.1 \
    https://github.com/Dao-AILab/flash-attention.git flash-attention-v2.8.1

# 验证版本
ls flash-attention-v2.8.1 | head
head -3 flash-attention-v2.8.1/flash_attn/__init__.py
```

编译：

```bash
cd /jizhicfs/fenghengyi/ai4sci/flash-attention-v2.8.1

export TORCH_CUDA_ARCH_LIST="9.0"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=64
unset CC CXX

pip install -v --no-build-isolation . 2>&1 | tee /tmp/flashattn_build.log | tail -20
```

如何监控：

```bash
ps -ef | grep -E "nvcc|cicc|ptxas" | grep -v grep | wc -l

tail -f /tmp/flashattn_build.log
```

### 1.4 transformers-engine 安装

这里安装 2.7 版本：

```bash
SITE=$(python -c "import sys, sysconfig; print(sysconfig.get_paths()['purelib'])")
export CUDNN_PATH=$SITE/nvidia/cudnn
echo "Using CUDNN_PATH=$CUDNN_PATH"

# 软链 libcudnn.so -> libcudnn.so.9（只在不存在时才建）
[ -e $CUDNN_PATH/lib/libcudnn.so ] || ln -s libcudnn.so.9 $CUDNN_PATH/lib/libcudnn.so
ls $CUDNN_PATH/lib/libcudnn.so*

cd /jizhicfs/fenghengyi/ai4sci/TransformerEngine-v2.7
rm -rf build/ dist/ *.egg-info 2>/dev/null

export NVTE_FRAMEWORK=pytorch
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="9.0"
export NVTE_CUDA_ARCHS="90"
export MAX_JOBS=32
unset CC CXX

export CMAKE_PREFIX_PATH="$CUDNN_PATH:${CMAKE_PREFIX_PATH:-}"
export CPLUS_INCLUDE_PATH="$CUDNN_PATH/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$CUDNN_PATH/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CUDNN_PATH/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"

export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90 -DCUDNN_INCLUDE_DIR=$CUDNN_PATH/include -DCUDNN_LIBRARY=$CUDNN_PATH/lib/libcudnn.so"

pip install -v . --no-build-isolation 2>&1 | tee /tmp/te_build.log | tail -30
```

监控：

```bash
ps -ef | grep -E "nvcc|cicc|ptxas|cmake|ninja" | grep -v grep | wc -l

tail -f /tmp/te_build.log

grep -n "Building wheel" /tmp/te_build.log
```

## 2. SFT 训练流程 (Step 0 / 1 / 1.5 / 2)

### 2.1 Step 0：模型准备、手动缝合与转换

缝合脚本在 `/jizhicfs/fenghengyi/ai4sci/PreMidTrainVL/tools/build_stage0_qwen3moe.py`。

这里模型 architecture 我没有特调，就是简单用一个 MLP adapter 接在 vision tower 和 LLM 之间，实际训练场景下也许需要特调。另外 vision 用的是 RICE-VIT，我不确定是不是最优的选择。

```bash
cd /jizhicfs/fenghengyi/ai4sci/PreMidTrainVL

mkdir -p /jizhicfs/fenghengyi/innovator_saves/step1

python tools/build_stage0_qwen3moe.py \
    --llm_path      /jizhicfs/fenghengyi/models/Qwen3-30B-A3B-Base \
    --rice_vit_path /jizhicfs/fenghengyi/models/rice-vit-large-patch14-560 \
    --save_path     /jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-30B-A3B-stage0 \
    2>&1 | tee /tmp/build_stage0_qwen3moe.log
```

HF 模型转 megatron：`/jizhicfs/fenghengyi/ai4sci/PreMidTrainVL/examples/innovator_vl/convert/convert_30b_a3b_mcore_to_hf.sh`。

这里建议切一个新目录：

```bash
mkdir -p /jizhicfs/fenghengyi/innovator_saves/step1/_convert_workdir
cd /jizhicfs/fenghengyi/innovator_saves/step1/_convert_workdir
```

模型格式转换：

```bash
AIAK_TRAINING_PATH=/jizhicfs/fenghengyi/ai4sci/PreMidTrainVL \
AIAK_MAGATRON_PATH=/jizhicfs/fenghengyi/ai4sci/PreMidTrainVL/aiak_megatron \
bash /jizhicfs/fenghengyi/ai4sci/PreMidTrainVL/examples/innovator_vl/convert/convert_30b_a3b_hf_to_mcore.sh \
    /jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-30B-A3B-stage0 \
    /jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-30B-A3B-stage0_mcore_tp1_pp1_ep8 \
    2>&1 | tee /tmp/convert_30b_a3b.log
```

TE v2.7 升级后，旧 convert 脚本写的 BytesIO 格式 `_extra_state` 跟新的 `set_extra_state` 实现不兼容，所以统一改成空 Tensor。这里需要简单运行一个 patch 脚本：

- `/jizhicfs/fenghengyi/ai4sci/PreMidTrainVL/tools/patch_extra_state_qwen3moe.py`
- `/jizhicfs/fenghengyi/ai4sci/PreMidTrainVL/tools/patch_extra_state_qwen3moe.sh`

预计会处理 8 个 shard files：

```bash
bash /jizhicfs/fenghengyi/ai4sci/PreMidTrainVL/tools/patch_extra_state_qwen3moe.sh \
    /jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-30B-A3B-stage0_mcore_tp1_pp1_ep8
```

### 2.2 Step 1

数据集转 energon 格式，脚本我写在 `tools/data_preprocess/convert_llava_pretrain_to_energon.py`：

```bash
cd /jizhicfs/fenghengyi/ai4sci/PreMidTrainVL

python tools/data_preprocess/convert_llava_pretrain_to_energon.py \
    --json_file  /jizhicfs/fenghengyi/data/llava_pretrain/blip_laion_cc_sbu_558k.json \
    --image_dir  /jizhicfs/fenghengyi/data/llava_pretrain \
    --output_dir /jizhicfs/fenghengyi/data/llava_pretrain_energon \
    --maxcount   5000 \
    --workers    32 \
    2>&1 | tee /tmp/convert_llava_pretrain.log
```

之后启动训练，脚本在这里 `/jizhicfs/fenghengyi/ai4sci/PreMidTrainVL/examples/innovator_vl/quick_start/train_auto_stage_1_alignment_innovator_vl_qwen3moe.sh`：

```bash
cd /jizhicfs/fenghengyi/ai4sci/PreMidTrainVL
unset NCCL_ALGO  # 这个能解决腾讯机器 NCCL 的一些问题
AIAK_TRAINING_PATH=/jizhicfs/fenghengyi/ai4sci/PreMidTrainVL \
DATA_PATH=/jizhicfs/fenghengyi/data/llava_pretrain_energon \
TOKENIZER_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-30B-A3B-stage0 \
CHECKPOINT_PATH=/jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-30B-A3B-stage0_mcore_tp1_pp1_ep8 \
bash examples/innovator_vl/quick_start/train_auto_stage_1_alignment_innovator_vl_qwen3moe.sh \
    1 1 4096 1 8 5 5 8
```

最后单独配置的参数分别为：`TP PP SEQ_LEN MBS GBS NSTEP SAVE_INTERVAL EP`。

另外，我现在 Step 1 / 1.5 / 2 的默认训练配置都是仅仅作验证用，实际训练的时候还需要多注意一下调整。另外一些 model 的类名等标识性信息还是建议之后该一下子，我目前的代码没有做大的改动。

### 2.3 Step 1.5

数据转 energon：

```bash
cd /jizhicfs/fenghengyi/ai4sci/PreMidTrainVL
python tools/data_preprocess/convert_mid_training_to_energon.py \
    --parquet_dir /jizhicfs/fenghengyi/data/LLaVA-OneVision-1.5-Mid-Training-85M \
    --output_dir  /jizhicfs/fenghengyi/data/llava_mid_training_energon \
    --maxcount    5000 \
    --workers     32
```

训练脚本：`examples/innovator_vl/quick_start/train_auto_stage_1.5_mid_training_innovator_vl_qwen3moe.sh`

```bash
cd /jizhicfs/fenghengyi/ai4sci/PreMidTrainVL
bash examples/innovator_vl/quick_start/train_auto_stage_1.5_mid_training_innovator_vl_qwen3moe.sh
```

### 2.4 Step 2

依旧是数据转 energon 格式：

```bash
cd /jizhicfs/fenghengyi/ai4sci/PreMidTrainVL
python tools/data_preprocess/convert_sft_instruct_to_energon.py \
    --parquet_dir /jizhicfs/fenghengyi/data/Innovator-VL-Instruct-46M \
    --output_dir  /jizhicfs/fenghengyi/data/innovator_instruct_energon \
    --maxcount    5000
```

训练脚本：`examples/innovator_vl/quick_start/train_auto_stage_2_instruct_innovator_vl_qwen3moe.sh`

```bash
bash examples/innovator_vl/quick_start/train_auto_stage_2_instruct_innovator_vl_qwen3moe.sh
```
