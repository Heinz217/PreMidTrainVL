#!/usr/bin/env bash
# Patch the _extra_state keys of an mcore checkpoint after running
# convert_30b_a3b_hf_to_mcore.sh (Qwen3-MoE / Innovator-VL-30B-A3B variant).
#
# Why this is needed: see tools/patch_extra_state_qwen3moe.py header comment.
#
# Usage
# -----
#   bash tools/patch_extra_state_qwen3moe.sh <ckpt_dir>
#
# Example
# -------
#   bash tools/patch_extra_state_qwen3moe.sh \
#       /jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-30B-A3B-stage0_mcore_tp1_pp1_ep8

set -euo pipefail

CKPT_DIR="${1:-}"

if [[ -z "$CKPT_DIR" ]]; then
    echo "Usage: $0 <ckpt_dir>"
    echo
    echo "Example:"
    echo "  $0 /jizhicfs/fenghengyi/innovator_saves/step1/Innovator-VL-30B-A3B-stage0_mcore_tp1_pp1_ep8"
    exit 1
fi

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "[ERROR] checkpoint directory does not exist: $CKPT_DIR"
    exit 1
fi

# Resolve the directory containing this script so we can locate the .py reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/patch_extra_state_qwen3moe.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "[ERROR] companion python script not found: $PY_SCRIPT"
    exit 1
fi

LOG="/tmp/patch_extra_state_qwen3moe_$(date +%Y%m%d_%H%M%S).log"
echo "[info] ckpt_dir = $CKPT_DIR"
echo "[info] log file = $LOG"
echo

python "$PY_SCRIPT" --ckpt_dir "$CKPT_DIR" 2>&1 | tee "$LOG"
