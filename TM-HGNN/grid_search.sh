#!/usr/bin/env bash
set -euo pipefail

# ==== 기본 설정 ====
ROOT_DIR="/data/project/juhyeon/MAGNET/OSS/TM-HGNN/tmhgnn"
TRAIN_PY="${ROOT_DIR}/train.py"

# === CPU Thread 제한 ===
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8

# === conda 환경 활성화 ===
CONDA_SH="/data/project/juhyeon/anaconda/etc/profile.d/conda.sh"

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
else
    echo "[WARN] cannot find conda.sh, falling back to ~/.bashrc"
    source ~/.bashrc
fi

conda activate myenv
# ======================

# TM-HGNN 학습 설정
TASK="in-hospital-mortality"
TOKENIZER="clinicalbert"
DLOAD="multi_hyper"
GPU=3
PATIENCE=5

# Grid Search 파라미터
LRS=("0.001" "0.0001")
WDS=("0" "1e-5" "1e-6" "1e-7" "1e-8")

# 결과 저장 루트
EXP_ROOT="${ROOT_DIR}/grid_search_results"
mkdir -p "${EXP_ROOT}"

# ROOT_DIR 기준 실행
cd "${ROOT_DIR}" || exit 1

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do

    EXP_NAME="lr_${lr}_wd_${wd}"
    EXP_DIR="${EXP_ROOT}/${EXP_NAME}"
    mkdir -p "${EXP_DIR}"

    echo "============================"
    echo " Running: LR=${lr}, WD=${wd}"
    echo " Saving to: ${EXP_DIR}"
    echo "============================"

    python3 "${TRAIN_PY}" \
      --task "${TASK}" \
      --tokenizer "${TOKENIZER}" \
      --dload "${DLOAD}" \
      --gpu "${GPU}" \
      --early_stop_patience "${PATIENCE}" \
      --init-lr "${lr}" \
      --weight-decay "${wd}" \
      --output "${EXP_DIR}/best_model.pth" \
      --exp-name "${EXP_DIR}/results" \
      > "${EXP_DIR}/log.txt" 2>&1

  done
done

echo "====== Grid Search Finished ======"
