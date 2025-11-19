#!/bin/bash

# ================================
# Grid Search for TM-HGNN (절대경로 버전)
# ================================

# train.py 가 있는 디렉토리 (네가 알려준 위치 기반)
ROOT_DIR="/data1/project/juhyeon/MAGNET/OSS/TM-HGNN/tmhgnn"
TRAIN_PY="${ROOT_DIR}/train.py"

# 그리드 서치 하이퍼파라미터
LRS=("0.001" "0.005")
WDS=("1e-3" "1e-4" "1e-5" "1e-6")

# 결과 저장 루트 (train.py 기준 상대 경로)
EXP_ROOT="${ROOT_DIR}/grid_search_results"
mkdir -p "${EXP_ROOT}"

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do

    EXP_NAME="lr_${lr}_wd_${wd}"
    EXP_DIR="${EXP_ROOT}/${EXP_NAME}"
    mkdir -p "${EXP_DIR}"

    echo "============================"
    echo " Running: LR=${lr}, WD=${wd}"
    echo " Saving to: ${EXP_DIR}"
    echo "============================"

    # 항상 ROOT_DIR 안에서 train.py 실행
    cd "${ROOT_DIR}" || exit 1

    python3 "${TRAIN_PY}" \
      --init-lr "${lr}" \
      --weight-decay "${wd}" \
      --output "${EXP_DIR}/best_model.pth" \
      --exp-name "${EXP_DIR}/results" \
      > "${EXP_DIR}/log.txt" 2>&1

  done
done

echo "====== Grid Search Finished ======"
