#!/usr/bin/env bash
set -e

# ===== 설정 =====
export MODEL_PATH="/data/project/juhyeon/MAGNET/OSS_2/TM-HGNN/tmhgnn/grid_search_results/lr_0.001_wd_1e-6/best_model.pth"
export DEVICE="cuda:0"     # or "cpu"

echo "MODEL_PATH = $MODEL_PATH"
echo "DEVICE = $DEVICE"

# ===== Conda 활성화 =====
source /data/project/juhyeon/anaconda/etc/profile.d/conda.sh
conda activate myenv

# ===== FastAPI 서버 실행 =====
uvicorn tmhgnn.api_server:app --host 0.0.0.0 --port 8000 --reload
