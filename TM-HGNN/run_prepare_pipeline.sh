#!/usr/bin/env bash
set -euo pipefail

# ==== 설정 부분 ====
BASE_DIR="/data/project/juhyeon/MAGNET/OSS/TM-HGNN/graph_construction/prepare_notes"

RAW_PATH="/data/project/juhyeon/MAGNET/OSS/TM-HGNN/data/DATA_RAW"
PRE_PATH="/data/project/juhyeon/MAGNET/OSS/TM-HGNN/data/DATA_PRE"
TASK="in-hospital-mortality"
TOKENIZER="clinicalbert"
DIMENSION=768
# ===================

# === CPU 코어 개수 제한 ===
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8

source /data/project/juhyeon/anaconda/etc/profile.d/conda.sh
conda activate myenv

cd "$BASE_DIR"

echo "=== 1) extract_cleaned_notes.py 실행 ==="
python3 extract_cleaned_notes.py \
  --tokenizer "$TOKENIZER" \
  --dimension "$DIMENSION" \
  --raw_path "$RAW_PATH"

echo "=== 2) create_hyper_df.py 실행 ==="
python3 create_hyper_df.py \
  --raw_path "${RAW_PATH}/" \
  --pre_path "${PRE_PATH}/" \
  --task "$TASK" \
  --action make

echo "=== 3) ConstructDatasetByNotes.py 실행 ==="
python3 ConstructDatasetByNotes.py

echo "=== 4) PygNotesGraphDataset.py (train split) 실행 ==="
python3 PygNotesGraphDataset.py \
  --name "$TASK" \
  --split train \
  --action create \
  --tokenizer "$TOKENIZER" \
  --pre_path "$PRE_PATH"

echo "=== 5) PygNotesGraphDataset.py (test split) 실행 ==="
python3 PygNotesGraphDataset.py \
  --name "$TASK" \
  --split test \
  --action create \
  --tokenizer "$TOKENIZER" \
  --pre_path "$PRE_PATH"

echo "=== 모든 단계 완료! ==="