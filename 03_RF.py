# ----------------------------------------------------------
# 목적: ClinicalBERT 임베딩 (.npz) + note_index_icu.csv 기반으로
#       RandomForest 학습 및 성능 평가, 결과 CSV로 저장
# ----------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# ----------------------------------------------------------
# 1. 데이터 로드
# ----------------------------------------------------------
IDX = pd.read_csv("data/interim/note_index_icu.csv")
EMB_DIR = Path("cache/note_embeddings")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)  # 로그 디렉토리 없으면 생성

def load_split(split):
    """split(train/val/test)에 해당하는 임베딩 벡터와 라벨 불러오기"""
    rows = IDX[IDX.split == split]
    X, y = [], []
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc=f"Loading {split}"):
        path = EMB_DIR / f"{r.key}.npz"
        if path.exists():
            v = np.load(path)["emb"]
            X.append(v)
            y.append(r.label)
    return np.stack(X), np.array(y)

X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")

print(f"Loaded: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

# ----------------------------------------------------------
# 2. Random Forest 학습
# ----------------------------------------------------------
print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)

rf.fit(X_train, y_train)

# ----------------------------------------------------------
# 3. 평가 (AUROC, AUPRC, F1)
# ----------------------------------------------------------
rf_val_prob = rf.predict_proba(X_val)[:, 1]
rf_test_prob = rf.predict_proba(X_test)[:, 1]

# 기본 threshold = 0.5 (나중에 튜닝 가능)
threshold = 0.5
rf_val_pred = (rf_val_prob >= threshold).astype(int)
rf_test_pred = (rf_test_prob >= threshold).astype(int)

val_auc = roc_auc_score(y_val, rf_val_prob)
val_auprc = average_precision_score(y_val, rf_val_prob)
val_f1 = f1_score(y_val, rf_val_pred)

test_auc = roc_auc_score(y_test, rf_test_prob)
test_auprc = average_precision_score(y_test, rf_test_prob)
test_f1 = f1_score(y_test, rf_test_pred)

print(f"\nValidation  AUROC={val_auc:.3f} | AUPRC={val_auprc:.3f} | F1={val_f1:.3f}")
print(f"Test        AUROC={test_auc:.3f} | AUPRC={test_auprc:.3f} | F1={test_f1:.3f}")
print("\nRandom Forest training complete")

# ----------------------------------------------------------
# 4. 결과 CSV 저장
# ----------------------------------------------------------
now = datetime.now().strftime("%Y%m%d_%H%M%S")
result = pd.DataFrame([{
    "timestamp": now,
    "model": "RandomForest",
    "n_estimators": rf.n_estimators,
    "max_depth": rf.max_depth,
    "threshold": threshold,
    "val_AUROC": val_auc,
    "val_AUPRC": val_auprc,
    "val_F1": val_f1,
    "test_AUROC": test_auc,
    "test_AUPRC": test_auprc,
    "test_F1": test_f1,
    "train_size": len(y_train),
    "val_size": len(y_val),
    "test_size": len(y_test),
}])

log_path = LOG_DIR / "rf_results_log.csv"

# 파일이 이미 있으면 append, 없으면 새로 생성
if log_path.exists():
    existing = pd.read_csv(log_path)
    result = pd.concat([existing, result], ignore_index=True)

result.to_csv(log_path, index=False)
print(f"\nSaved results to: {log_path}")