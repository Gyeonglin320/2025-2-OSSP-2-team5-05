# src/03_train_baselines.py
# ----------------------------------------------------------
# 목적: ClinicalBERT 임베딩 (.npz)과 note_index.csv를 불러와
#       MLP / RandomForest 분류기를 학습하고 성능 평가
# ----------------------------------------------------------

import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
from tqdm import tqdm

# 1. 데이터 로드
IDX = pd.read_csv("data/interim/note_index.csv")
EMB_DIR = Path("cache/note_embeddings")

def load_split(split):
    # split(train/val/test)에 해당하는 임베딩 벡터와 라벨 불러오기
    rows = IDX[IDX.split == split]
    X, y = [], []
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc=f"Loading {split}"):
        path = EMB_DIR / f"{r.key}.npz"
        if path.exists():
            v = np.load(path)["emb"]
            X.append(v)
            y.append(r.label)
    X = np.stack(X)
    y = np.array(y)
    return X, y

X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")

print(f"Loaded: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

# ----------------------------------------------------------
# 2. MLP 모델 정의 (PyTorch)
# ----------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=768, hidden=256, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
mlp = MLP().to(device)

# ----------------------------------------------------------
# 3. 학습 루프
# ----------------------------------------------------------
criterion = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(mlp.parameters(), lr=1e-4)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_val_t   = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

for epoch in range(10):
    mlp.train()
    opt.zero_grad()
    out = mlp(X_train_t)
    loss = criterion(out, y_train_t)
    loss.backward()
    opt.step()

    mlp.eval()
    with torch.no_grad():
        val_logits = mlp(X_val_t)
        val_prob = torch.sigmoid(val_logits).cpu().numpy().ravel()
        val_loss = criterion(val_logits, y_val_t).item()
        auc = roc_auc_score(y_val, val_prob)
        auprc = average_precision_score(y_val, val_prob)
    print(f"Epoch {epoch+1}/10 | train_loss={loss.item():.4f} | val_loss={val_loss:.4f} | AUROC={auc:.3f} | AUPRC={auprc:.3f}")

# ----------------------------------------------------------
# 4️. Random Forest
# ----------------------------------------------------------
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred)
rf_auprc = average_precision_score(y_test, rf_pred)
print(f"RF Test AUROC={rf_auc:.3f}, AUPRC={rf_auprc:.3f}")

# ----------------------------------------------------------
# 5. 최종 MLP 평가
# ----------------------------------------------------------
mlp.eval()
with torch.no_grad():
    y_pred = torch.sigmoid(mlp(torch.tensor(X_test, dtype=torch.float32).to(device)))
    prob = y_pred.cpu().numpy().ravel()
test_auc = roc_auc_score(y_test, prob)
test_auprc = average_precision_score(y_test, prob)
print(f"\nMLP Test AUROC={test_auc:.3f}, AUPRC={test_auprc:.3f}")
