# ----------------------------------------------------------
# ClinicalBERT 임베딩(.npz)을 시간순 시퀀스로 묶어
# LSTM으로 "입원 단위" 사망여부를 예측하는 베이스라인.
# - 시퀀스 단위: (SUBJECT_ID, HADM_ID)
# - 시퀀스 구성: CHARTTIME 오름차순으로 note 임베딩 나열
# - 라벨: 같은 입원 내 노트가 공유하는 HOSPITAL_EXPIRE_FLAG (0/1)
# ----------------------------------------------------------

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# --------------------
# 하이퍼파라미터
# --------------------
EMB_DIR = Path("cache/note_embeddings")
IDX_CSV = Path("data/interim/note_index.csv")
BATCH_SIZE = 64
HIDDEN = 256
LR = 1e-3
EPOCHS = 10
MAX_SEQ = 32      # 입원당 최대 노트 수(초과 시 앞부분 잘라서 최근 MAX_SEQ 유지)
INPUT_DIM = 768   # ClinicalBERT 임베딩 차원
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------
# 데이터셋 준비
# --------------------
idx = pd.read_csv(IDX_CSV, parse_dates=["CHARTTIME"])

# (SUBJECT_ID, HADM_ID) 단위로 그룹핑해서
# - 시간순 정렬
# - split과 label은 그룹 내 공통값 사용
def build_sequences(split):
    g = idx[idx["split"] == split].copy()
    # 그룹: 환자-입원 단위
    grouped = g.groupby(["SUBJECT_ID", "HADM_ID"], sort=False)
    seqs, labels = [], []

    for (sid, hadm), df in tqdm(grouped, desc=f"Build {split} sequences"):
        df = df.sort_values("CHARTTIME")
        xs = []
        for _, r in df.iterrows():
            path = EMB_DIR / f"{r.key}.npz"
            if not path.exists():
                continue
            try:
                v = np.load(path)["emb"]  # (768,)
                xs.append(v.astype(np.float32))
            except Exception:
                continue

        if len(xs) == 0:
            continue

        y = int(df["label"].iloc[0])  # 동일 입원 내 라벨 동일
        # 길이 제한: 최근 MAX_SEQ개만 사용 (초기 48시간 내에서도 최근 노트가 더 informative 가정)
        if len(xs) > MAX_SEQ:
            xs = xs[-MAX_SEQ:]
        seqs.append(np.stack(xs))  # (L, 768)
        labels.append(y)

    return seqs, np.array(labels, dtype=np.int64)

class SeqDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, i):
        x = self.sequences[i]       # (L,768)
        y = self.labels[i]          # scalar
        L = x.shape[0]
        return x, L, y

def collate_batch(batch):
    # batch: list of (np.array[L,768], length, label)
    lengths = torch.tensor([b[1] for b in batch], dtype=torch.long)
    labels  = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
    maxL = lengths.max().item()
    B = len(batch)
    xpad = torch.zeros(B, maxL, INPUT_DIM, dtype=torch.float32)
    for i, (x, L, _) in enumerate(batch):
        xpad[i, :L] = torch.from_numpy(x)
    return xpad, lengths, labels

# --------------------
# 모델
# --------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden=HIDDEN, bidir=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True,
                            bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x, lengths):
        # x: (B, T, D), lengths: (B,)
        # pack → LSTM → unpack 안 해도 마지막 유효 타임스텝의 hidden을 직접 뽑을 수도 있으나
        # 여기서는 pack_padded_sequence 사용
        lengths_cpu = lengths.cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)
        # h_n: (num_directions, B, hidden)
        # bidir이면 [fwd_last, bwd_last] concat
        if self.lstm.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)
        else:
            h = h_n[-1]                               # (B, H)
        h = self.dropout(h)
        logit = self.fc(h)                            # (B,1)
        return logit

# --------------------
# 데이터 로드
# --------------------
train_seqs, y_train = build_sequences("train")
val_seqs,   y_val   = build_sequences("val")
test_seqs,  y_test  = build_sequences("test")

print(f"#admissions   | train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}")

train_loader = DataLoader(SeqDataset(train_seqs, y_train), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, collate_fn=collate_batch)
val_loader   = DataLoader(SeqDataset(val_seqs,   y_val),   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, collate_fn=collate_batch)
test_loader  = DataLoader(SeqDataset(test_seqs,  y_test),  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, collate_fn=collate_batch)

# --------------------
# 학습 루프
# --------------------
model = LSTMClassifier().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def evaluate(loader):
    model.eval()
    all_prob, all_y = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(DEVICE); lengths = lengths.to(DEVICE); y = y.to(DEVICE)
            logit = model(x, lengths)
            prob = torch.sigmoid(logit).cpu().numpy().ravel()
            all_prob.append(prob)
            all_y.append(y.cpu().numpy().ravel())
    all_prob = np.concatenate(all_prob)
    all_y = np.concatenate(all_y).astype(int)
    auc = roc_auc_score(all_y, all_prob)
    auprc = average_precision_score(all_y, all_prob)
    return auc, auprc

for epoch in range(1, EPOCHS+1):
    model.train()
    running = 0.0
    for x, lengths, y in train_loader:
        x = x.to(DEVICE); lengths = lengths.to(DEVICE); y = y.to(DEVICE)
        optimizer.zero_grad()
        logit = model(x, lengths)
        loss = criterion(logit, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)

    train_loss = running / len(train_loader.dataset)
    val_auc, val_auprc = evaluate(val_loader)
    print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | "
          f"val_AUROC={val_auc:.3f} | val_AUPRC={val_auprc:.3f}")

# --------------------
# 최종 테스트
# --------------------
test_auc, test_auprc = evaluate(test_loader)
print(f"\n[TEST] AUROC={test_auc:.3f} | AUPRC={test_auprc:.3f}")
