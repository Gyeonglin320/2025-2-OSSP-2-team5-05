# src/02_embed_notes_icu.py
# ----------------------------------------------------------------------
# 목적:
#   ICU 48h 전처리 결과(notes_icu_48h.csv)를 읽어,
#   각 노트(TEXT)를 ClinicalBERT로 임베딩(768d)하여 .npz로 캐시하고
#   인덱스(note_index_icu.csv)를 생성한다.
#
# 특징:
#   - SAMPLE_FRAC로 split별 20%만 샘플링(중간보고용). 전체면 1.0.
#   - 기존 .npz 있으면 재사용(스킵), 없는 키만 추가 생성 → 시간 절약
#   - TEXT를 문자열로 강제 캐스팅 + 안전 처리(특수문자 제거)
#   - groupby.sample 사용(경고 제거)
#   - try/except로 문제 레코드는 스킵하고 계속 진행
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm

# =====================
# 설정
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL  = "emilyalsentzer/Bio_ClinicalBERT"
MAXLEN = 512
SAMPLE_FRAC = 0.2  # 20%만 임베딩(중간 실험). 전체 돌릴 땐 1.0로 변경.

INP = Path("data/interim") / "notes_icu_48h.csv"
OUT_DIR = Path("cache/note_embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_IDX = Path("data/interim") / "note_index_icu.csv"

# =====================
# 데이터 로드 & 샘플링
# =====================
df = pd.read_csv(
    INP,
    parse_dates=["CHARTTIME", "ADMITTIME", "DEATHTIME", "INTIME", "OUTTIME"],
)

# TEXT를 안전하게 문자열로 강제 (NaN → "")
# astype(str)만 쓰면 NaN이 "nan" 문자열이 될 수 있어, where로 먼저 비우고 변환
if "TEXT" in df.columns:
    df["TEXT"] = df["TEXT"].where(df["TEXT"].notna(), "")
    # 혹시 모를 널문자 제거(토크나이저 오류 방지)
    df["TEXT"] = df["TEXT"].astype(str).str.replace("\x00", "", regex=False)
else:
    # 혹시 TEXT가 누락된 데이터셋 방지
    df["TEXT"] = ""

# split 비율 유지한 채로 20% 샘플링 (pandas FutureWarning 없는 방식)
if SAMPLE_FRAC < 1.0:
    df = (
        df.groupby("split", group_keys=False)
          .sample(frac=SAMPLE_FRAC, random_state=42)
          .reset_index(drop=True)
    )

# =====================
# 모델 로드
# =====================
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()

# =====================
# 임베딩 함수
#  - 긴 노트는 512 토큰씩 쪼개 평균
#  - 빈 텍스트는 0벡터
# =====================
def embed_text(text: str) -> np.ndarray:
    if not isinstance(text, str) or len(text) == 0:
        return np.zeros(768, dtype=np.float32)

    # 빠른 토크나이저 경고 관련: 여기서는 청크 단위로 last_hidden_state 평균을 쓰기 때문에
    # add_special_tokens=False로 전체를 토큰화한 뒤 512씩 슬라이스 → prepare_for_model로 포맷 정리
    tokens = tok(text, add_special_tokens=False, return_tensors=None, truncation=False)
    ids = tokens.get("input_ids", [])
    if len(ids) == 0:
        return np.zeros(768, dtype=np.float32)

    vecs = []
    with torch.no_grad():
        for i in range(0, len(ids), MAXLEN):
            chunk = ids[i:i + MAXLEN]
            enc = tok.prepare_for_model(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=MAXLEN,
            )
            # 배치 차원 보정
            for k in ("input_ids", "token_type_ids", "attention_mask"):
                if k in enc and enc[k].dim() == 1:
                    enc[k] = enc[k].unsqueeze(0)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            out = mdl(**enc).last_hidden_state.mean(dim=1)  # [1,768]
            vecs.append(out.squeeze(0).cpu().numpy().astype(np.float32))

    return np.mean(vecs, axis=0).astype(np.float32)

# =====================
# 임베딩/인덱스 생성
# =====================
rows = []
new_cnt = 0
skipped = 0

it = tqdm(df.itertuples(index=False), total=len(df), desc="Indexing/Embedding (ICU)")
for r in it:
    # df 컬럼 접근: itertuples로 가져온 속성명은 대문자 그대로 매핑됨
    # r.SUBJECT_ID, r.HADM_ID, r.CHARTTIME, r.label, r.split, r.TEXT
    key = f"{getattr(r, 'SUBJECT_ID')}_{getattr(r, 'HADM_ID')}_{pd.Timestamp(getattr(r, 'CHARTTIME')).value}"
    path = OUT_DIR / f"{key}.npz"

    try:
        if not path.exists():
            v = embed_text(getattr(r, "TEXT"))
            np.savez_compressed(path, emb=v)
            new_cnt += 1

        rows.append({
            "key": key,
            "SUBJECT_ID": int(getattr(r, "SUBJECT_ID")),
            "HADM_ID": int(getattr(r, "HADM_ID")),
            "CHARTTIME": getattr(r, "CHARTTIME"),
            "label": int(getattr(r, "label")),
            "split": getattr(r, "split"),
        })

    except Exception as e:
        skipped += 1
        print(f"[skip] {key}: {e}")
        continue

# 저장
pd.DataFrame(rows).to_csv(OUT_IDX, index=False)
print(f"\nSaved index: {OUT_IDX} | rows={len(rows)}")
print(f"New embeddings created: {new_cnt}  | Skipped rows: {skipped}")
print(f"Embeddings dir: {OUT_DIR.resolve()}")
