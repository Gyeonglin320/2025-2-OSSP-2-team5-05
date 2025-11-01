# src/02_embed_notes.py
# ------------------------------------------------------------------------
# 목적:
#   01단계에서 전처리된 notes_48h.csv 데이터를 불러와
#   각 임상 노트(TEXT)를 ClinicalBERT 모델로 임베딩(768차원 벡터)한 뒤
#   .npz 파일로 저장하고, 인덱스(note_index.csv)를 생성함.
# ------------------------------------------------------------------------
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm

# 설정 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # GPU 사용 가능하면 GPU로, 아니면 CPU
MODEL  = "emilyalsentzer/Bio_ClinicalBERT" # 사전학습된 ClinicalBERT 모델
MAXLEN = 512 # BERT 최대 토큰 길이
SAMPLE_FRAC = 0.2  # 전체 데이터 중 일부만 사용(0.2=20%) — 테스트용

INP = Path("data/interim")/"notes_48h.csv" # 전처리된 입력 데이터 경로
OUT = Path("cache/note_embeddings"); OUT.mkdir(parents=True, exist_ok=True) # 임베딩 저장 폴더 생성


# 1) 데이터 로드 및 샘플링
df = pd.read_csv(INP, parse_dates=["CHARTTIME","ADMITTIME","DEATHTIME"])

# 20%만 샘플링할 경우 (훈련/검증/테스트 비율 유지)
if SAMPLE_FRAC < 1.0:
    df = (df.groupby("split", group_keys=False)
            .apply(lambda g: g.sample(frac=SAMPLE_FRAC, random_state=42))
            .reset_index(drop=True))

# 2) ClinicalBERT 모델 및 토크나이저 로드
tok = AutoTokenizer.from_pretrained(MODEL) # 텍스트를 토큰 ID로 변환
mdl = AutoModel.from_pretrained(MODEL).to(DEVICE).eval() # 모델 로드 + 평가 모드 설정(eval)

# 3) 텍스트 -> 임베딩 함수 정의
def embed_text(text: str):
    # 한 개의 임상 노트를 입력받아 ClinicalBERT 임베딩(768차원 벡터)으로 변환
    # 긴 노트는 MAXLEN(512) 토큰 단위로 쪼개어 각각 임베딩한 뒤 평균을 구함
    tokens = tok(text, add_special_tokens=False, return_tensors=None, truncation=False)
    ids = tokens["input_ids"]

    # 빈 텍스트인 경우 0벡터 반환
    if len(ids) == 0:
        return np.zeros(768, dtype=np.float32)

    vecs = [] # 각 청크의 임베딩을 저장할 리스트
    with torch.no_grad(): # 모델을 학습시키지 않고 순전파만 수행
        for i in range(0, len(ids), MAXLEN):
            # 512 토큰 단위로 청크 분할
            chunk = ids[i:i+MAXLEN]

            # BERT 입력 형식에 맞게 인코딩
            enc = tok.prepare_for_model(chunk, return_tensors="pt",
                                        truncation=True, max_length=MAXLEN)
            
            # prepare_for_model이 1D 텐서를 줄 수 있어서 배치 차원 보정
            for k in ("input_ids","token_type_ids","attention_mask"):
                if k in enc and enc[k].dim() == 1:
                    enc[k] = enc[k].unsqueeze(0)

            # GPU or CPU로 텐서 이동
            enc = {k: v.to(DEVICE) for k,v in enc.items()}

            # ClinicalBERT를 이용해 문맥 임베딩 생성
            # 각 토큰의 마지막 hidden state(768차원) 평균을 청크 임베딩으로 사용
            out = mdl(**enc).last_hidden_state.mean(dim=1)  # [1,768]
            vecs.append(out.squeeze(0).cpu().numpy().astype(np.float32))

    # 여러 청크의 평균 -> 최종 문서 임베딩 (1x768)
    return np.mean(vecs, axis=0).astype(np.float32)

# 4) 모든 노트에 대해 임베딩 생성 및 저장
rows = [] # 인덱스 정보 저장용 리스트

for _, r in tqdm(df.iterrows(), total=len(df), desc="Embedding notes"):
    # 파일 이름을 환자ID_입원ID_작성시각 으로 지정
    key = f'{r.SUBJECT_ID}_{r.HADM_ID}_{pd.Timestamp(r.CHARTTIME).value}'
    path = OUT / f"{key}.npz"
    try:
        # 이미 저장된 파일이 없으면 임베딩 생성 및 저장
        if not path.exists():
            v = embed_text(r["TEXT"])
            np.savez_compressed(path, emb=v) # .npz 파일로 저장 (용량 절약)

        # 인덱스용 정보 저장
        rows.append({"key":key,
                     "SUBJECT_ID":r.SUBJECT_ID,
                     "HADM_ID":r.HADM_ID,
                     "CHARTTIME":r.CHARTTIME,
                     "label":int(r.label),
                     "split":r.split})
        
    except Exception as e:
        # 임베딩 중 오류 발생 시 건너뜀
        print(f"[skip] {key}: {e}")

# 5) 인덱스 파일 저장
# note_index.csv: 각 노트의 메타정보(환자ID, 입원ID, 작성시각, 라벨, split)
pd.DataFrame(rows).to_csv("data/interim/note_index.csv", index=False)
print("saved: data/interim/note_index.csv  rows:", len(rows))
print("Note embeddings saved under cache/note_embeddings/")
