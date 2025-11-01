# src/01_prepare_data.py
# ------------------------------------------------------------------------
# 목적:
#   MIMIC-III (Kaggle 10k) 데이터셋을 기반으로
#   1. 입원(admissions) 정보와 임상노트(noteevents)를 병합하고
#   2. 입원 후 48시간 이내에 작성된 노트만 남긴 뒤
#   3. 사망 여부(label)과 train/val/test 분할 정보를 추가하여 CSV로 저장한다.
# ------------------------------------------------------------------------

import pandas as pd
from pathlib import Path

# 1) 경로 준비
RAW = Path("data/raw")
OUT = Path("data/interim"); OUT.mkdir(parents=True, exist_ok=True)

# 2) 원본 CSV 로드
#    날짜형 칼럼을 datetime 타입으로 자동 변환하여 시간 비교 가능하게 함.
adm = pd.read_csv(RAW / "ADMISSIONS.csv",
                  parse_dates=["ADMITTIME", "DEATHTIME", "DISCHTIME", "EDREGTIME", "EDOUTTIME"])
notes = pd.read_csv(RAW / "NOTEEVENTS.csv",
                    parse_dates=["CHARTDATE", "CHARTTIME", "STORETIME"])

# 3) 필요한 칼럼만 추출해 메모리 절약 및 가독성 향상
adm = adm[["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DEATHTIME", "HOSPITAL_EXPIRE_FLAG"]]
notes = notes[["SUBJECT_ID", "HADM_ID", "CHARTTIME", "CATEGORY", "ISERROR", "TEXT"]]

# 4) 오류 노트 제거
#    ISERROR == 1 → 작성 오류가 표시된 노트 → 제거
#    NaN(결측)은 0(False)으로 처리하여 유지
notes = notes[notes["ISERROR"].fillna(0) != 1]

# 5) ADMISSIONS + NOTEEVENTS 병합
#    공통키 (SUBJECT_ID, HADM_ID)를 기준으로 inner join
#    → 두 테이블 모두에 존재하는 입원-노트 조합만 남김
df = notes.merge(adm, on=["SUBJECT_ID", "HADM_ID"], how="inner")

# 6) 입원 기준 48시간 이내 노트만 필터링
#    CHARTTIME(노트 작성 시각)이 ADMITTIME 이후 48시간 이하인지 검사
#    비교가 불가능한 NaT(결측)값은 dropna로 제거
df = df.dropna(subset=["ADMITTIME", "CHARTTIME"])
df = df[(df["CHARTTIME"] >= df["ADMITTIME"]) &
        (df["CHARTTIME"] <= df["ADMITTIME"] + pd.Timedelta(hours=48))]

# 7) 사망 여부 라벨 생성
#    HOSPITAL_EXPIRE_FLAG: 1=사망, 0=생존
#    결측값은 0으로 채우고 int형으로 변환
df["label"] = df["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int)

# 8) 텍스트 전처리
#    - 결측 텍스트는 빈 문자열("")로 대체
#    - 연속된 공백을 하나로 축소
#    - 양끝 공백 제거
df["TEXT"] = df["TEXT"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()

# 9) train/val/test 분할 (환자 단위)
#    같은 환자(SUBJECT_ID)가 여러 입원을 가질 수 있으므로,
#    데이터 누설 방지를 위해 환자 기준으로 분할
#    70% train / 15% val / 15% test 비율로 랜덤 분리
pats = (df[["SUBJECT_ID", "label"]]
        .drop_duplicates("SUBJECT_ID") # 환자별 1개씩만 남김
        .sample(frac=1, random_state=42)) # 순서 섞기
n = len(pats)
train_ids = set(pats.iloc[:int(n*0.70)].SUBJECT_ID)
val_ids   = set(pats.iloc[int(n*0.70):int(n*0.85)].SUBJECT_ID)

# SUBJECT_ID를 매핑하여 각 행의 split 컬럼(train/val/test) 지정
df["split"] = df["SUBJECT_ID"].map(lambda x: "train" if x in train_ids
                                   else ("val" if x in val_ids else "test"))

# 10) 결과 저장
#     data/interim/notes_48h.csv 파일로 저장
out_path = OUT / "notes_48h.csv"
df.to_csv(out_path, index=False)
print(f"saved: {out_path}  rows: {len(df)}  (train/val/test counts:",
      df["split"].value_counts().to_dict(), ")")
