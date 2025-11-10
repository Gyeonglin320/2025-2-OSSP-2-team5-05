# src/01_prepare_data_icu.py
# ICU 입실(INTIME) 기준 48시간 내 노트만 남기고, 라벨/스플릿 포함 CSV 생성

import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/interim"); OUT.mkdir(parents=True, exist_ok=True)

# 1) 원본 로드
adm = pd.read_csv(RAW / "ADMISSIONS.csv",
                  parse_dates=["ADMITTIME", "DEATHTIME", "DISCHTIME", "EDREGTIME", "EDOUTTIME"])
notes = pd.read_csv(RAW / "NOTEEVENTS.csv",
                    parse_dates=["CHARTDATE", "CHARTTIME", "STORETIME"])
icu = pd.read_csv(RAW / "ICUSTAYS.csv",
                  parse_dates=["INTIME","OUTTIME"])

# 2) 필요한 칼럼만
adm = adm[["SUBJECT_ID","HADM_ID","ADMITTIME","DEATHTIME","HOSPITAL_EXPIRE_FLAG"]]
notes = notes[["SUBJECT_ID","HADM_ID","CHARTTIME","CATEGORY","ISERROR","TEXT"]]
icu = icu[["SUBJECT_ID","HADM_ID","ICUSTAY_ID","INTIME","OUTTIME"]]

# ++ 동일 HADM_ID에 ICU 입실이 여러 번이면, 가장 이른 INTIME만 사용
icu = icu.sort_values("INTIME").drop_duplicates(["SUBJECT_ID","HADM_ID"], keep="first")

# 3) 오류 노트 제거 + 텍스트 정제
notes = notes[notes["ISERROR"].fillna(0) != 1]
notes["TEXT"] = notes["TEXT"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()

# 4) 병합: NOTEEVENTS + ADMISSIONS + ICUSTAYS
df = notes.merge(adm, on=["SUBJECT_ID","HADM_ID"], how="inner") \
          .merge(icu,  on=["SUBJECT_ID","HADM_ID"], how="inner")

# 5) ICU 입실 기준 48시간 필터
df = df.dropna(subset=["INTIME","CHARTTIME"])
df = df[(df["CHARTTIME"] >= df["INTIME"]) &
        (df["CHARTTIME"] <= df["INTIME"] + pd.Timedelta(hours=48))]

# IF, ICU 체류 내로 제한한다면?
# df = df[df["CHARTTIME"] <= df["OUTTIME"]]

# 6) 라벨
df["label"] = df["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int)

# 7) 환자 단위 split (70/15/15)
pats = (df[["SUBJECT_ID","label"]].drop_duplicates("SUBJECT_ID").sample(frac=1, random_state=42))
n = len(pats)
train_ids = set(pats.iloc[:int(n*0.70)].SUBJECT_ID)
val_ids   = set(pats.iloc[int(n*0.70):int(n*0.85)].SUBJECT_ID)
df["split"] = df["SUBJECT_ID"].map(lambda x: "train" if x in train_ids else ("val" if x in val_ids else "test"))

# 8) 저장
out_path = OUT / "notes_icu_48h.csv"
df.to_csv(out_path, index=False)
print("saved:", out_path, "| rows:", len(df))
print("split counts:", df["split"].value_counts().to_dict())