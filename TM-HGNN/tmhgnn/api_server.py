# tmhgnn/api_server.py

import os
import sys
import json
import math
import torch
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ====== 경로 설정 (프로젝트 루트 추가) ======
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# ====== TM-HGNN 관련 import ======
from tmhgnn.conf import get_conf          # args 불러오는 함수 (train 때 쓰던 거)
from tmhgnn.net import TM_HGNN          # 실제 모델 클래스 이름에 맞추기
from tmhgnn import net as networks

# ====== 캐시 / 모델 경로 설정 ======
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
GRAPH_DIR = os.path.join(CACHE_DIR, "graphs")
META_PATH = os.path.join(CACHE_DIR, "patients_meta.json")

# ====== 환경 변수 기반 설정 ======
DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "tmhgnn",
    "grid_search_results",
    "lr_0.001_wd_1e-6",
    "best_model.pth",
)

MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

DEVICE = os.environ.get("DEVICE", "cuda")
if DEVICE.startswith("cuda") and torch.cuda.is_available():
    device = torch.device(DEVICE)
else:
    device = torch.device("cpu")


# ====== FastAPI 앱 초기화 ======
app = FastAPI(
    title="ICU Mortality Prediction API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 나중에 v0.dev 도메인으로 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== 메타데이터 로딩 ======
if not os.path.exists(META_PATH):
    raise RuntimeError(f"patients_meta.json 이 없음: {META_PATH}")

with open(META_PATH, "r") as f:
    PATIENTS_META = json.load(f)

ID_TO_META = {p["id"]: p for p in PATIENTS_META}

# ====== 모델 로딩 ======
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"MODEL_PATH가 잘못됨: {MODEL_PATH}")

args = get_conf()        

# train.py 와 동일하게 node_features 설정
if args.tokenizer == "clinicalbert":
    in_dim = 4 + 768
elif args.tokenizer == "word2vec":
    in_dim = 4 + 100
else:
    raise ValueError(f"Unknown tokenizer: {args.tokenizer}")

args.node_features = in_dim
model = getattr(networks, args.model)(
    num_features=args.node_features,
    hidden_channels=args.hidden_channels * args.heads_1,
)
model.to(device)
model.eval()
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# ====== 응답 스키마 정의 ======
class Patient(BaseModel):
    id: str
    label: int
    age: int | None = None
    sex: str | None = None
    icu: str | None = None

class PredictionResult(BaseModel):
    patient: Patient
    pred_prob: float
    ground_truth: int


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

# ====== 엔드포인트 ======

@app.get("/patients", response_model=List[Patient])
def list_patients():
    """
    캐시해둔 test_0 ~ test_9 환자 리스트 반환
    """
    return [Patient(**p) for p in PATIENTS_META]


@app.get("/patients/{patient_id}", response_model=PredictionResult)
def predict_patient(patient_id: str):
    """
    환자별 하이퍼그래프 캐시 로딩 -> TM-HGNN 실제 inference -> 예측 확률 반환
    """
    if patient_id not in ID_TO_META:
        raise HTTPException(status_code=404, detail="Unknown patient_id")

    meta = ID_TO_META[patient_id]
    graph_path = os.path.join(GRAPH_DIR, f"{patient_id}.pt")

    if not os.path.exists(graph_path):
        raise HTTPException(status_code=404, detail="Cached graph not found")

    # 1) 그래프 로드
    data = torch.load(graph_path, map_location=device)
    data = data.to(device)

    # 2) 모델 inference
    with torch.no_grad():
        logits = model(data.x_n, data.edge_index_n, data.edge_index_mask, data.batch)

        if isinstance(logits, torch.Tensor):
            logit = float(logits.view(-1).item())
        else:
            raise RuntimeError("예상치 못한 model output 형식입니다.")


    prob = sigmoid(logit)

    patient = Patient(**meta)
    return PredictionResult(
        patient=patient,
        pred_prob=prob,
        ground_truth=meta["label"],
    )
