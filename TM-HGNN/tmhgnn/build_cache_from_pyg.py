# build_cache_from_pyg.py
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import json
import torch
from tqdm import tqdm

# PygNotesGraphDataset 불러오기
from graph_construction.prepare_notes.PygNotesGraphDataset import Load_PygNotesGraphDataset

# ===== 경로 / 설정 =====
TASK = "in-hospital-mortality"
TOKENIZER = "clinicalbert"

# 너가 실제로 vocab.txt / 데이터 만든 경로
RAW_PATH = "/data/project/juhyeon/MAGNET/OSS_2/TM-HGNN/data/DATA_RAW"
PRE_PATH = "/data/project/juhyeon/MAGNET/OSS_2/TM-HGNN/data/DATA_PRE"

VOCAB_PATH = os.path.join(RAW_PATH, "root", "vocab.txt")

CACHE_DIR = "./cache"  # TM-HGNN 루트 기준
GRAPH_DIR = os.path.join(CACHE_DIR, "graphs")

os.makedirs(GRAPH_DIR, exist_ok=True)

def main():
    # 1) vocab 로드
    dictionary = open(VOCAB_PATH).read().split()

    # 2) 이미 만들어진 test_hyper_clinicalbert.pt 로딩
    dataset = Load_PygNotesGraphDataset(
        name=TASK,
        split="test",
        tokenizer=TOKENIZER,
        dictionary=dictionary,
        data_type="hyper",
    )

    print(f"총 test 샘플 수: {len(dataset)}")
    print("앞 10개(0~9 index)만 캐시로 저장합니다.")

    patients_meta = []

    # index 0~9만 사용 (데모용)
    max_idx = min(10, len(dataset))
    for idx in tqdm(range(max_idx), desc="Saving cache (test[0-9])"):
        data = dataset[idx]        # torch_geometric.data.Data 객체
        # label은 data.y_p 에 들어있음 (shape [1])
        label = int(data.y_p.item())

        # patient_id 를 간단히 test_0, test_1 ... 로 정의
        pid = f"test_{idx}"

        # 그래프 저장
        graph_path = os.path.join(GRAPH_DIR, f"{pid}.pt")
        torch.save(data, graph_path)

        # 메타 정보 (나중에 /patients 에서 쓸 것)
        patients_meta.append(
            {
                "id": pid,
                "label": label,
                "age": None,   # 따로 안 쓰면 일단 None
                "sex": None,
                "icu": None,
            }
        )

    # meta json 저장
    meta_path = os.path.join(CACHE_DIR, "patients_meta.json")
    with open(meta_path, "w") as f:
        json.dump(patients_meta, f, indent=2)

    print(f"✅ {max_idx}개 환자 캐시 저장 완료:")
    print(f"- 그래프: {GRAPH_DIR}")
    print(f"- 메타데이터: {meta_path}")

if __name__ == "__main__":
    main()
