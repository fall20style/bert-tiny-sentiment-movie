import os
import gc
import shutil
import torch
from datasets import Dataset, load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

# 1. 하드웨어 환경 및 WandB 무조건 차단
device = "cpu"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"  # 선택창(1,2,3번) 팝업 원천 차단

LOCAL_MODEL_DIR = "./local_model_base"
LOCAL_DATASET_DIR = "./local_dataset_cache"
FINAL_REWARD_MODEL_DIR = "./best_korean_reward_model"
CLONED_TXT_PATH = r"C:\Users\fall2\Korpora\nsmc\ratings_train.txt"

# 덜 학습되었거나 꼬여있는 이전 가짜 결과물/캐시 폴더가 있다면 완벽하게 리셋
if os.path.exists(LOCAL_DATASET_DIR):
    try: shutil.rmtree(LOCAL_DATASET_DIR)
    except Exception: pass
if os.path.exists(FINAL_REWARD_MODEL_DIR):
    try: shutil.rmtree(FINAL_REWARD_MODEL_DIR)
    except Exception: pass

# ==========================================
# 2. [MODEL] 475MB 대형 로컬 한국어 모델 가중치 다이렉트 로드
# ==========================================
print(f"🔌 [로컬 오프라인 모드] '{LOCAL_MODEL_DIR}'에서 한국어 전용 대형 베이스 모델을 로드합니다.")
tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR, num_labels=2, local_files_only=True)
model = model.to(device)


# ==========================================
# 3. [DATASET] 내 PC 절대 경로 텍스트 파일 파싱 및 빌드
# ==========================================
print(f"📂 [데이터셋 구축] 로컬 경로({CLONED_TXT_PATH})에서 영화 리뷰 파싱을 시작합니다.")
documents = []
labels = []

with open(CLONED_TXT_PATH, "r", encoding="utf-8", errors="ignore") as f:
    header = f.readline()  # 첫 줄 헤더(id\tdocument\tlabel) 건너뛰기
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            _, doc_text, label_val = parts
            if doc_text.strip():  # 빈 문장 유효성 필터
                documents.append(doc_text)
                labels.append(int(label_val))
        if len(documents) >= 5000:  # CPU 오버헤드 제어를 위한 고속 실험용 상한선
            break

print(f"📊 로컬 물리 파싱 완료: 총 {len(documents)}개의 한국어 문장이 정상 적재되었습니다.")
raw_dataset = Dataset.from_dict({"document": documents, "label": labels})

def preprocess_function(examples):
    tokenized = tokenizer(examples["document"], padding="max_length", truncation=True, max_length=128)
    tokenized["labels"] = examples["label"]
    return tokenized

print("🧹 한국어 전용 토크나이징 및 데이터셋 직렬화 변환 중...")
processed_dataset = raw_dataset.map(preprocess_function, batched=True)
processed_dataset.save_to_disk(LOCAL_DATASET_DIR)
print(f"💾 가공 완료 데이터셋 복사본 저장 완료.\n")


# ==========================================
# 4. [고도화] 한국어 대형 분류를 위한 가중치 업데이트 극대화 설정
# ==========================================
training_args = TrainingArguments(
    output_dir="./hf_korean_reward_model_checkpoint",
    learning_rate=5e-5,               # 새로 얹은 채점 분류 레이어가 빠르게 적응하도록 학습률 상향
    per_device_train_batch_size=8,    # 475MB 체급 모델을 RAM 16GB CPU 환경에서 버틸 수 있는 최적 배치
    gradient_accumulation_steps=1,   # 누적 배치를 1로 변경하여 가중치 갱신 횟수(Step)를 4배 더 촘촘하게 적용!
    num_train_epochs=3,               # 완벽한 가치 판단 경계선 구축을 위해 3에포크로 고밀도 훈련
    weight_decay=0.01,
    eval_strategy="no",               # 평가 오버헤드를 생략하여 CPU 속도 올인
    logging_steps=50,
    use_cpu=True,                     # 랩톱 CPU 코어 연산 제어
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
)

print("\n🚀 고성능 한국어 리워드 모델 파인튜닝 학습을 시작합니다...")
print("[-] 모델 체급이 커졌으므로 사양에 따라 약 8~12분 정도 소요될 수 있습니다. 랩톱 충전기를 연결해 주세요.")
trainer.train()

# 5. 최종 완성 모델 디스크 안전 소장 및 메모리 반환
model.save_pretrained(FINAL_REWARD_MODEL_DIR)
tokenizer.save_pretrained(FINAL_REWARD_MODEL_DIR)
gc.collect()

print(f"\n✅ 프로세스 완벽 종료! 진짜 한국어 품질 평가 채점관 모델이 '{FINAL_REWARD_MODEL_DIR}'에 최종 빌드되었습니다.")
