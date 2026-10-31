import os
import gc
import shutil
import torch
from datasets import Dataset, load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

# 1. 하드웨어 세팅 및 WandB 비활성화
device = "cpu"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

LOCAL_MODEL_DIR = "./local_model_base"
LOCAL_DATASET_DIR = "./local_dataset_cache"
FINAL_REWARD_MODEL_DIR = "./best_korean_reward_model"
CLONED_TXT_PATH = r"C:\Users\fall2\Korpora\nsmc\ratings_train.txt"

# [업데이트] 데이터 조건이 바뀌었으므로 기존 300개짜리 옛날 캐시 폴더를 강제 리셋합니다.
if os.path.exists(LOCAL_DATASET_DIR):
    try: shutil.rmtree(LOCAL_DATASET_DIR)
    except Exception: pass

# 2. 475MB 대형 로컬 한국어 모델 가중치 로드
print(f"🔌 [로컬 오프라인 모드] '{LOCAL_MODEL_DIR}'에서 한국어 베이스 모델을 로드합니다.")
tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR, num_labels=2, local_files_only=True)
model = model.to(device)


# ==========================================
# 3. [연구 스케일업] 대용량 샘플링 및 컨텍스트 복원
# ==========================================
print(f"📂 [데이터셋 확장 구축] 로컬 경로({CLONED_TXT_PATH})에서 영화 리뷰 파싱을 시작합니다.")
documents = []
labels = []

with open(CLONED_TXT_PATH, "r", encoding="utf-8", errors="ignore") as f:
    header = f.readline()  # 헤더 스킵
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            _, doc_text, label_val = parts
            if doc_text.strip():
                documents.append(doc_text)
                labels.append(int(label_val))
        
        # [밸런스 수정] 일반화 성능 확장을 위해 데이터 수를 300개에서 2,000개로 대폭 상향!
        if len(documents) >= 2000: 
            break

print(f"📊 로컬 물리 파싱 완료: 총 {len(documents)}개의 한국어 문장이 정상 적재되었습니다.")
raw_dataset = Dataset.from_dict({"document": documents, "label": labels})

def preprocess_function(examples):
    # [밸런스 수정] 문장 최대 길이를 64에서 128로 원복하여 더 긴 문맥 패턴을 온전히 흡수합니다.
    tokenized = tokenizer(examples["document"], padding="max_length", truncation=True, max_length=128)
    tokenized["labels"] = examples["label"]
    return tokenized

print("🧹 고해상도 토크나이징 및 데이터셋 직렬화 변환 중...")
processed_dataset = raw_dataset.map(preprocess_function, batched=True)
processed_dataset.save_to_disk(LOCAL_DATASET_DIR)
print(f"💾 2,000개 밸런스 데이터셋 로컬 캐싱 세이브 완료.\n")


# ==========================================
# 4. [골디락스 존] CPU 20~30분 최적 밸런스 훈련 아규먼트
# ==========================================
training_args = TrainingArguments(
    output_dir="./hf_korean_reward_model_checkpoint",
    learning_rate=4e-5,               # 대량 데이터셋에 맞춰 러닝 레이트를 미세 조정하여 안정적 수렴 유도
    per_device_train_batch_size=16,   # 배치를 16으로 키워 16GB RAM의 여유 공간을 쓰고 총 Step 수를 압축
    gradient_accumulation_steps=1,
    num_train_epochs=3,               # 3에포크 반복 복습으로 가치 판단 경계 정밀화
    weight_decay=0.01,
    eval_strategy="no",               # 불필요한 메모리 차감 방지
    logging_steps=10,                 # 로그 출력 주기 조정
    use_cpu=True,                     # 순수 노트북 CPU 코어 훈련
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
)

print("\n🚀 [연구 스케일업] 진짜 한국어 품질 평가 채점관 모델 최종 학습을 시작합니다...")
print("[-] 총 2,000개 문장 복합 연산이 작동합니다. 사양에 따라 약 20~30분 소요됩니다.")
trainer.train()

# 5. 최종 완성 모델 세이브 및 가비지 컬렉션
model.save_pretrained(FINAL_REWARD_MODEL_DIR)
tokenizer.save_pretrained(FINAL_REWARD_MODEL_DIR)
gc.collect()

print(f"\n✅ 프로세스 완벽 종료! 진화된 한국어 리워드 모델 채점관이 '{FINAL_REWARD_MODEL_DIR}'에 빌드되었습니다.")
