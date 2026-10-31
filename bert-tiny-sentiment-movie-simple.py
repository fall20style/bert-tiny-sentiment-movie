# bert-tiny-sentiment-movie-simple.py

import os
import gc
import shutil
import torch
from datasets import Dataset, load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

device = "cpu"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

LOCAL_MODEL_DIR = "./local_model_base"
LOCAL_DATASET_DIR = "./local_dataset_cache"
FINAL_REWARD_MODEL_DIR = "./best_korean_reward_model"
CLONED_TXT_PATH = r"C:\Users\fall2\Korpora\nsmc\ratings_train.txt"

# 기존 캐시 초기화
if os.path.exists(LOCAL_DATASET_DIR):
    try: shutil.rmtree(LOCAL_DATASET_DIR)
    except Exception: pass

# 1. 모델 로드
print(f"🔌 [로컬 오프라인 모드] '{LOCAL_MODEL_DIR}'에서 대형 베이스 모델을 로드합니다.")
tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR, num_labels=2, local_files_only=True).to(device)

# 2. 데이터셋 구축
print(f"📂 [데이터셋 구축] 로컬 경로에서 영화 리뷰 파싱을 시작합니다.")
documents = []
labels = []

with open(CLONED_TXT_PATH, "r", encoding="utf-8", errors="ignore") as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            _, doc_text, label_val = parts
            if doc_text.strip():
                documents.append(doc_text)
                labels.append(int(label_val))
        # [핵심 수정] CPU 고속 검증을 위해 샘플 수를 5,000개에서 300개로 대폭 다이어트!
        if len(documents) >= 300: 
            break

print(f"📊 로컬 물리 파싱 완료: 총 {len(documents)}개의 문장이 정상 적재되었습니다.")
raw_dataset = Dataset.from_dict({"document": documents, "label": labels})

def preprocess_function(examples):
    # [핵심 수정] 문장 최대 길이를 128에서 64로 줄여 연산량을 4배 감소시킵니다.
    tokenized = tokenizer(examples["document"], padding="max_length", truncation=True, max_length=64)
    tokenized["labels"] = examples["label"]
    return tokenized

print("🧹 초고속 토크나이징 및 데이터셋 변환 중...")
processed_dataset = raw_dataset.map(preprocess_function, batched=True)
processed_dataset.save_to_disk(LOCAL_DATASET_DIR)

# 3. CPU 맞춤형 초고속 파인튜닝 설정
training_args = TrainingArguments(
    output_dir="./hf_korean_reward_model_checkpoint",
    learning_rate=5e-5,
    per_device_train_batch_size=16,   # 배치를 16으로 키워 총 스텝 수를 줄입니다.
    gradient_accumulation_steps=1,
    num_train_epochs=3,               # 3번 반복 학습
    weight_decay=0.01,
    eval_strategy="no",
    logging_steps=5,                  # 로그를 더 자주 출력
    use_cpu=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
)

print("\n🚀 [고속 프로토타입] 리워드 모델 파인튜닝 학습을 시작합니다...")
print("[-] 연산 압축을 완료했습니다. 약 2~4분 내로 완료됩니다.")
trainer.train()

# 4. 저장
model.save_pretrained(FINAL_REWARD_MODEL_DIR)
tokenizer.save_pretrained(FINAL_REWARD_MODEL_DIR)
gc.collect()

print(f"\n✅ 프로세스 완벽 종료! 채점관 모델이 '{FINAL_REWARD_MODEL_DIR}'에 최종 빌드되었습니다.")
