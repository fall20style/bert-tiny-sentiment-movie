import os
import gc
import json
import requests
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# 1. 환경 설정 및 동적 탐색 엔드포인트 정의
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GEN_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

REWARD_MODEL_DIR = "./best_korean_reward_model"
GOLDEN_DATASET_PATH = "./golden_self_improvement_dataset.jsonl"

REWARD_THRESHOLD = 0.85
NUM_GENERATIONS = 100

device = "cpu"

# ==========================================
# 로컬에 깔린 실제 Ollama 모델명 자동 탐색
# ==========================================
OLLAMA_MODEL = ""
try:
    print("🔍 내 노트북에 설치된 Ollama 로컬 모델 리스트를 탐색합니다...")
    tag_response = requests.get(OLLAMA_TAGS_URL, timeout=5)
    if tag_response.status_code == 200:
        models_info = tag_response.json().get("models", [])
        installed_models = [m.get("name") for m in models_info]
        print(f"📦 발견된 로컬 모델 목록: {installed_models}")
        
        if installed_models:
            # 1순위: deepseek가 포함된 모델 자동 탐색
            ds_models = [m for m in installed_models if "deepseek" in m.lower()]
            if ds_models:
                OLLAMA_MODEL = ds_models[0]  # 리스트가 아닌 문자열로 명확히 추출
            else:
                OLLAMA_MODEL = installed_models[0]
            print(f"🎯 [자동 매핑 성공] 학습 데이터 생성에 사용될 타깃 모델: '{OLLAMA_MODEL}'")
except Exception as e:
    print(f"⚠️ 모델 목록 탐색 실패 (Ollama 상태 확인 필요): {e}")

if not OLLAMA_MODEL:
    OLLAMA_MODEL = "deepseek-r1:1.5b"  # 대비용 폴백 모델명 고정

# 2. 리워드 모델 로드
print(f"\n🔌 [파이프라인 부팅] '{REWARD_MODEL_DIR}'에서 완성된 채점관 가중치를 불러옵니다.")
tokenizer = BertTokenizer.from_pretrained(REWARD_MODEL_DIR, local_files_only=True)
reward_model = BertForSequenceClassification.from_pretrained(REWARD_MODEL_DIR, num_labels=2, local_files_only=True).to(device)
reward_model.eval()

def evaluate_reward(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        outputs = reward_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        # [핵심 수정] 2가지 요소 중 index 1(고품질/긍정 확률)만 정확하게 슬라이싱해서 Scalar로 변환합니다!
        return probs[0][1].item()

# 3. 데이터 합성용 시드 프롬프트
seed_prompts = [
    "소프트웨어 개발 프로젝트에서 코드 리뷰가 중요한 이유를 친절하게 설명해줘.",
    "이 영화는 연출과 스토리가 너무 완벽해서 감동적이었다는 내용으로 한 줄 써줘.",
    "알고리즘을 최적화하여 연산 속도를 대폭 개선해낸 훌륭한 엔지니어의 경험담을 작성해줘."
]

print(f"\n🚀 완전 자동화 데이터 합성 및 실시간 리워드 필터링 시작...")
print("-" * 75)

accepted_count = 0

for i in range(NUM_GENERATIONS):
    current_prompt = seed_prompts[i % len(seed_prompts)]
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{current_prompt}\n조건: 딱 한 문장의 한국어로만 최종 답변을 작성해줘.",
        "stream": False,
        "options": {"temperature": 0.7}
    }
    
    print(f"[{i+1}/{NUM_GENERATIONS}] 로컬 LLM 에이전트('{OLLAMA_MODEL}')에 텍스트 생성 요청 중...")
    
    try:
        response = requests.post(OLLAMA_GEN_URL, json=payload, timeout=45)
        
        if response.status_code != 200:
            print(f" ⚠️ 응답 실패! 상태 코드: {response.status_code}")
            continue
            
        raw_response = response.json().get("response", "")
        
        # DeepSeek-R1 특유의 생각 태그 분리 처리
        if "</think>" in raw_response:
            clean_output = raw_response.split("</think>")[-1].strip()
        else:
            clean_output = raw_response.strip()
            
        if not clean_output:
            continue
            
        # 117분 동안 정교하게 학습시킨 내 고성능 한국어 리워드 모델 채점 시작
        score = evaluate_reward(clean_output)
        status = "🔥 합격 (골드 데이터셋 저장)" if score >= REWARD_THRESHOLD else "❌ 불합격 (필터링 탈락)"
        
        print(f" ├─ 에이전트 답변: {clean_output}")
        print(f" ├─ 리워드 점수: {score:.4f}")
        print(f" └─ 최종 판정: {status}\n")
        
        if score >= REWARD_THRESHOLD:
            accepted_count += 1
            record = {"prompt": current_prompt, "synthetic_text": clean_output, "reward_score": score}
            with open(GOLDEN_DATASET_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
    except Exception as e:
        print(f" ❌ 오류 발생: {e}")

gc.collect()
print("-" * 75)
print(f"✅ 파이프라인 세션 최종 완료! 새로 수집된 우수 데이터: {accepted_count}건")
