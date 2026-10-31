import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# 1. 환경 세팅 및 학습 완료된 모델 로드
device = "cpu"
FINAL_REWARD_MODEL_DIR = "./best_korean_reward_model"

if not os.path.exists(FINAL_REWARD_MODEL_DIR):
    print(f"❌ '{FINAL_REWARD_MODEL_DIR}' 폴더를 찾을 수 없습니다. 학습을 먼저 진행해 주세요.")
    exit(1)

print(f"🔌 [로컬 작동] 방금 학습한 리워드 가중치({FINAL_REWARD_MODEL_DIR})를 로드합니다...")
tokenizer = BertTokenizer.from_pretrained(FINAL_REWARD_MODEL_DIR, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(FINAL_REWARD_MODEL_DIR, local_files_only=True).to(device)
model.eval()  # 추론 모드 고정 (메모리 절약)

# 2. 리워드 스코어 계산 함수 (0.0 ~ 1.0)
def get_reward_score(text_list):
    scores = []
    with torch.no_grad():  # 연산 가속화 및 RAM 보호
        for text in text_list:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = model(**inputs)
            
            # 소프트맥스를 적용하여 각 분류(0:부정, 1:긍정) 확률 추출
            probs = F.softmax(outputs.logits, dim=-1)
            # 인덱스 1(긍정/고품질)의 확률을 최종 리워드 스코어로 채택
            reward_value = probs[0][1].item()
            scores.append(reward_value)
    return scores

# ==========================================
# 3. 실전 테스트 진행
# ==========================================
# 연구 중인 AI 에이전트(Ollama 등)가 새로 생성해 낸 문장 시뮬레이션
test_ai_outputs = [
    "코드의 가독성을 높이기 위해 리팩토링을 완료했고 완벽하게 작동합니다.", # 고품질 후보
    "대충 돌아가니까 이대로 빌드하고 제출해 버려야겠습니다.",                # 저품질 후보
    "이 영화는 연출과 스토리가 매우 탄탄해서 깊은 감동을 줍니다.",          # 긍정 소스
    "배우들 연기도 어색하고 돈 아까운 최악의 지루한 영화였습니다."           # 부정 소스
]

print("\n🔍 새로운 문장들에 대한 리워드 채점 결과:")
print("-" * 60)

scores = get_reward_score(test_ai_outputs)

for text, score in zip(test_ai_outputs, scores):
    # 점수 기준 0.5점을 넘기면 우수 데이터로 채택하는 가이드라인 기재
    status = "🔥 고품질 (자기개선 데이터 채택)" if score >= 0.5 else "❌ 저품질 (필터링 탈락)"
    print(f"[*] 입력 문장 : {text}")
    print(f"[*] 보상 점수 : {score:.4f} -> 판정: {status}\n")
print("-" * 60)

