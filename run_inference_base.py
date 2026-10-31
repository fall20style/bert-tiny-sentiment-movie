import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# 1. 환경 세팅 및 [학습 안 된 기본 베이스 모델] 로드
device = "cpu"
BASE_MODEL_DIR = "./local_model_base"

if not os.path.exists(BASE_MODEL_DIR):
    print(f"❌ '{BASE_MODEL_DIR}' 폴더를 찾을 수 없습니다.")
    exit(1)

print(f"🔌 [로컬 작동] 학습되지 않은 순수 베이스 모델 가중치({BASE_MODEL_DIR})를 로드합니다...")
tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_DIR, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(BASE_MODEL_DIR, num_labels=2, local_files_only=True).to(device)
model.eval()

# 2. 리워드 스코어 계산 함수 (0.0 ~ 1.0)
def get_reward_score(text_list):
    scores = []
    with torch.no_grad():
        for text in text_list:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = model(**inputs)
            
            # [수정 포인트] 배치 차원([0])에서 1번 인덱스(고품질 확률)만 쏙 뽑아냅니다.
            probs = F.softmax(outputs.logits, dim=-1)[0]
            reward_value = probs[1].item()  # 2개의 요소 중 index 1만 Scalar로 안전하게 변환
            scores.append(reward_value)
    return scores

# ==========================================
# 3. 실전 테스트 진행
# ==========================================
test_ai_outputs = [
    "코드의 가독성을 높이기 위해 리팩토링을 완료했고 완벽하게 작동합니다.",
    "대충 돌아가니까 이대로 빌드하고 제출해 버려야겠습니다.",
    "이 영화는 연출과 스토리가 매우 탄탄해서 깊은 감동을 줍니다.",
    "배우들 연기도 어색하고 돈 아까운 최악의 지루한 영화였습니다."
]

print("\n🔍 [학습 안 된 모델] 새로운 문장들에 대한 리워드 채점 결과:")
print("-" * 60)

scores = get_reward_score(test_ai_outputs)

for text, score in zip(test_ai_outputs, scores):
    status = "🔥 고품질 (자기개선 데이터 채택)" if score >= 0.5 else "❌ 저품질 (필터링 탈락)"
    print(f"[*] 입력 문장 : {text}")
    print(f"[*] 보상 점수 : {score:.4f} -> 판정: {status}\n")
print("-" * 60)
