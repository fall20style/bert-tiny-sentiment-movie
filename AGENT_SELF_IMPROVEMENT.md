# 🤖 로컬 LLM 에이전트 자립형 데이터 합성 및 리워드 필터 파이프라인 (AGENT_SELF_IMPROVEMENT.md)

본 문서는 사양이 제한된 16GB RAM CPU 노트북 환경에서 외부 상용 API(OpenAI 등)를 일절 호출하지 않고, 로컬 생성형 모델과 자체 파인튜닝된 고성능 리워드 모델을 결합하여 양질의 한국어 합성 데이터셋을 무한 자동 수집하는 **재귀적 자기개선(Recursive Self-Improvement) 코어 파이프라인**의 기술 명세입니다.

---

## 1. 파이프라인 아키텍처 및 메커니즘 (Pipeline Workflow)

인터넷 연결이 완벽하게 차단된 오프라인 환경에서도 단 1초의 병목 없이 동적으로 작동하는 선순환 루프 구조입니다.

```text
[시드 프롬프트 주기적 공급]
        │
        ▼
[Ollama: deepseek-r1:1.5b] ──(추론 및 생성)──> [원본 출력 도출]
                                                     │
                                                     ▼
                                          [<think> 사고 태그 자동 분리]
                                                     │
                                                     ▼
[골드 데이터셋 영구 적재] <──(합격: >= 0.85)── [로컬 한국어 BERT 채점관]
(.jsonl 파일 무한 축적)                                (F.softmax 확률 반환)
                                                     │
                                                (탈락: < 0.85)
                                                     ▼
                                              [시스템 자동 폐기]
```

---

## 2. 세부 컴포넌트 기능 명세

### ① 로컬 오프라인 모델명 자동 매핑 (Ollama Auto-Discovery)
- 하드코딩된 모델 주소 실패(404 에러)를 방지하기 위해, 백그라운드에 구동 중인 Ollama 커널 엔드포인트(`/api/tags`)를 동적 모니터링합니다.
- 내 PC 디스크에 실제 설치된 모델 목록을 파싱하여 `deepseek` 계열의 최적 소형 모델(예: `deepseek-r1:1.5b`)을 자동으로 추적·지정함으로써 시스템 연결 강건성(Robustness)을 보장합니다.

### ② 추론 사고 과정 후처리 디바인딩 (Post-Processing)
- DeepSeek-R1 추론 모델 특유의 사유 체계인 `<think>...</think>` 태그 영역을 파이썬 문자열 슬라이싱 가속 알고리즘으로 분리해 냅니다.
- 최종 정제된 문장 알맹이만 필터 커널로 전달하여 리워드 스코어 연산의 왜곡을 방지합니다.

### ③ 2차원 임베딩 텐서 스칼라 분리 (`probs.item()`)
- 개조된 한국어 BERT 분류기(`BertForSequenceClassification`)가 반환하는 2차원의 확률 벡터 `[저품질 확률, 고품질 확률]` 구조에서 **1번 인덱스(고품질/긍정 확률)만 정확하게 타깃팅하여 단일 실수(Scalar) 값으로 변환**합니다.
- 이를 통해 파이토치 행렬 연산 도중 발생할 수 있는 `RuntimeError: a Tensor with 2 elements cannot be converted to Scalar` 예외를 원천 차단합니다.

---

## 3. 핵심 하이퍼파라미터 및 가이드라인

- **`REWARD_THRESHOLD = 0.85` (합격 한계선)**: AI가 임의 합성한 문장이 이 임계점을 넘어야만 합격작으로 인정됩니다. 본 연구에서는 2,000개 샘플로 고밀도 훈련된 BERT가 `0.90~0.99`점의 선명한 확신도를 보여주므로, `0.85`선은 어색한 번역체 한글이나 영어가 혼입되는 것을 완벽하게 방어하는 안전한 가치 경계선입니다.
- **`NUM_GENERATIONS` (합성 목표 사이클 수)**: 한 번의 명령으로 반복 구동할 총 사이클 수입니다. `seed_prompts` 리스트 내부의 다양한 질문 도메인을 나머지 연산자(`%`) 구조로 무한 뱅글뱅글 순환시키며 대량 수집을 유도합니다.

---

## 4. 자원 방어 및 안정성 최적화 (Resource Management)

- **`requests.post(..., timeout=45)`**: CPU 기반 언어 모델 추론 특성상 가끔 하드웨어 연산 병목으로 스루풋(Throughput)이 멈출 수 있습니다. 45초 인터럽트 타임아웃을 강제하여 무한 대기 상태(Deadlock)에 빠지는 현상을 막았습니다.
- **`torch.no_grad()` & 드롭아웃 해제**: 채점 프로세스 가동 시 미분 그래디언트 그래프 메모리 적재를 완전 봉쇄하여 16GB RAM의 잔여 자원을 상시 쾌적하게 보존합니다.

---

## 🚀 파이프라인 가동 및 결과 검증 방법

1. **자동 생성 및 실시간 여과 엔진 실행**
   ```powershell
   python .\agent_self_improvement.py
   ```
2. **누적된 수집 데이터 실시간 확인 및 카운트**
   ```powershell
   # 현재까지 자동으로 확보된 골드 데이터셋 총 라인 수 조회
   (Get-Content .\golden_self_improvement_dataset.jsonl).Length
   ```
3. **연구 자산 아카이빙 (Git Commit)**
   ```powershell
   git add .\agent_self_improvement.py
   git commit -m "feat: Ollama API 자동 탐색 및 텐서 차원 분리가 완비된 자동 진화 파이프라인 스크립트 확보"
   ```
