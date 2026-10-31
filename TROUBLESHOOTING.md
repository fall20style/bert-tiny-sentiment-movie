# 🛠️ 한국어 리워드 모델 CPU 파인튜닝 트러블슈팅 가이드 (Troubleshooting)

본 문서에는 16GB RAM CPU 노트북 환경 및 외부 네트워크 차단 환경에서 발생한 의존성 패키지 충돌, Hugging Face API 차단, 하드웨어 연산 병목 현상을 해결한 핵심 트러블슈팅 과정을 기록합니다.

---

## 1. 토크나이저 인스턴스화 실패 에러
### ❌ 문제 상황
`AutoTokenizer.from_pretrained` 구동 시 필수 텍스트 처리 백엔드가 누락되어 예외 발생 및 스크립트 강제 종료.
```text
ValueError: Couldn't instantiate the backend tokenizer from one of:
You need to have sentencepiece or tiktoken installed to convert a slow tokenizer to a fast one.
```
### 💡 원인 분석
Hugging Face의 최신 `transformers` 엔진은 연산 가속을 위해 Rust 기반의 `FastTokenizer` 변환을 자동 시도하나, 구형 메타데이터 포맷 모델이거나 로컬 백엔드 가속 모듈(`sentencepiece`, `tiktoken`)이 누락되었을 때 발생합니다.
### ✨ 해결 방법
1. 로컬 환경에 누락된 의존성 패키지를 즉시 설치하여 우회:
   ```bash
   pip install sentencepiece tiktoken
   ```
2. 자동 변환 팩토리(`AutoTokenizer`) 클래스를 사용하는 대신, 내부 백엔드 예외를 원천 차단하는 순수 BERT 전용 단일 클래스(`BertTokenizer`)로 코드를 고정 매핑하여 예외를 완벽 분리함.

---

## 2. AutoFactory 설정 매핑 불일치 에러
### ❌ 문제 상황
토크나이저 우회 후, 시퀀스 분류 헤드를 부착하는 과정에서 설정 자산 매핑 예외 발생.
```text
File ".../transformers/models/auto/auto_factory.py", line 345, in from_pretrained
    config, kwargs = AutoConfig.from_pretrained(...)
ValueError: ...
```
### 💡 원인 분석
`prajjwal1/bert-tiny` 혹은 일부 로컬 베이스 모델은 순수 사전학습 가중치(`BertModel`) 상태로만 구성되어 있어, 범용적인 `AutoModelForSequenceClassification` 팩토리가 다이렉트로 인덱스를 해석하고 이진 분류 헤드(`num_labels=2`)를 사전에 부착하지 못해 터진 현상입니다.
### ✨ 해결 방법
제네릭 팩토리 아키텍처를 과감히 제거하고, 파이토치 단에서 새로운 고유 분류 레이어(Dense Linear Layer)를 강제로 탑재해 주는 **`BertForSequenceClassification`** 전용 아키텍처 클래스를 명시적으로 바인딩하여 해결함.

---

## 3. Hugging Face 원격 스크립트 및 데이터 허브 차단
### ❌ 문제 상황
온라인 허브에서 한국어 오픈소스 데이터셋을 긁어오려고 시도할 때 차단 정책 발동으로 학습 진입 불가.
```text
RuntimeError: Dataset scripts are no longer supported, but found nsmc.py
Datasets...DatasetNotFoundError: Dataset '...' doesn't exist on the Hub or cannot be accessed.
```
### 💡 원인 분석
1. Hugging Face 라이브러리가 보안 및 유지보수 정책을 강화하면서 원격 `.py` 로딩 스크립트 가동을 차단함.
2. 로컬 네트워크의 일시적 소켓 장애, 방화벽 차단 또는 SSL 인증서 문제로 원격 허브 서버 접근 자체가 완벽 차단됨.
### ✨ 해결 방법
인터넷 소켓 커넥션(`load_dataset`)을 일절 호출하지 않는 **완전 자립형 로컬 파이프라인**으로 전환. 내 PC 디스크 절대 경로(`C:\Users\fall2\Korpora\nsmc\ratings_train.txt`)에 이미 적재되어 있는 물리 텍스트 소스를 파이썬 내장 `open()` 함수로 직접 끄집어내어 Arrow 데이터 구조로 빌드함으로써 허브 에러 확률을 0%로 무력화함.

---

## 4. `TrainingArguments` 최신 문법 호환성 오류
### ❌ 문제 상황
학습 인수 설정 단계에서 예기치 못한 키워드 인자 컴파일 에러 발생.
```text
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'
```
### 💡 원인 분석
Hugging Face `transformers` 버전이 업그레이드되면서, 기존 레거시 파라미터명이었던 `evaluation_strategy`가 내부 규격에서 완전히 퇴출 및 폐기되었습니다.
### ✨ 해결 방법
최신 표준 규격 키워드 명칭인 **`eval_strategy="no"`**로 문법을 마이그레이션하여 오동작을 수정함.

---

## 5. 하드웨어 스펙 한계로 인한 극단적 연산 병목 (최대 위기)
### ❌ 문제 상황
대형 한국어 모델(475MB) 로드 후 훈련 루프 가동 시 예상 학습 소요 시간이 폭발적으로 늘어남.
```text
 35%|██████▋                                 | 663/1875 [2:01:11<20:07:08, 59.76s/it]
 ValueError: num_samples should be a positive integer value, but got num_samples=0
```
### 💡 원인 분석
1. 1차 시도 시 잘못된 경로 매핑으로 0개 문장이 잡혀 `RandomSampler` 연산 오류 발생.
2. 2차 시도 시 5,000개 샘플 및 128토큰 설정을 475MB 대형 모델과 엮어 순수 CPU 코어로만 밀어붙이니 이터레이션당 1분(`59.76s/it`)이 소요되어 총 20시간 이상이 강제되는 병목 상태 유발.
### ✨ 해결 방법
본 연구의 목적은 대규모 상용 모델 빌드가 아닌 **자기개선 파이프라인 프로토타입 대조 검증**이므로 연산 다이어트(Compression) 단행:
- 원천 파싱 샘플 수를 5,000개 ➡️ **300개**로 대폭 최적화.
- 문장 맥스 길이를 128토큰 ➡️ **64토큰**으로 줄여 행렬 연산 복잡도를 4배 감소.
- 배치 사이즈를 16으로 확장하여 총 가중치 업데이트 횟수를 **57스텝**으로 고속 압축.
- ➡️ 최종 **9분 19초 만에 성공적으로 완주**하였으며, `Loss`를 `0.73`에서 `0.19`까지 완벽하게 하향 안정 수렴시킴.

---

## 6. 학습 전후 보상 점수 뭉침 현상 극복 완료
### ❌ 결과 대조
- **학습 전 (Base Model)**: 모든 한국어 문맥 점수가 `0.32 ~ 0.54` 사이에 무작위로 수렴하여 판정 불가.
- **학습 후 (Tuned Model)**: 촘촘한 가중치 업데이트 분배 수정을 통해 고품질 코드/감동 문장(**`0.90 ~ 0.97`**), 저품질/최악 문장(**`0.05 ~ 0.35`**)으로 양극단 가치 판단 기준 확립 완료.
