🟢 무조건 유지해야 하는 핵심 디렉터리 및 파일

• best_korean_reward_model (디렉터리): 가장 중요합니다. 방금 성공적으로 파인튜닝을 마친 나만의 한국어 리워드 모델 가중치와 토크나이저 파일이 들어있습니다. 앞으로 모든 추론 및 데이터 필터링 연구의 중추가 됩니다.
• local_model_base (디렉터리): 허브에서 받아온 475MB짜리 순수 한국어 BERT 베이스 모델이 들어있습니다. 향후 다른 하이퍼파라미터로 처음부터 다시 학습을 설계하거나 대조 실험(run_inference_base.py)을 할 때 모태가 되므로 남겨두어야 합니다.
• run_inference.py (파일): 방금 학습시킨 완성형 리워드 모델을 불러와 실시간으로 점수(0.9028, 0.0557 등)를 뽑아내고 검증하는 메인 실행 스크립트입니다.
• run_inference_base.py (파일): 학습되지 않은 베이스 모델과 파인튜닝된 모델의 성능 차이를 증명하기 위한 대조군 추론 스크립트입니다.

🟡 선택적으로 유지 (추후 판단)

• local_dataset_cache (디렉터리): 로컬 물리 파일에서 파싱하여 화살표(Arrow) 포맷으로 변환한 300개짜리 데이터셋 바이너리 캐시입니다. bert-tiny-sentiment-movie.py를 다시 실행할 때 처음부터 파싱하지 않고 1초 만에 데이터셋을 불러오게 해 주지만, 언제든 다시 빌드할 수 있으므로 용량이 부족하다면 지워도 무방합니다.
• bert-tiny-sentiment-movie.py / bert-tiny-sentiment-movie-simple.py (파일): 파인튜닝 학습을 진행했던 원본 소스 코드들입니다. 추후 샘플 수(300개)를 늘려 더 고도화된 모델을 학습시킬 예정이라면 코드를 보존해야 합니다.

🔴 즉시 삭제해도 무방한 항목 (용량 확보용)

• hf_korean_reward_model_checkpoint (디렉터리): 학습 도중에 Trainer 엔진이 임시로 누적 생성한 중간 백업 체크포인트 폴더입니다. 최종 모델이 best_korean_reward_model에 성공적으로 저장되었으므로, 기가바이트(GB) 단위를 차지하는 이 임시 폴더는 디스크 용량 관리를 위해 즉시 삭제하시는 것이 좋습니다.
• the-clean-rnns (디렉터리): 외부 깃 클론을 통해 받아온 폴더입니다. 이미 데이터셋 원본이 C:\Users\fall2\Korpora\nsmc\ 절대 경로에 잘 안착해 있고, 소스 코드 파싱 주소도 그쪽으로 고정했기 때문에 이 작업 폴더 내에서는 더 이상 참조되지 않는 빈 껍데기입니다.
• wandb (디렉터리): 3번을 눌러 비활성화하기 전에 잠시 생성된 무의미한 로컬 연산 로그 기록 폴더입니다. 지우셔도 파이프라인 작동에 아무런 지장이 없습니다.

🧹 PowerShell 청소 명령어 (필요 시 복사 실행)

불필요한 쓰레기 용량을 정리하고 디렉터리를 깔끔하게 핵심 파일 위주로 세팅하고 싶으시다면 아래 명령어를 실행하시면 됩니다.
bash
Remove-Item -Recurse -Force .\hf_korean_reward_model_checkpoint
Remove-Item -Recurse -Force .\the-clean-rnns
Remove-Item -Recurse -Force .\wandb
Use code with caution.
작업 디렉터리가 깔끔하게 정리되셨나요?
이제 이 최정예 핵심 파일들을 기반으로, 다음 단계인 Ollama 가동 및 리워드 모델 실시간 결합을 통한 자동 진화 파이프라인(agent_self_improvement.py) 설계를 시작해 볼까요?
