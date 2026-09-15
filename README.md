# dl_study

Jupyter 노트북으로 PyTorch 기반 딥러닝의 기본 구조와 학습 과정을 공부하는 프로젝트입니다.

## 구성

- `notebooks/00_cnn_mnist.ipynb`: 설명과 주석을 추가한 MNIST CNN 실습. 모델 저장 및 불러오기 포함.
- `notebooks/01_rnn_sine.ipynb`: 사인파 시계열 예측 실습 노트북.

## 실행

Python과 PyTorch, torchvision, NumPy, Matplotlib이 필요합니다.
노트북은 Jupyter 또는 VS Code의 Jupyter 확장에서 실행합니다.

노트북은 위에서 아래로 실행합니다. 노트북의 `DATA_DIR`, `OUTPUT_DIR`은
실행 환경에 맞게 설정하세요. 빠르게 확인하려면 `QUICK_RUN = True`를 사용합니다.

## Git 관리

노트북과 학습 문서를 관리하며, 데이터셋과 학습 결과(`outputs/`, `*.pt` 등),
Python 캐시와 가상환경은 `.gitignore`로 제외합니다.
저장된 모델은 Git에 포함되지 않으므로 별도로 보관하거나 학습으로 생성해야 합니다.
