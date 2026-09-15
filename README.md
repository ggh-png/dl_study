# deeplearning_study

PyTorch로 딥러닝의 기본 구조와 학습 과정을 공부하는 프로젝트입니다.

## 구성

- `notebooks/00_cnn_mnist.ipynb`: 설명과 주석을 추가한 MNIST CNN 실습. 모델 저장 및 불러오기 포함.
- `notebooks/01_cnn_mnist.ipynb`: MNIST CNN 실습 노트북.
- `notebooks/02_rnn_sine.ipynb`: 사인파 시계열 예측 실습 노트북.
- `examples/00_check_environment.py`: PyTorch 실행 환경 확인.
- `examples/01_cnn_mnist.py`: MNIST CNN 학습 스크립트.
- `examples/02_rnn_sine.py`: RNN, GRU, LSTM 시계열 예측 스크립트.

## 실행

Python과 PyTorch, torchvision, NumPy, Matplotlib이 필요합니다.
노트북은 Jupyter 또는 VS Code의 Jupyter 확장에서 실행합니다.

```bash
python examples/00_check_environment.py
python examples/01_cnn_mnist.py --data-dir ./data --output-dir ./outputs/cnn
python examples/02_rnn_sine.py --output-dir ./outputs/rnn
```

노트북은 위에서 아래로 실행합니다. 노트북의 `DATA_DIR`, `OUTPUT_DIR`은
실행 환경에 맞게 설정하세요. 빠르게 확인하려면 `QUICK_RUN = True`를 사용합니다.

## Git 관리

코드와 노트북을 관리하며, 데이터셋과 학습 결과(`outputs/`, `*.pt` 등),
Python 캐시와 가상환경은 `.gitignore`로 제외합니다.
저장된 모델은 Git에 포함되지 않으므로 별도로 보관하거나 학습으로 생성해야 합니다.
