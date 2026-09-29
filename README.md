# dl_study

Jupyter 노트북으로 PyTorch 기반 딥러닝의 기본 구조와 학습 과정을 공부하는 프로젝트입니다.

## 구성

- `notebooks/00_cnn_mnist.ipynb`: 설명과 주석을 추가한 MNIST CNN 실습. 모델 저장 및 불러오기 포함.
- `notebooks/01_rnn_sine.ipynb`: 사인파 시계열 예측 실습 노트북.
- `notebooks/02_vae_mnist.ipynb`: ResNet 인코더 기반 MNIST VAE. 학습·검증·테스트 분리, KL annealing, 이미지 복원·생성 및 모델 저장/불러오기.
- `notebooks/03_vae_tuning.ipynb`: 한 변수씩 바꾸는 VAE 실험 실행, 지표·복원 이미지 비교 및 최종 모델 불러오기.
- `examples/vae_tuning.py`: 순차 탐색과 전체 데이터 확인용 실행 모듈.
- `reports/vae_tuning_20260929.md`: 실험 조건, 결과, 비교 이미지와 해석 범위.

## 실행

Python과 PyTorch, torchvision, NumPy, Matplotlib이 필요합니다.
노트북은 Jupyter 또는 VS Code의 Jupyter 확장에서 실행합니다.

노트북은 위에서 아래로 실행합니다. 코드 업데이트 후에는 커널을 재시작하고 전체 셀을 다시 실행하세요.
설정 셀을 실행하지 않고 저장 셀부터 실행하면 `ANNEAL_EPOCHS` 등의 변수가 없어 오류가 발생할 수 있습니다. 노트북의 `DATA_DIR`, `OUTPUT_DIR`은
실행 환경에 맞게 설정하세요. 빠르게 확인하려면 `QUICK_RUN = True`를 사용합니다.

## Git 관리

노트북과 학습 문서를 관리하며, 데이터셋과 학습 결과(`outputs/`, `*.pt` 등),
Python 캐시와 가상환경은 `.gitignore`로 제외합니다.
저장된 모델은 Git에 포함되지 않으므로 별도로 보관하거나 학습으로 생성해야 합니다.

## VAE 튜닝

현재 VAE 기본값은 β=2, latent dimension=8, learning rate=1e-3, hidden dimension=800, batch size=32입니다.
전체 데이터에서는 20 epoch 중 처음 10 epoch 동안 β를 0→2로 증가시킵니다.
원본 학습 데이터는 학습 54,000장/검증 6,000장으로 나누고, 테스트 10,000장은 최종 평가에만 사용합니다.
모델 선택에는 목표 β에 도달한 이후의 검증 BCE+KL(고정 가중치 1)을 사용합니다.
2차원 잠재 공간 시각화는 `LATENT_DIM = 2`일 때만 실행됩니다.


`notebooks/03_vae_tuning.ipynb`에서 후보별 결과와 복원 이미지를 비교합니다.
`examples/vae_tuning.py`는 기존 VAE 클래스 정의를 재사용하여 β → latent dimension → learning rate → hidden dimension → batch size → KL annealing 순으로 한 변수씩 비교합니다.

```bash
python examples/vae_tuning.py --output outputs/notebooks/vae/new_search --epochs 10 --train-limit 6000 --val-limit 1000
```

전체 데이터 탐색은 `--train-limit 0 --val-limit 0 --epochs 20`으로 실행합니다.
출력 경로는 새 디렉토리를 지정해야 하며, 결과 CSV·모델·이미지는 `outputs/`에 저장됩니다.
검증 BCE+KL(고정 가중치 1)을 모델 선택 기준으로 사용하고 각 후보의 train/validation total, reconstruction, KL을 따로 기록합니다. 테스트 집합은 탐색에 사용하지 않습니다.

전체 데이터에서 기존 설정과 튜닝 설정을 확인하고 최종 테스트를 수행하려면:

```bash
python examples/vae_tuning.py --confirm-from outputs/notebooks/vae/new_search --output outputs/notebooks/vae/new_confirmation --epochs 20
```

실행한 비교 결과와 한계는 [VAE 튜닝 보고서](reports/vae_tuning_20260929.md)에 정리했습니다.

새로 clone한 환경에는 `outputs/`의 결과가 없습니다. 학습 없이 핵심 결과를 확인하려면 보고서를 읽으세요.
비교 노트북을 실행하려면 위 명령으로 결과를 생성한 뒤 `RESULT_DIR`과 `CONFIRM_DIR`을 해당 경로로 설정하세요.
모델 구조를 변경했다면 기존 탐색 결과를 재사용하지 말고 새 경로에서 탐색부터 다시 실행하세요.
