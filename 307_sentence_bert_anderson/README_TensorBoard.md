# Sentence BERT with TensorBoard Logging

Sentence BERT 모델의 학습 과정을 TensorBoard로 모니터링할 수 있도록 구현했습니다.

## 모델 구조

### Classification Objective Function (논문 구현)
```
o = softmax(Wt(u, v, |u − v|))
```

여기서:
- `u`: 첫 번째 문장의 임베딩
- `v`: 두 번째 문장의 임베딩  
- `|u − v|`: 두 임베딩의 절댓값 차이
- `Wt`: 학습 가능한 가중치 행렬 (R^(3n×k))
- `n`: 문장 임베딩 차원
- `k`: 레이블 수 (SNLI: 3개 - entailment, contradiction, neutral)

### 구현된 특성
- **u (embeddings_1)**: 첫 번째 문장의 BERT 임베딩
- **v (embeddings_2)**: 두 번째 문장의 BERT 임베딩
- **|u-v| (diff)**: 두 임베딩의 절댓값 차이
- **Concatenation**: `[u, v, |u-v|]` 형태로 결합
- **Classification**: 3n 차원 → k 클래스 (Cross-entropy loss)

## 주요 기능

### 1. TensorBoard 로깅
- **실시간 학습/검증 손실** 추적
- **정확도** 모니터링
- **학습률** 변화 추적
- **모델 파라미터** 히스토그램
- **배치별 손실** 기록

### 2. 사용 가능한 메트릭
- `Loss/Train_Epoch`: Epoch별 학습 손실
- `Loss/Val_Epoch`: Epoch별 검증 손실
- `Accuracy/Train_Epoch`: Epoch별 학습 정확도
- `Accuracy/Val_Epoch`: Epoch별 검증 정확도
- `Learning_Rate`: 학습률 변화
- `Loss/Train_Batch`: 배치별 학습 손실
- `Parameters/*`: 모델 파라미터 분포

## 설치 및 설정

### 1. 필요한 패키지 설치
```bash
pip install torch transformers datasets tensorboard tqdm matplotlib
```

### 2. TensorBoard 설치 (별도)
```bash
pip install tensorboard
```

## 사용법

### 1. 빠른 시작
```python
# 데모 실행
python tensorboard_example.py
```

### 2. 학습 스크립트 생성 및 실행

```python
# 학습 스크립트 생성
from model import create_training_script

create_training_script()

# 학습 실행 (TensorBoard 자동 시작 - 기본값)
python
train_sentence_bert.py - -epochs
5 - -batch_size
16

# TensorBoard 비활성화
python
train_sentence_bert.py - -epochs
5 - -batch_size
16 - -no_tensorboard
```

### 3. 수동으로 TensorBoard 시작

```python
from model import start_tensorboard

# 기본 설정으로 시작
start_tensorboard()

# 커스텀 설정으로 시작
start_tensorboard(log_dir="runs", port=6006)
```

### 4. 명령줄에서 TensorBoard 시작
```bash
tensorboard --logdir=runs --port=6006
```

## 파일 구조

```
306 Text Similarity Without Training/
├── sentence_bert_model.py      # 메인 모델 및 TensorBoard 로깅
├── tensorboard_example.py      # 사용 예제
├── train_sentence_bert.py      # 자동 생성되는 학습 스크립트
├── README_TensorBoard.md       # 이 파일
└── runs/                       # TensorBoard 로그 디렉토리
    ├── sentence_bert_20241201_143022/
    ├── demo_run/
    └── ...
```

## TensorBoard 대시보드

### 1. SCALARS 탭
- 학습/검증 손실 그래프
- 정확도 변화 추이
- 학습률 스케줄링 확인

### 2. HISTOGRAMS 탭
- 모델 파라미터 분포
- 가중치 변화 추적

### 3. IMAGES 탭 (향후 확장 가능)
- 임베딩 시각화
- 어텐션 맵

## 주요 함수

### `train_sentence_bert()`
```python
train_losses, val_losses, val_accuracies = train_sentence_bert(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=3,
    lr=2e-5,
    log_dir="runs/my_experiment"  # TensorBoard 로그 디렉토리
)
```

### `start_tensorboard()`
```python
# TensorBoard 서버 시작 및 브라우저 자동 열기
start_tensorboard(log_dir="runs", port=6006)
```

### `list_tensorboard_runs()`
```python
# 사용 가능한 TensorBoard 실행 목록 확인
list_tensorboard_runs("runs")
```

## 로그 디렉토리 구조

```
runs/
├── sentence_bert_20241201_143022/
│   ├── events.out.tfevents.1701415822.hostname
│   └── ...
├── demo_run/
│   ├── events.out.tfevents.1701415900.hostname
│   └── ...
└── ...
```

## 문제 해결

### 1. TensorBoard가 시작되지 않는 경우
```bash
# 포트 확인
lsof -i :6006

# 다른 포트 사용
tensorboard --logdir=runs --port=6007
```

### 2. 로그가 보이지 않는 경우
```python
# 로그 디렉토리 확인
list_tensorboard_runs()

# 이벤트 파일 존재 확인
import os
print(os.listdir("runs"))
```

### 3. 메모리 부족 문제
```python
# 배치 크기 줄이기
train_loader = DataLoader(dataset, batch_size=8)  # 16 → 8

# 데이터셋 크기 줄이기
dataset = load_dataset('snli', split='train[:5000]')  # 10000 → 5000
```

## 고급 사용법

### 1. 커스텀 메트릭 추가
```python
# train_sentence_bert 함수 내에서
writer.add_scalar('Custom/Metric', custom_value, epoch)
```

### 2. 모델 체크포인트 저장
```python
# 학습 중 모델 저장
if val_accuracy > best_accuracy:
    torch.save(model.state_dict(), f"{log_dir}/best_model.pth")
    writer.add_text('Model/Checkpoint', f"Best model saved at epoch {epoch}")
```

### 3. 하이퍼파라미터 로깅
```python
# 하이퍼파라미터 기록
hparams = {
    'learning_rate': lr,
    'batch_size': batch_size,
    'epochs': epochs
}
writer.add_hparams(hparams, {'hparam/accuracy': val_accuracy})
```

## 참고 자료

- [PyTorch TensorBoard Tutorial](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)
- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)
- [SNLI Dataset](https://nlp.stanford.edu/projects/snli/) 