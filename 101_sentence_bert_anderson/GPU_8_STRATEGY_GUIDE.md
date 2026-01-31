# GPU 8개 활용 Sentence BERT 학습 전략 가이드

## 🚀 개요

Nvidia GPU 8개를 활용하여 Sentence BERT 모델을 효율적으로 학습하기 위한 전략과 구현 방법을 제시합니다.

## 📊 전략 비교

### 1. **Distributed Data Parallel (DDP)** - ⭐ **권장**

#### 장점:
- **최고 성능**: GPU 간 통신 오버헤드 최소화
- **메모리 효율성**: 각 GPU가 독립적인 모델 복사본 유지
- **스케일링**: 선형적 성능 향상
- **안정성**: 개별 프로세스로 인한 높은 안정성

#### 적용 방법:
```bash
python train_sentence_bert.py \
    --gpu_strategy ddp \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --batch_size 64 \
    --epochs 5 \
    --use_amp
```

#### 예상 성능:
- **학습 속도**: 8배 가속 (이론적)
- **실제 가속**: 6-7배 (통신 오버헤드 고려)
- **메모리 사용량**: GPU당 독립적

### 2. **Data Parallel (DP)** - 🔄 **대안**

#### 장점:
- **구현 단순성**: 한 줄 코드로 적용 가능
- **디버깅 용이성**: 단일 프로세스

#### 단점:
- **GIL 병목**: Python GIL로 인한 성능 제약
- **GPU 0 부하**: 마스터 GPU에 집중된 연산
- **메모리 불균형**: GPU 간 메모리 사용량 차이

#### 적용 방법:
```bash
python train_sentence_bert.py \
    --gpu_strategy dp \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --batch_size 32
```

#### 예상 성능:
- **학습 속도**: 4-5배 가속
- **메모리 사용량**: GPU 0에 집중

### 3. **Mixed Precision Training (AMP)** - 🔥 **필수**

#### 장점:
- **메모리 절약**: 50% 메모리 사용량 감소
- **속도 향상**: Tensor Core 활용으로 1.5-2배 가속
- **정확도 유지**: 자동 스케일링으로 수치 안정성

#### 적용:
```python
# 자동으로 활성화됨
--use_amp
```

## 🎯 최적화 권장사항

### 배치 크기 전략

| GPU 전략 | 개별 GPU 배치 크기 | 총 유효 배치 크기 | 권장 이유 |
|---------|-------------------|------------------|-----------|
| DDP     | 64-128            | 512-1024         | 최적 처리량 |
| DP      | 32-64             | 256-512          | 메모리 제약 |

### 학습률 조정

```python
# 유효 배치 크기에 따른 학습률 스케일링
base_lr = 2e-5
effective_batch_size = batch_size * num_gpus
scale_factor = effective_batch_size / 32  # 기본 배치 크기
adjusted_lr = base_lr * scale_factor
```

### 메모리 최적화

1. **Gradient Checkpointing**: 메모리 사용량 감소
2. **Pin Memory**: 데이터 로딩 속도 향상
3. **Non-blocking Transfer**: CPU-GPU 전송 최적화

## 🔧 실행 방법

### 1. DDP 실행 (권장)
```bash
# 실행 스크립트 사용
./run_multi_gpu_training.sh

# 또는 직접 실행
python train_sentence_bert.py \
    --gpu_strategy ddp \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --batch_size 64 \
    --epochs 5 \
    --lr 2e-5 \
    --use_amp \
    --train_samples 100000
```

### 2. DataParallel 실행
```bash
python train_sentence_bert.py \
    --gpu_strategy dp \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --batch_size 32 \
    --epochs 3 \
    --lr 1e-5
```

## 📈 성능 모니터링

### 1. GPU 사용률 확인
```bash
# 실시간 모니터링
watch -n 1 nvidia-smi

# GPU 사용률 로깅
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv --loop=1
```

### 2. TensorBoard 모니터링
```bash
tensorboard --logdir=runs/
```

### 3. 통신 효율성 확인
```bash
# NCCL 디버그 정보
export NCCL_DEBUG=INFO
```

## ⚠️ 주의사항

### 1. 메모리 관리
- **OOM 방지**: 배치 크기 점진적 증가
- **메모리 정리**: `torch.cuda.empty_cache()` 사용

### 2. 동기화 이슈
- **배치 정규화**: `SyncBatchNorm` 사용 권장
- **시드 설정**: 재현 가능한 결과를 위한 시드 고정

### 3. 데이터 로딩
- **Sampler 설정**: `DistributedSampler` 필수
- **Worker 수**: CPU 코어 수에 맞게 조정

## 🎊 기대 효과

### 학습 시간 단축
- **단일 GPU**: 10시간
- **DDP 8 GPU**: 1.5-2시간 (6-7배 가속)
- **DP 8 GPU**: 2-3시간 (4-5배 가속)

### 모델 품질
- **DDP**: 원본과 동일한 품질
- **Mixed Precision**: 미미한 품질 차이 (< 1%)

### 자원 활용도
- **GPU 사용률**: 95%+ (DDP)
- **메모리 효율성**: 50% 절약 (AMP)

## 🔄 트러블슈팅

### 일반적인 문제들

1. **CUDA OOM 에러**
   ```bash
   # 배치 크기 감소
   --batch_size 32
   ```

2. **NCCL 타임아웃**
   ```bash
   export NCCL_TIMEOUT=3600
   ```

3. **포트 충돌**
   ```python
   # 다른 포트 사용
   setup_distributed(rank, world_size, port="12356")
   ```

## 📚 참고 자료

- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)

---

**최종 권장사항**: DDP + Mixed Precision + 적절한 배치 크기 조정으로 최대 7배 학습 속도 향상 달성 가능! 