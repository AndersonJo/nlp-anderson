# Sentence BERT from Scratch

PyTorch로 처음부터 구현한 Sentence BERT 모델입니다. SNLI 데이터셋을 사용하여 문장 쌍 분류 태스크를 학습합니다.

## 프로젝트 구조

```
.
├── sentence_bert_model.py      # Sentence BERT 모델 구현
├── train_sentence_bert.py      # 학습 스크립트
├── inference.py                # 추론 스크립트
├── requirements.txt            # 필요한 패키지들
└── README.md                   # 프로젝트 설명
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 모델 학습

```bash
python train_sentence_bert.py
```

이 명령어는:
- SNLI 데이터셋을 자동으로 다운로드
- Sentence BERT 모델을 학습
- 학습 과정을 시각화
- 모델을 `sentence_bert_snli_model.pth`로 저장

### 2. 추론

```bash
python inference.py
```

학습된 모델을 사용하여 문장 쌍의 관계를 예측합니다.

### 3. 프로그래밍 방식 사용

```python
from inference import SentenceBERTInference

# 모델 로드
model = SentenceBERTInference('sentence_bert_snli_model.pth')

# 문장 쌍 관계 예측
result = model.predict_relationship(
    "A soccer game with multiple males playing.",
    "Some men are playing a sport."
)
print(result['relationship'])  # 'entailment'

# 코사인 유사도 계산
similarity = model.compute_similarity(
    "A man is playing guitar.",
    "A woman is playing guitar."
)
print(similarity['similarity'])  # 0.85

# 단일 문장 인코딩
embedding = model.encode_sentence("Hello world!")
print(embedding.shape)  # (1, 768)
```

## 모델 아키텍처

### SentenceBERT 클래스
- **BERT 인코더**: `bert-base-uncased` 사용
- **Mean Pooling**: attention mask를 고려한 평균 계산
- **Pooling Layer**: 768차원 임베딩을 위한 선형 레이어
- **Classifier**: 3클래스 분류 (entailment, contradiction, neutral)

### 학습 방식
- **Loss**: CrossEntropyLoss
- **Optimizer**: AdamW with weight decay
- **Scheduler**: CosineAnnealingLR
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Max Length**: 128 tokens

## 데이터셋

### SNLI (Stanford Natural Language Inference)
- **목적**: 자연어 추론
- **크기**: 약 570K 문장 쌍
- **레이블**: entailment, contradiction, neutral
- **평가**: 정확도 (Accuracy)

### 예제 데이터
```
Premise: "A soccer game with multiple males playing."
Hypothesis: "Some men are playing a sport."
Label: entailment

Premise: "A black race car starts up in front of a crowd of people."
Hypothesis: "A man is driving down a lonely road."
Label: contradiction
```

## 성능

일반적으로 다음과 같은 성능을 기대할 수 있습니다:
- **Validation Accuracy**: ~85-90%
- **Test Accuracy**: ~85-90%
- **Training Time**: GPU 기준 약 2-3시간 (3 epochs)

## 하이퍼파라미터 튜닝

`train_sentence_bert.py`에서 다음 하이퍼파라미터를 조정할 수 있습니다:

```python
BATCH_SIZE = 16          # 배치 크기
MAX_LENGTH = 128         # 최대 토큰 길이
EPOCHS = 3              # 학습 에포크
LEARNING_RATE = 2e-5    # 학습률
```

## 주의사항

1. **GPU 메모리**: BERT 모델은 큰 메모리를 사용하므로 GPU가 권장됩니다.
2. **데이터 다운로드**: 첫 실행 시 SNLI 데이터셋이 자동으로 다운로드됩니다.
3. **학습 시간**: CPU에서는 매우 오래 걸릴 수 있습니다.

## 확장 가능성

이 구현을 기반으로 다음과 같은 확장이 가능합니다:

1. **다른 데이터셋**: STS, Quora Question Pairs 등
2. **다른 모델**: RoBERTa, DistilBERT 등
3. **다른 태스크**: 문장 유사도, 질문-답변 등
4. **다국어 지원**: 한국어 BERT 모델 사용

## 참고 자료

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [SNLI Dataset](https://nlp.stanford.edu/projects/snli/)
- [Hugging Face Transformers](https://huggingface.co/transformers/) 