# Commerce POC with DeepSeek-R1-Distill-Llama-70B

로컬 DeepSeek-R1-Distill-Llama-70B 모델을 사용한 상품 비교 POC입니다.

## 기능

1. **상품 검색**: CSV 파일에서 검색어에 맞는 상품을 찾습니다
2. **로컬 LLM 분석**: DeepSeek-R1-Distill-Llama-70B 모델로 상품을 비교하고 장단점을 분석합니다
3. **의도 파악**: "아이 생일케이크와 장난감" 같은 복합 검색어도 이해합니다
4. **추천 제공**: 고객에게 최적의 상품을 추천합니다

## 시스템 요구사항

**⚠️ 중요: DeepSeek-R1-Distill-Llama-70B는 대형 모델입니다**
- **최소 메모리**: 140GB RAM/VRAM (4bit 양자화 사용시)
- **GPU**: NVIDIA GPU 권장 (CUDA 지원), 24GB+ VRAM
- **디스크 공간**: 모델 다운로드용 150GB+

## 설치 및 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 실행 (자동으로 DeepSeek-R1-Distill-Llama-70B 다운로드)
python commerce_poc.py

# 또는 다른 모델 지정
python commerce_poc.py "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
```

## 모델 로딩 과정

첫 실행시 자동으로:
1. Hugging Face에서 DeepSeek-R1-Distill-Llama-70B 모델 다운로드
2. 4bit 양자화를 통한 메모리 최적화
3. GPU/CPU 자동 감지 및 설정

## 오류 처리

충분한 메모리가 없거나 의존성이 누락된 경우 명확한 오류 메시지와 함께 프로그램이 종료됩니다.

## 사용 예시

```
🔍 검색어: 스마트폰 추천
🔍 검색어: 아이 생일케이크
🔍 검색어: 캠핑 장비
```

## 파일 구조

- `commerce_poc.py`: 메인 애플리케이션 (DeepSeek-R1-Distill-Llama-70B 통합)
- `products.csv`: 한국어 상품 데이터 (20개 가상 상품)
- `requirements.txt`: Python 의존성 (PyTorch, Transformers 등)

## 모델 정보

- **모델**: [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
- **크기**: 70B parameters
- **최적화**: 4bit 양자화, Llama 아키텍처
- **언어**: 다국어 지원 (한국어 포함)
- **특징**: DeepSeek R1의 증류 모델로 추론 능력 강화