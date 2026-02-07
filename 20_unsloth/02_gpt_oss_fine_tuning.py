from datasets import load_dataset
from transformers import TextStreamer, BatchEncoding
from unsloth import FastLanguageModel


class GptOss:

    def __init__(self, model_name: str = 'unsloth/gpt-oss-20b', max_seq_length: int = 2000):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            dtype=None,  # torch.bfloat16,  # None for auto detection
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            full_finetuning=False,
            low_cpu_mem_usage=True,
            device_map="cuda"  # Explicitly load to CUDA
        )

    def add_lora_adapters(self, r: int = 8, lora_alpha: int = 16):
        """
        기존 LLM 모델에 LoRA (Low-Rank Adaptation) 어댑터를 추가하여 PEFT (Parameter-Efficient Fine-Tuning) 모델로 변환합니다.

        Arguments:
        ----------
        model : PreTrainedModel
            LoRA 어댑터를 적용할 기본 모델입니다.

        r : int (기본값: 8)
            LoRA의 rank(순위) 값입니다.
            - 저순위 행렬의 차원을 결정합니다.
            - 값이 클수록 모델의 표현력이 증가하지만 메모리 사용량도 증가합니다.
            - 권장 값: 8, 16, 32, 64, 128
            - 일반적인 사용:
              * 8: 가벼운 파인튜닝, 메모리 효율 최우선
              * 16-32: 일반적인 파인튜닝에 적합
              * 64-128: 복잡한 태스크나 고품질 결과가 필요할 때

        target_modules : list[str]
            LoRA를 적용할 모델 내 모듈(레이어) 이름 목록입니다.
            - q_proj, k_proj, v_proj, o_proj: Attention 레이어의 Query, Key, Value, Output projection
            - gate_proj, up_proj, down_proj: FFN(Feed-Forward Network) 레이어

        lora_alpha : int (기본값: 16)
            LoRA 스케일링 파라미터입니다.
            - 실제 적용되는 스케일링 = lora_alpha / r
            - 값이 높을수록 LoRA 레이어의 영향력이 커집니다.
            - 일반적으로 r의 2배를 사용합니다 (r=8이면 lora_alpha=16).

        lora_dropout : float (기본값: 0)
            LoRA 레이어에 적용할 드롭아웃 비율입니다.
            - 0: 드롭아웃 없음 (Unsloth에서 최적화됨, 가장 빠름)
            - 0.05~0.1: 약간의 정규화 효과
            - 과적합 방지가 필요한 경우에만 0보다 큰 값 사용

        bias : str (기본값: "none")
            bias 파라미터 학습 방식을 지정합니다.
            - "none": bias를 학습하지 않음 (Unsloth에서 최적화됨, 권장)
            - "all": 모든 bias를 학습
            - "lora_only": LoRA 레이어의 bias만 학습

        use_gradient_checkpointing : bool | str (기본값: "unsloth")
            Gradient Checkpointing 사용 여부입니다.
            - True: 기본 gradient checkpointing 사용 (메모리 절약)
            - "unsloth": Unsloth 최적화 버전 사용
              * 30% 적은 VRAM 사용
              * 2배 큰 배치 사이즈 가능
              * 매우 긴 컨텍스트 학습에 적합

        random_state : int (기본값: 3407)
            재현성을 위한 난수 시드입니다.
            - 동일한 값을 사용하면 동일한 초기화 결과를 얻을 수 있습니다.
            - 3407은 Andrej Karpathy가 권장하는 "magic number"입니다.

        use_rslora : bool (기본값: False)
            Rank-Stabilized LoRA 사용 여부입니다.
            - True: rsLoRA 활성화 (스케일링 = lora_alpha / sqrt(r))
            - 높은 rank 값에서 더 안정적인 학습을 제공합니다.

        loftq_config : dict | None (기본값: None)
            LoftQ (LoRA-Fine-Tuning-aware Quantization) 설정입니다.
            - None: LoftQ 미사용
            - 양자화된 모델에서 LoRA 학습 시 초기화를 개선합니다.
            - 설정 예시: {"loftq_bits": 4, "loftq_iter": 1}
        """
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", ],
            lora_alpha=lora_alpha,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )

    def create_input(self, messages: list[dict], reasoning_effort: str = 'low') -> BatchEncoding:
        """
        @param reasoning_error: ['low', 'medium', 'high']
        """
        inputs: BatchEncoding = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=reasoning_effort,  # **NEW!** Set reasoning effort to low, medium or high
        ).to("cuda")
        return inputs


    def generate(self,
                 messages: list[dict],
                 max_new_tokens: int = 2048,
                 reasoning_effort: str = 'low'):
        inputs = self.create_input(messages, reasoning_effort)
        _ = self.model.generate(**inputs,
                                max_new_tokens=max_new_tokens,
                                streamer=TextStreamer(self.tokenizer),
                                use_cache=True)
        return _


def main():
    dataset = load_dataset("gretelai/synthetic_text_to_sql")
    dataset

    messages = [
        {'role': 'user', 'content': 'Solve x^5 + 3x^4 - 10 = 3.'}
    ]
    # gptoss = GptOss()
    # gptoss.add_lora_adapters()


if __name__ == '__main__':
    main()
