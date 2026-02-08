"""
해당 파일 핵심은 gpt-oss-20b 실행시키고, inference 하는 것
이때 reasoning_error 설정도 할 수 있음

- unsloth 가 transformers 보다 더 먼저 import 되야 함
"""

from unsloth import FastLanguageModel
from transformers import TextStreamer, BatchEncoding


class GptOss:

    def __init__(self, model_name: str = 'unsloth/gpt-oss-20b-BF16', max_seq_length: int = 2000):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            dtype="bfloat16",
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            full_finetuning=False,
            low_cpu_mem_usage=True,
            device_map="cuda"  # Explicitly load to CUDA
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
    messages = [
        {'role': 'user', 'content': 'Solve x^5 + 3x^4 - 10 = 3.'}
    ]
    gptoss = GptOss()
    # gptoss.generate(messages, reasoning_effort='low')
    gptoss.generate(messages, reasoning_effort='medium')
    # gptoss.generate(messages, reasoning_effort='high')


if __name__ == '__main__':
    main()
