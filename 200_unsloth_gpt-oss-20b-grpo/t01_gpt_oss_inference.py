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
