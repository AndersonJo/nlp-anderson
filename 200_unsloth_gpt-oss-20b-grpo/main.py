from unsloth import FastLanguageModel


def load():
    max_seq_length = 2048
    dtype = None

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='unsloth/gpt-oss-20b',
        dtype=dtype,  # None for auto detection
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
        low_cpu_mem_usage=True
    )


if __name__ == '__main__':
    load()

