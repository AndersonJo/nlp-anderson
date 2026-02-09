"""
Main GptOss model class.

Design Pattern:
- Facade: Simple API hiding Unsloth/PEFT complexity
- Explicit State: Clear mode switching (train/inference)
- Composition: Uses ChatFormatter for prompt handling

Usage:
    from gpt_oss import GptOssModel, ModelConfig, LoRAConfig
    
    # Initialize with LoRA
    model = GptOssModel(
        model_config=ModelConfig(),
        lora_config=LoRAConfig(r=16, lora_alpha=32)
    )
    
    # Training workflow
    model.set_mode("train")
    # ... use model.model with SFTTrainer
    
    # Inference workflow
    model.set_mode("inference")
    output = model.generate(
        system="You are an SQL expert...",
        user="Schema: ... Question: ..."
    )
"""
from unsloth import FastLanguageModel

print('unsloth should be import firstly')

from typing import Literal, TYPE_CHECKING

from transformers import TextStreamer, BatchEncoding

from .config import ModelConfig, LoRAConfig, GenerationConfig
from .formatter import ChatFormatter

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


class GptOssModel:
    """
    High-level interface for GPT-OSS with LoRA fine-tuning support.
    
    Attributes:
        model: The underlying HuggingFace/PEFT model
        tokenizer: Associated tokenizer
        formatter: ChatFormatter instance for prompt handling
        model_config: Model configuration
        lora_config: LoRA configuration (None if no LoRA applied)
    """

    def __init__(
            self,
            model_config: ModelConfig | None = None,
            lora_config: LoRAConfig | None = None
    ):
        """
        Initialize model with optional LoRA adapters.
        
        Args:
            model_config: Base model settings (uses defaults if None)
            lora_config: LoRA settings (skips LoRA if None)
        """
        self.model_config = model_config or ModelConfig()
        self.lora_config = lora_config
        self._mode: Literal["train", "inference"] = "inference"

        # Load base model
        self.model, self.tokenizer = self._load_model()
        self.formatter = ChatFormatter(self.tokenizer)

        # Apply LoRA if config provided
        if lora_config is not None:
            self._apply_lora()

    def _load_model(self) -> tuple["PreTrainedModel", "PreTrainedTokenizer"]:
        """Load the base model from HuggingFace/Unsloth."""
        cfg = self.model_config
        return FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            dtype=cfg.dtype,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            full_finetuning=cfg.full_finetuning,
            low_cpu_mem_usage=cfg.low_cpu_mem_usage,
            device_map=cfg.device_map
        )

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to the model."""
        if self.lora_config is None:
            raise ValueError("LoRA config not provided")

        cfg = self.lora_config
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=cfg.r,
            target_modules=list(cfg.target_modules),
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.bias,
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
            random_state=cfg.random_state,
            use_rslora=cfg.use_rslora,
            loftq_config=cfg.loftq_config,
        )

    @property
    def mode(self) -> Literal["train", "inference"]:
        """Current model mode (train or inference)."""
        return self._mode

    @property
    def max_seq_length(self) -> int:
        """Maximum sequence length from model config."""
        return self.model_config.max_seq_length

    def set_mode(self, mode: Literal["train", "inference"]) -> None:
        """
        Switch between training and inference modes.
        
        Args:
            mode: "train" for fine-tuning, "inference" for generation
        
        Raises:
            ValueError: If mode is not 'train' or 'inference'
        """
        if mode not in ("train", "inference"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'inference'")

        if mode == "train":
            FastLanguageModel.for_training(self.model)
        else:
            FastLanguageModel.for_inference(self.model)

        self._mode = mode

    def create_input(
            self,
            messages: list[dict],
            reasoning_effort: str = "low"
    ) -> BatchEncoding:
        """
        Tokenize messages for generation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            reasoning_effort: GPT-OSS specific ['low', 'medium', 'high']
        
        Returns:
            BatchEncoding ready for model.generate()
        """
        inputs: BatchEncoding = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=reasoning_effort,
        ).to("cuda")
        return inputs

    def generate(
            self,
            system: str,
            user: str,
            config: GenerationConfig | None = None,
            stream: bool = True
    ):
        """
        Generate a response given system and user messages.
        
        Args:
            system: System instruction
            user: User input/question
            config: Generation parameters (uses defaults if None)
            stream: Whether to stream output to console
        
        Returns:
            Generated token IDs
        
        Raises:
            RuntimeError: If not in inference mode
        """
        if self._mode != "inference":
            raise RuntimeError(
                f"Model is in '{self._mode}' mode. "
                "Call set_mode('inference') before generating."
            )

        config = config or GenerationConfig()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        inputs = self.create_input(messages, config.reasoning_effort)

        streamer = TextStreamer(self.tokenizer) if stream else None

        output = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            streamer=streamer,
            use_cache=config.use_cache
        )
        return output

    def generate_from_messages(
            self,
            messages: list[dict],
            config: GenerationConfig | None = None,
            stream: bool = True
    ):
        """
        Generate from raw message list (for advanced use cases).
        
        Args:
            messages: List of message dicts
            config: Generation parameters
            stream: Whether to stream output
        
        Returns:
            Generated token IDs
        """
        if self._mode != "inference":
            raise RuntimeError(
                f"Model is in '{self._mode}' mode. "
                "Call set_mode('inference') before generating."
            )

        config = config or GenerationConfig()
        inputs = self.create_input(messages, config.reasoning_effort)

        streamer = TextStreamer(self.tokenizer) if stream else None

        output = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            streamer=streamer,
            use_cache=config.use_cache
        )
        return output

    def save(self, path: str, save_tokenizer: bool = True) -> None:
        """
        Save LoRA weights to disk.
        
        Args:
            path: Directory path to save the model
            save_tokenizer: Whether to also save the tokenizer
        """
        self.model.save_pretrained(path)
        if save_tokenizer:
            self.tokenizer.save_pretrained(path)

    def save_merged(self, path: str) -> None:
        """
        Merge LoRA weights into base model and save.
        
        Args:
            path: Directory path to save the merged model
        """
        self.model.save_pretrained_merged(path, self.tokenizer)
