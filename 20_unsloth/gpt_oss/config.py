"""
Configuration dataclasses for GptOss model.

Design Pattern:
- Immutable configurations via frozen dataclasses
- Serializable for experiment tracking (wandb, mlflow)
- Type-safe with proper defaults

Usage:
    model_config = ModelConfig(max_seq_length=8192)
    lora_config = LoRAConfig(r=16, lora_alpha=32)
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    """
    Base model configuration.
    
    Attributes:
        model_name: HuggingFace model identifier
        max_seq_length: Maximum context length (increase for long inputs like SQL schemas)
        load_in_4bit: Enable 4-bit quantization (saves VRAM, slight quality tradeoff)
        device_map: Device placement strategy ("cuda", "auto", etc.)
    """
    model_name: str = "unsloth/gpt-oss-20b"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    full_finetuning: bool = False
    low_cpu_mem_usage: bool = True
    device_map: str = "cuda"
    dtype: str | None = None  # Auto detection (likely bfloat16)


@dataclass(frozen=True)
class LoRAConfig:
    """
    LoRA (Low-Rank Adaptation) hyperparameters.
    
    Key Parameters:
        r: LoRA rank. 8 (lightweight) ~ 128 (complex tasks). Default: 8
        lora_alpha: Scaling factor. Typically 2x rank. Default: 16
        target_modules: Layers to apply LoRA (attention + MLP projections)
        lora_dropout: 0 is fastest. Default: 0
    
    Practical Tips:
        - r=8~16: General fine-tuning
        - r=64~128: High-quality requirements
        - lora_alpha = 2 × r rule of thumb
    """
    r: int = 8
    lora_alpha: int = 16
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    )
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str | bool = "unsloth"  # "unsloth" saves 30% VRAM
    random_state: int = 3407
    use_rslora: bool = False  # Rank-stabilized LoRA for high ranks
    loftq_config: dict | None = None


@dataclass(frozen=True)
class GenerationConfig:
    """
    Inference-time generation parameters.
    
    Attributes:
        max_new_tokens: Maximum tokens to generate
        reasoning_effort: GPT-OSS specific parameter ['low', 'medium', 'high']
        use_cache: Enable KV-cache for faster generation
    """
    max_new_tokens: int = 2048
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    use_cache: bool = True
