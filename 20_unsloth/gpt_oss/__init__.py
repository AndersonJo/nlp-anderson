"""
GptOss: Modular wrapper for GPT-OSS fine-tuning with Unsloth.

This package provides a clean, maintainable interface for:
- Model loading and configuration
- LoRA adapter management
- Chat template formatting
- Dataset preprocessing

Quick Start:
    from gpt_oss import GptOssModel, ModelConfig, LoRAConfig

    # Initialize with LoRA
    model = GptOssModel(
        model_config=ModelConfig(max_seq_length=4096),
        lora_config=LoRAConfig(r=16, lora_alpha=32)
    )

    # Training
    model.set_mode("train")
    # ... use model.model and model.tokenizer with SFTTrainer

    # Inference
    model.set_mode("inference")
    model.generate(
        system="You are an SQL expert...",
        user="Generate a query to..."
    )

Architecture:
    - config.py: Dataclass configurations (ModelConfig, LoRAConfig, GenerationConfig)
    - formatter.py: ChatFormatter for prompt handling
    - model.py: GptOssModel main class
    - preprocessing.py: Dataset utilities
"""

from .config import ModelConfig, LoRAConfig, GenerationConfig
from .formatter import ChatFormatter
from .model import GptOssModel
from .preprocessing import format_sql_example, preprocess_sql_dataset

__all__ = [
    # Configs
    "ModelConfig",
    "LoRAConfig", 
    "GenerationConfig",
    # Classes
    "GptOssModel",
    "ChatFormatter",
    # Functions
    "format_sql_example",
    "preprocess_sql_dataset",
]
