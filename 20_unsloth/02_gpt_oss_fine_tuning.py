"""
GPT-OSS Fine-Tuning Script (Text-to-SQL)

This script demonstrates fine-tuning GPT-OSS with LoRA for text-to-SQL tasks
using the modular gpt_oss package.

Usage:
    python 02_gpt_oss_fine_tuning.py
"""
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from gpt_oss import GptOssModel, ModelConfig, LoRAConfig, preprocess_sql_dataset


# =============================================================================
# WORKAROUND: Patch to_dict() to fix token obfuscation bug
# Bug: Original to_dict() replaces ALL fields ending with '_token' with
# placeholders like '<EOS_TOKEN>', even when the value is None.
# Fix: Restore actual token values after calling original to_dict().
# =============================================================================
_original_to_dict = SFTConfig.to_dict

def _patched_to_dict(self):
    d = _original_to_dict(self)
    # Restore actual token values that were obfuscated
    for k, v in d.items():
        if k.endswith("_token"):
            d[k] = getattr(self, k, v)
    return d

SFTConfig.to_dict = _patched_to_dict
# =============================================================================


def main():
    # ========================================================================
    # Step 1: Initialize Model with LoRA
    # ========================================================================
    print("[1/4] Initializing model...")
    
    model = GptOssModel(
        model_config=ModelConfig(
            model_name="unsloth/gpt-oss-20b",
            max_seq_length=4096
        ),
        lora_config=LoRAConfig(
            r=16,           # Higher rank for SQL understanding
            lora_alpha=32   # 2x rank as recommended
        )
    )
    
    # Enable training mode
    model.set_mode("train")
    
    # ========================================================================
    # Step 2: Load and Preprocess Dataset
    # ========================================================================
    print("\n[2/4] Loading and preprocessing dataset...")
    
    raw_dataset = load_dataset("gretelai/synthetic_text_to_sql")
    
    # Start with 'single join' complexity
    # Expand later: ['basic', 'single join', 'aggregation', ...]
    train_dataset = preprocess_sql_dataset(
        dataset=raw_dataset['train'],
        formatter=model.formatter,
        complexity_filter='single join',
        max_samples=None
    )
    
    print(f"Training dataset size: {len(train_dataset)} samples")
    print(f"Sample formatted text:\n{train_dataset[0]['text'][:500]}...")
    
    # ========================================================================
    # Step 3: Configure Training
    # ========================================================================
    print("\n[3/4] Configuring training...")
    
    # ========================================================================
    # Step 4: Train
    # ========================================================================
    print("\n[4/4] Starting training...")

    
    trainer = SFTTrainer(
        model=model.model,
        processing_class=model.tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            # Note: eos_token is NOT set here - the tokenizer already has it configured
            # Setting it explicitly causes a bug where to_dict() obfuscates it to '<EOS_TOKEN>'
            per_device_train_batch_size=8,
            gradient_accumulation_steps=32,
            warmup_steps=5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=30,
            learning_rate=2e-4,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Use TrackIO/WandB etc
            # Fix for accelerate device placement with Unsloth models
            fp16=False,
            bf16=True,
        ),
    )
    
    # trainer = SFTTrainer(
    #     model=model.model,
    #     processing_class=model.tokenizer,
    #     train_dataset=train_dataset,
    #     args=sft_config,
    # )

    trainer.train()
    
    # ========================================================================
    # Save Model
    # ========================================================================
    print("\nSaving trained model...")
    
    model.save("./text2sql_lora_model")
    
    # Optionally merge and save full model
    # model.save_merged("./text2sql_merged_model")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("LoRA weights saved to: ./text2sql_lora_model")
    print("=" * 60)


if __name__ == '__main__':
    main()
