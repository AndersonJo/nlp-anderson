#!/usr/bin/env python3
"""
SimCSE Training Script (Optimized with Preprocessed Data)
"""

import torch
import os
import time
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from config import Config
from model import SimCSEForTraining
from data_loader import prepare_data
from utils import (
    set_seed, 
    save_model, 
    evaluate_similarity, 
    AverageMeter, 
    format_time,
    create_optimizer_and_scheduler
)

def train_epoch(model, dataloader, optimizer, scheduler, config, writer, epoch, global_step, scaler=None):
    """
    Train for one epoch
    """
    model.train()
    
    losses = AverageMeter()
    epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for step, batch in enumerate(epoch_iterator):
        # Move batch to device with non-blocking transfer
        batch = {k: v.to(config.device, non_blocking=True) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if config.fp16 and scaler is not None:
            with autocast():
                outputs = model(**batch)
                loss = outputs['loss']
                
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
        else:
            # Standard forward pass
            outputs = model(**batch)
            loss = outputs['loss']
            
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
            
            loss.backward()
        
        # Update model
        if (step + 1) % config.gradient_accumulation_steps == 0:
            if config.fp16 and scaler is not None:
                # Gradient clipping with scaler
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard gradient clipping and optimization
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Logging
            if global_step % config.logging_steps == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step)
        
        # Update metrics
        losses.update(loss.item())
        
        # Update progress bar
        epoch_iterator.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    return global_step, losses.avg

def train(config):
    """
    Main training function
    """
    # Set seed
    set_seed(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Prepare preprocessed data
    print("Preparing preprocessed data...")
    train_dataloader, tokenizer = prepare_data(config, config.preprocessed_data_path)
    
    # Initialize model
    print(f"Initializing model: {config.model_name}")
    model = SimCSEForTraining(
        model_name=config.model_name,
        pooler_type=config.pooler_type,
        temp=config.temp
    )
    
    # Move model to device
    model.to(config.device)
    
    # Use DataParallel for multi-GPU training
    if config.use_data_parallel and config.num_gpus > 1:
        print(f"Using DataParallel with {config.num_gpus} GPUs")
        model = torch.nn.DataParallel(model)
        effective_batch_size = config.batch_size * config.num_gpus
        print(f"Effective batch size with DataParallel: {effective_batch_size}")
    elif config.num_gpus > 0:
        print(f"Using single GPU: {config.device}")
    else:
        print("Using CPU for training")
    
    # Compile model for PyTorch 2.0+ speed improvements
    if config.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
            print("Model compilation successful!")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Continuing without compilation...")
    
    # Calculate total training steps
    num_training_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    
    # Initialize optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, num_training_steps)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if config.fp16 else None
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'logs'))
    
    # Print training info
    print("=" * 70)
    print("Training Information:")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Mixed Precision (FP16): {config.fp16}")
    print(f"  Model compilation: {config.compile_model}")
    print(f"  DataLoader workers: {config.dataloader_num_workers}")
    print(f"  Training samples: {len(train_dataloader.dataset)}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max sequence length: {config.max_seq_length}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Total training steps: {num_training_steps}")
    print("=" * 70)
    
    # Training loop
    global_step = 0
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 50)
        
        # Train for one epoch
        global_step, avg_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, config, writer, epoch, global_step, scaler
        )
        
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\nTraining completed in {format_time(total_time)}")
    
    # Save final model
    print("Saving final model...")
    model_to_save = model.module if hasattr(model, 'module') else model
    save_model(model_to_save, tokenizer, config.output_dir)
    
    # Final evaluation
    print("\nFinal evaluation:")
    eval_model = model.module.simcse if hasattr(model, 'module') else model.simcse
    correlation, similarities = evaluate_similarity(eval_model, tokenizer, config.device)
    
    writer.close()
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description='Train SimCSE with preprocessed data')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    parser.add_argument('--preprocessed_data', type=str, 
                       default='./preprocessed_data/train_preprocessed.pkl',
                       help='Path to preprocessed data file')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # Load from file if provided
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.Config()
    else:
        # Use default config
        config = Config()
    
    # Override config with command line arguments
    if args.preprocessed_data:
        config.preprocessed_data_path = args.preprocessed_data
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    
    # Validate preprocessed data exists
    if not os.path.exists(config.preprocessed_data_path):
        print(f"❌ Error: Preprocessed data not found at {config.preprocessed_data_path}")
        print(f"📝 Please run the following command first:")
        print(f"   python preprocess.py --max_length {config.max_seq_length}")
        return
    
    print("=" * 80)
    print("🚀 SimCSE Training with Preprocessed Data")
    print("=" * 80)
    print(f"📁 Preprocessed data: {config.preprocessed_data_path}")
    print(f"📁 Output directory: {config.output_dir}")
    print(f"🔧 Batch size: {config.batch_size}")
    print(f"🔧 Epochs: {config.num_epochs}")
    print("=" * 80)
    
    # Run training
    model, tokenizer = train(config)
    
    print(f"\n✅ Training completed successfully!")
    print(f"📁 Model saved to: {config.output_dir}")
    print(f"📊 Logs saved to: {os.path.join(config.output_dir, 'logs')}")

if __name__ == "__main__":
    main() 