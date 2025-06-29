#!/usr/bin/env python3
"""
Sentence BERT training script
Complete training example with TensorBoard logging and Multi-GPU support
"""

import argparse
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import (
    SentenceBERT, SNLIDataset, TripletDataset,
    compute_similarity, triplet_loss, cosine_triplet_loss
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


def setup_distributed(rank, world_size, port="12355"):
    """
    Setup distributed training
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """
    Cleanup distributed training
    """
    dist.destroy_process_group()


def get_gpu_strategy(strategy, model, gpu_ids=None):
    """
    Configure GPU strategy for training
    
    Args:
        strategy: 'single', 'dp', 'ddp'
        model: PyTorch model
        gpu_ids: List of GPU IDs to use
    
    Returns:
        Configured model and device
    """
    if strategy == 'single':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Using single GPU: {device}")
        
    elif strategy == 'dp':
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        device = torch.device(f'cuda:{gpu_ids[0]}')
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"Using DataParallel with GPUs: {gpu_ids}")
        print(f"Primary device: {device}")
        
    elif strategy == 'ddp':
        # DDP setup will be handled in distributed training function
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        model = model.to(device)
        print(f"Using DistributedDataParallel with device: {device}")
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return model, device


def create_distributed_data_loaders(train_data, val_data, test_data, batch_size=16, 
                                   max_length=128, use_triplet=False, rank=0, world_size=1):
    """
    Create DataLoaders with distributed sampling support
    """
    from model import SentenceBERT, SNLIDataset, TripletDataset
    
    # Initialize model and tokenizer
    model = SentenceBERT()
    tokenizer = model.tokenizer
    
    # Create datasets
    if use_triplet:
        train_dataset = TripletDataset(train_data, tokenizer, max_length)
        val_dataset = TripletDataset(val_data, tokenizer, max_length)
        test_dataset = TripletDataset(test_data, tokenizer, max_length)
    else:
        train_dataset = SNLIDataset(train_data, tokenizer, max_length)
        val_dataset = SNLIDataset(val_data, tokenizer, max_length)
        test_dataset = SNLIDataset(test_data, tokenizer, max_length)
    
    # Create samplers for distributed training
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle = True
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        sampler=test_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, model


def train_sentence_bert_classification_multi_gpu(model, train_loader, val_loader, device, epochs=3, lr=2e-5, 
                                                log_dir=None, use_amp=True, rank=0, world_size=1):
    """
    Multi-GPU Sentence BERT model training function for classification (with TensorBoard logging and Mixed Precision)
    """
    # Setup model for multi-GPU training
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None

    # TensorBoard setup (only on rank 0)
    if rank == 0:
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"runs/sentence_bert_classification_multi_gpu_{timestamp}"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
        print(f"To view logs, run: tensorboard --logdir={log_dir}")
    else:
        writer = None

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []

    if rank == 0:
        print("Starting multi-GPU classification training...")
        print(f"Using {world_size} GPU(s)")
        print(f"Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")

    for epoch in range(epochs):
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        # Training loop
        if rank == 0:
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        else:
            train_pbar = train_loader

        for batch_idx, batch in enumerate(train_pbar):
            input_ids_1 = batch['input_ids_1'].to(device, non_blocking=True)
            attention_mask_1 = batch['attention_mask_1'].to(device, non_blocking=True)
            input_ids_2 = batch['input_ids_2'].to(device, non_blocking=True)
            attention_mask_2 = batch['attention_mask_2'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision forward pass
            if use_amp:
                with autocast():
                    logits, _, _ = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                    loss = criterion(logits, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _, _ = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # TensorBoard: batch loss logging (only on rank 0)
            if writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)

            if rank == 0:
                train_pbar.set_postfix({'loss': loss.item(), 'acc': train_correct / train_total})

        # Synchronize metrics across all GPUs
        if world_size > 1:
            train_loss_tensor = torch.tensor(total_train_loss).to(device)
            train_correct_tensor = torch.tensor(train_correct).to(device)
            train_total_tensor = torch.tensor(train_total).to(device)
            
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
            
            train_loss = train_loss_tensor.item() / (len(train_loader) * world_size)
            train_accuracy = train_correct_tensor.item() / train_total_tensor.item()
        else:
            train_loss = total_train_loss / len(train_loader)
            train_accuracy = train_correct / train_total

        train_losses.append(train_loss)

        # Validation (similar structure with synchronization)
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            if rank == 0:
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')
            else:
                val_pbar = val_loader

            for batch in val_pbar:
                input_ids_1 = batch['input_ids_1'].to(device, non_blocking=True)
                attention_mask_1 = batch['attention_mask_1'].to(device, non_blocking=True)
                input_ids_2 = batch['input_ids_2'].to(device, non_blocking=True)
                attention_mask_2 = batch['attention_mask_2'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)

                if use_amp:
                    with autocast():
                        logits, _, _ = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                        loss = criterion(logits, labels)
                else:
                    logits, _, _ = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                    loss = criterion(logits, labels)

                total_val_loss += loss.item()

                # Accuracy calculation
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                if rank == 0:
                    val_pbar.set_postfix({'loss': loss.item(), 'acc': val_correct / val_total})

        # Synchronize validation metrics
        if world_size > 1:
            val_loss_tensor = torch.tensor(total_val_loss).to(device)
            val_correct_tensor = torch.tensor(val_correct).to(device)
            val_total_tensor = torch.tensor(val_total).to(device)
            
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)
            
            val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)
            val_accuracy = val_correct_tensor.item() / val_total_tensor.item()
        else:
            val_loss = total_val_loss / len(val_loader)
            val_accuracy = val_correct / val_total

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Learning rate update
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # TensorBoard logging (only on rank 0)
        if writer is not None:
            writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
            writer.add_scalar('Accuracy/Train_Epoch', train_accuracy, epoch)
            writer.add_scalar('Accuracy/Val_Epoch', val_accuracy, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)

        if rank == 0:
            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            print(f'  Learning Rate: {current_lr:.2e}')
            print('-' * 50)

    # Close TensorBoard
    if writer is not None:
        writer.close()

    return train_losses, val_losses, val_accuracies


def train_sentence_bert_triplet_multi_gpu(model, train_loader, val_loader, device, epochs=3, lr=2e-5,
                                         log_dir=None, margin=1.0, use_cosine=False, use_amp=True, 
                                         rank=0, world_size=1):
    """
    Multi-GPU Sentence BERT model training function using triplet loss
    """
    from model import triplet_loss, cosine_triplet_loss
    
    # Setup model for multi-GPU training
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None

    # TensorBoard setup (only on rank 0)
    if rank == 0:
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"runs/sentence_bert_triplet_multi_gpu_{timestamp}"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
        print(f"To view logs, run: tensorboard --logdir={log_dir}")
        print(f"Using {'cosine' if use_cosine else 'euclidean'} triplet loss with margin={margin}")
        print(f"Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    else:
        writer = None

    # Loss function and optimizer
    if use_cosine:
        criterion = lambda a, p, n: cosine_triplet_loss(a, p, n, margin=margin)
    else:
        criterion = lambda a, p, n: triplet_loss(a, p, n, margin=margin)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training history
    train_losses = []
    val_losses = []

    if rank == 0:
        print("Starting multi-GPU triplet training...")
        print(f"Using {world_size} GPU(s)")

    for epoch in range(epochs):
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        total_train_loss = 0

        # Training loop
        if rank == 0:
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        else:
            train_pbar = train_loader

        for batch_idx, batch in enumerate(train_pbar):
            input_ids_a = batch['input_ids_a'].to(device, non_blocking=True)
            attention_mask_a = batch['attention_mask_a'].to(device, non_blocking=True)
            input_ids_p = batch['input_ids_p'].to(device, non_blocking=True)
            attention_mask_p = batch['attention_mask_p'].to(device, non_blocking=True)
            input_ids_n = batch['input_ids_n'].to(device, non_blocking=True)
            attention_mask_n = batch['attention_mask_n'].to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision forward pass
            if use_amp:
                with autocast():
                    embeddings_a, embeddings_p, embeddings_n = model.module.forward_triplet(
                        input_ids_a, attention_mask_a,
                        input_ids_p, attention_mask_p,
                        input_ids_n, attention_mask_n
                    ) if hasattr(model, 'module') else model.forward_triplet(
                        input_ids_a, attention_mask_a,
                        input_ids_p, attention_mask_p,
                        input_ids_n, attention_mask_n
                    )
                    loss = criterion(embeddings_a, embeddings_p, embeddings_n)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                embeddings_a, embeddings_p, embeddings_n = model.module.forward_triplet(
                    input_ids_a, attention_mask_a,
                    input_ids_p, attention_mask_p,
                    input_ids_n, attention_mask_n
                ) if hasattr(model, 'module') else model.forward_triplet(
                    input_ids_a, attention_mask_a,
                    input_ids_p, attention_mask_p,
                    input_ids_n, attention_mask_n
                )
                loss = criterion(embeddings_a, embeddings_p, embeddings_n)
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()

            # TensorBoard: batch loss logging (only on rank 0)
            if writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)

            if rank == 0:
                train_pbar.set_postfix({'loss': loss.item()})

        # Synchronize training loss across all GPUs
        if world_size > 1:
            train_loss_tensor = torch.tensor(total_train_loss).to(device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = train_loss_tensor.item() / (len(train_loader) * world_size)
        else:
            train_loss = total_train_loss / len(train_loader)

        train_losses.append(train_loss)

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            if rank == 0:
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')
            else:
                val_pbar = val_loader

            for batch in val_pbar:
                input_ids_a = batch['input_ids_a'].to(device, non_blocking=True)
                attention_mask_a = batch['attention_mask_a'].to(device, non_blocking=True)
                input_ids_p = batch['input_ids_p'].to(device, non_blocking=True)
                attention_mask_p = batch['attention_mask_p'].to(device, non_blocking=True)
                input_ids_n = batch['input_ids_n'].to(device, non_blocking=True)
                attention_mask_n = batch['attention_mask_n'].to(device, non_blocking=True)

                if use_amp:
                    with autocast():
                        embeddings_a, embeddings_p, embeddings_n = model.module.forward_triplet(
                            input_ids_a, attention_mask_a,
                            input_ids_p, attention_mask_p,
                            input_ids_n, attention_mask_n
                        ) if hasattr(model, 'module') else model.forward_triplet(
                            input_ids_a, attention_mask_a,
                            input_ids_p, attention_mask_p,
                            input_ids_n, attention_mask_n
                        )
                        loss = criterion(embeddings_a, embeddings_p, embeddings_n)
                else:
                    embeddings_a, embeddings_p, embeddings_n = model.module.forward_triplet(
                        input_ids_a, attention_mask_a,
                        input_ids_p, attention_mask_p,
                        input_ids_n, attention_mask_n
                    ) if hasattr(model, 'module') else model.forward_triplet(
                        input_ids_a, attention_mask_a,
                        input_ids_p, attention_mask_p,
                        input_ids_n, attention_mask_n
                    )
                    loss = criterion(embeddings_a, embeddings_p, embeddings_n)

                total_val_loss += loss.item()

                if rank == 0:
                    val_pbar.set_postfix({'loss': loss.item()})

        # Synchronize validation loss
        if world_size > 1:
            val_loss_tensor = torch.tensor(total_val_loss).to(device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)
        else:
            val_loss = total_val_loss / len(val_loader)

        val_losses.append(val_loss)

        # Learning rate update
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # TensorBoard logging (only on rank 0)
        if writer is not None:
            writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)

        if rank == 0:
            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Learning Rate: {current_lr:.2e}')
            print('-' * 50)

    # Close TensorBoard
    if writer is not None:
        writer.close()

    return train_losses, val_losses


def train_sentence_bert_classification(model, train_loader, val_loader, device, epochs=3, lr=2e-5, log_dir=None):
    """
    Sentence BERT model training function for classification (with TensorBoard logging)
    """
    model.to(device)

    # TensorBoard setup
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/sentence_bert_classification_{timestamp}"

    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"To view logs, run: tensorboard --logdir={log_dir}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []

    print("Starting classification training...")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        # Training loop
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for batch_idx, batch in enumerate(train_pbar):
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            logits, _, _ = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # TensorBoard: batch loss logging
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)

            train_pbar.set_postfix({'loss': loss.item(), 'acc': train_correct / train_total})

        train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')
            for batch in val_pbar:
                input_ids_1 = batch['input_ids_1'].to(device)
                attention_mask_1 = batch['attention_mask_1'].to(device)
                input_ids_2 = batch['input_ids_2'].to(device)
                attention_mask_2 = batch['attention_mask_2'].to(device)
                labels = batch['label'].to(device)

                logits, _, _ = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                loss = criterion(logits, labels)

                total_val_loss += loss.item()

                # Accuracy calculation
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_pbar.set_postfix({'loss': loss.item(), 'acc': val_correct / val_total})

        val_loss = total_val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Learning rate update
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # TensorBoard: epoch metrics logging
        writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
        writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
        writer.add_scalar('Accuracy/Train_Epoch', train_accuracy, epoch)
        writer.add_scalar('Accuracy/Val_Epoch', val_accuracy, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Model parameter histograms (only for first epoch)
        if epoch == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f'Parameters/{name}', param.data, epoch)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        print(f'  Learning Rate: {current_lr:.2e}')
        print('-' * 50)

    # Close TensorBoard
    writer.close()

    return train_losses, val_losses, val_accuracies


def train_sentence_bert_triplet(model, train_loader, val_loader, device, epochs=3, lr=2e-5,
                                log_dir=None, margin=1.0, use_cosine=False):
    """
    Sentence BERT model training function using triplet loss
    """
    model.to(device)

    # TensorBoard setup
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/sentence_bert_triplet_{timestamp}"

    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"To view logs, run: tensorboard --logdir={log_dir}")

    # Loss function and optimizer
    if use_cosine:
        criterion = lambda a, p, n: cosine_triplet_loss(a, p, n, margin=margin)
        print(f"Using cosine triplet loss with margin={margin}")
    else:
        criterion = lambda a, p, n: triplet_loss(a, p, n, margin=margin)
        print(f"Using euclidean triplet loss with margin={margin}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training history
    train_losses = []
    val_losses = []

    print("Starting triplet training...")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        # Training loop
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for batch_idx, batch in enumerate(train_pbar):
            input_ids_a = batch['input_ids_a'].to(device)
            attention_mask_a = batch['attention_mask_a'].to(device)
            input_ids_p = batch['input_ids_p'].to(device)
            attention_mask_p = batch['attention_mask_p'].to(device)
            input_ids_n = batch['input_ids_n'].to(device)
            attention_mask_n = batch['attention_mask_n'].to(device)

            optimizer.zero_grad()

            # Triplet forward pass
            embeddings_a, embeddings_p, embeddings_n = model.forward_triplet(
                input_ids_a, attention_mask_a,
                input_ids_p, attention_mask_p,
                input_ids_n, attention_mask_n
            )

            # Calculate triplet loss
            loss = criterion(embeddings_a, embeddings_p, embeddings_n)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # TensorBoard: batch loss logging
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Triplet_Train_Batch', loss.item(), global_step)

            train_pbar.set_postfix({'loss': loss.item()})

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')
            for batch in val_pbar:
                input_ids_a = batch['input_ids_a'].to(device)
                attention_mask_a = batch['attention_mask_a'].to(device)
                input_ids_p = batch['input_ids_p'].to(device)
                attention_mask_p = batch['attention_mask_p'].to(device)
                input_ids_n = batch['input_ids_n'].to(device)
                attention_mask_n = batch['attention_mask_n'].to(device)

                embeddings_a, embeddings_p, embeddings_n = model.forward_triplet(
                    input_ids_a, attention_mask_a,
                    input_ids_p, attention_mask_p,
                    input_ids_n, attention_mask_n
                )

                loss = criterion(embeddings_a, embeddings_p, embeddings_n)
                total_val_loss += loss.item()

                val_pbar.set_postfix({'loss': loss.item()})

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        # Learning rate update
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # TensorBoard: epoch metrics logging
        writer.add_scalar('Loss/Triplet_Train_Epoch', train_loss, epoch)
        writer.add_scalar('Loss/Triplet_Val_Epoch', val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Model parameter histograms (only for first epoch)
        if epoch == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f'Parameters/{name}', param.data, epoch)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Learning Rate: {current_lr:.2e}')
        print('-' * 50)

    # Close TensorBoard
    writer.close()

    return train_losses, val_losses


def plot_training_history(train_losses, val_losses, val_accuracies=None):
    """
    Plot training history
    """
    if val_accuracies is not None:
        # Classification training (with accuracy)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
    else:
        # Triplet training (loss only)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.plot(train_losses, label='Train Loss')
        ax.plot(val_losses, label='Validation Loss')
        ax.set_title('Triplet Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def load_and_prepare_snli_data(train_samples=10000, val_samples=1000, test_samples=1000):
    """
    Load and preprocess SNLI data
    """
    print("Loading SNLI dataset...")

    # Load SNLI dataset
    snli = load_dataset("snli")

    # Filter valid data (remove invalid labels)
    def filter_valid_data(example):
        return example['label'] != -1

    # Filter valid data from each split
    train_data = snli['train'].filter(filter_valid_data)
    validation_data = snli['validation'].filter(filter_valid_data)
    test_data = snli['test'].filter(filter_valid_data)

    # Limit samples if specified
    if train_samples:
        train_data = train_data.select(range(min(train_samples, len(train_data))))
    if val_samples:
        validation_data = validation_data.select(range(min(val_samples, len(validation_data))))
    if test_samples:
        test_data = test_data.select(range(min(test_samples, len(test_data))))

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(validation_data)}")
    print(f"Test samples: {len(test_data)}")

    return train_data, validation_data, test_data


def create_data_loaders(train_data, val_data, test_data, batch_size=16, max_length=128, use_triplet=False):
    """
    Create DataLoaders
    """
    # Initialize model (for tokenizer)
    model = SentenceBERT()
    tokenizer = model.tokenizer

    if use_triplet:
        # Create triplet datasets
        train_dataset = TripletDataset(train_data, tokenizer, max_length)
        val_dataset = TripletDataset(val_data, tokenizer, max_length)
        test_dataset = TripletDataset(test_data, tokenizer, max_length)
    else:
        # Create classification datasets
        train_dataset = SNLIDataset(train_data, tokenizer, max_length)
        val_dataset = SNLIDataset(val_data, tokenizer, max_length)
        test_dataset = SNLIDataset(test_data, tokenizer, max_length)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader, test_loader, model


def evaluate_model(model, test_loader, device):
    """
    Evaluate model performance on test data
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    label_names = ['entailment', 'contradiction', 'neutral']
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}

    with torch.no_grad():
        for batch in test_loader:
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            labels = batch['label'].to(device)

            logits, _, _ = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            _, predicted = torch.max(logits, 1)

            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            # Calculate class-wise accuracy
            for i in range(3):
                mask = (labels == i)
                class_total[i] += mask.sum().item()
                class_correct[i] += (predicted[mask] == labels[mask]).sum().item()

    overall_accuracy = total_correct / total_samples
    print(f"\nOverall Test Accuracy: {overall_accuracy:.4f}")

    print("\nClass-wise Accuracy:")
    for i, name in enumerate(label_names):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f"  {name}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")

    return overall_accuracy


def test_similarity_functionality(model, device):
    """
    Test sentence similarity functionality
    """
    print("\n" + "=" * 50)
    print("Testing Sentence Similarity Functionality")
    print("=" * 50)

    # Test sentences
    test_sentences = [
        ("A man is playing guitar on stage.", "A musician is performing with a guitar."),
        ("The cat is sleeping on the couch.", "A dog is running in the park."),
        ("The weather is sunny today.", "It's raining heavily outside."),
        ("I love this movie.", "This film is amazing."),
        ("The food tastes delicious.", "This meal is terrible.")
    ]

    model.eval()
    with torch.no_grad():
        for i, (sentence1, sentence2) in enumerate(test_sentences, 1):
            # Tokenize sentences
            encoding1 = model.tokenizer(
                sentence1,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )

            encoding2 = model.tokenizer(
                sentence2,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )

            # Get embeddings
            embedding1 = model.encode(
                encoding1['input_ids'].to(device),
                encoding1['attention_mask'].to(device)
            )

            embedding2 = model.encode(
                encoding2['input_ids'].to(device),
                encoding2['attention_mask'].to(device)
            )

            # Calculate similarity
            similarity = compute_similarity(embedding1, embedding2).item()

            print(f"Pair {i}:")
            print(f"  Sentence 1: {sentence1}")
            print(f"  Sentence 2: {sentence2}")
            print(f"  Similarity: {similarity:.4f}")
            print()


def distributed_main(rank, world_size, args):
    """
    Main function for distributed training
    """
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Load data
    train_data, val_data, test_data = load_and_prepare_snli_data(
        args.train_samples, args.val_samples, args.test_samples
    )
    
    # Create DataLoaders with distributed sampling
    train_loader, val_loader, test_loader, model = create_distributed_data_loaders(
        train_data, val_data, test_data, args.batch_size, args.max_length, 
        args.triplet, rank, world_size
    )
    
    # Setup model and device
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    
    if rank == 0:
        print(f"DataLoaders created successfully!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print("-" * 50)
    
    # Train model with multi-GPU support
    if rank == 0:
        print("Starting distributed training...")
    
    if args.triplet:
        # Triplet training with DDP
        train_losses, val_losses = train_sentence_bert_triplet_multi_gpu(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            log_dir=args.log_dir,
            margin=args.margin,
            use_cosine=args.use_cosine,
            use_amp=args.use_amp,
            rank=rank,
            world_size=world_size
        )
        val_accuracies = None
    else:
        # Classification training with DDP
        train_losses, val_losses, val_accuracies = train_sentence_bert_classification_multi_gpu(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            log_dir=args.log_dir,
            use_amp=args.use_amp,
            rank=rank,
            world_size=world_size
        )
    
    # Save model (only on rank 0)
    if rank == 0 and args.save_model:
        model_save_path = f'sentence_bert_snli_{"triplet" if args.triplet else "classification"}_multi_gpu_model.pth'
        # Extract the actual model from DDP wrapper
        model_to_save = model.module if hasattr(model, 'module') else model
        save_dict = {
            'model_state_dict': model_to_save.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'hyperparameters': {
                'batch_size': args.batch_size,
                'max_length': args.max_length,
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'world_size': world_size,
                'use_amp': args.use_amp,
                'training_mode': 'triplet' if args.triplet else 'classification'
            }
        }
        
        if val_accuracies is not None:
            save_dict['val_accuracies'] = val_accuracies
        if args.triplet:
            save_dict['margin'] = args.margin
            save_dict['use_cosine'] = args.use_cosine
            
        torch.save(save_dict, model_save_path)
        print(f"\nModel saved as '{model_save_path}'")
    
    # Cleanup
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='Train Sentence BERT with Multi-GPU support and TensorBoard logging')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--log_dir', type=str, default=None, help='TensorBoard log directory')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable TensorBoard server (enabled by default)')
    parser.add_argument('--max_length', type=int, default=160, help='Maximum sequence length')
    parser.add_argument('--train_samples', type=int, default=0, help='Number of training samples to use')
    parser.add_argument('--val_samples', type=int, default=0, help='Number of validation samples to use')
    parser.add_argument('--test_samples', type=int, default=0, help='Number of test samples to use')
    parser.add_argument('--save_model', action='store_true', default=True, help='Save trained model')
    parser.add_argument('--test_similarity', action='store_true', help='Test sentence similarity functionality')
    parser.add_argument('--triplet', action='store_true', default=True, help='Use triplet loss instead of classification')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
    parser.add_argument('--use_cosine', action='store_true', help='Use cosine distance for triplet loss')
    
    # Multi-GPU arguments
    parser.add_argument('--gpu_strategy', type=str, choices=['single', 'dp', 'ddp'], default='ddp',
                       help='GPU strategy: single, dp (DataParallel), ddp (DistributedDataParallel)')
    parser.add_argument('--gpu_ids', type=str, default=None, 
                       help='GPU IDs to use (e.g., "0,1,2,3,4,5,6,7" for 8 GPUs)')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use mixed precision training')
    parser.add_argument('--world_size', type=int, default=None, help='Number of GPUs for distributed training')
    
    args = parser.parse_args()
    
    # GPU setup
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        args.gpu_strategy = 'single'
    
    # Parse GPU IDs
    if args.gpu_ids is not None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        args.world_size = len(gpu_ids)
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
        if args.world_size is None:
            args.world_size = len(gpu_ids)
    
    print(f"🚀 GPU Strategy: {args.gpu_strategy.upper()}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Using GPUs: {gpu_ids}")
    print(f"World Size: {args.world_size}")

    # Hyperparameters
    print(f"Hyperparameters:")
    print(f"  Batch Size per GPU: {args.batch_size}")
    print(f"  Effective Batch Size: {args.batch_size * args.world_size}")
    print(f"  Max Length: {args.max_length}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Mixed Precision: {args.use_amp}")
    print(f"  Train Samples: {args.train_samples}")
    print(f"  Val Samples: {args.val_samples}")
    print(f"  Test Samples: {args.test_samples}")
    if args.triplet:
        print(f"  Training Mode: Triplet Loss")
        print(f"  Margin: {args.margin}")
        print(f"  Distance: {'Cosine' if args.use_cosine else 'Euclidean'}")
    else:
        print(f"  Training Mode: Classification")
    print("-" * 70)
    
    # Execute training based on strategy
    if args.gpu_strategy == 'ddp' and args.world_size > 1:
        print("🔥 Starting Distributed Data Parallel training...")
        mp.spawn(distributed_main, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        # Single GPU or DataParallel training
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load data
        train_data, val_data, test_data = load_and_prepare_snli_data(
            args.train_samples, args.val_samples, args.test_samples
        )

        # Create DataLoaders
        train_loader, val_loader, test_loader, model = create_data_loaders(
            train_data, val_data, test_data, args.batch_size, args.max_length, args.triplet
        )
        
        # Configure model for single GPU or DataParallel
        if args.gpu_strategy == 'dp' and args.world_size > 1:
            model, device = get_gpu_strategy(args.gpu_strategy, model, gpu_ids)
        else:
            model = model.to(device)

        print(f"DataLoaders created successfully!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print("-" * 50)

        # Train model
        print("Starting training...")
        if args.triplet:
            # Triplet training
            train_losses, val_losses = train_sentence_bert_triplet(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                log_dir=args.log_dir,
                margin=args.margin,
                use_cosine=args.use_cosine
            )
            val_accuracies = None
        else:
            # Classification training
            train_losses, val_losses, val_accuracies = train_sentence_bert_classification(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                log_dir=args.log_dir
            )

        print("Training completed!")

        # Plot training history
        print("Plotting training history...")
        plot_training_history(train_losses, val_losses, val_accuracies)

        # Evaluate on test set (only for classification)
        if not args.triplet:
            print("\nEvaluating on test set...")
            test_accuracy = evaluate_model(model, test_loader, device)
        else:
            test_accuracy = None

        # Test similarity functionality
        if args.test_similarity:
            test_similarity_functionality(model, device)

        # Save model
        if args.save_model:
            model_save_path = f'sentence_bert_snli_{args.gpu_strategy}_model.pth'
            # Extract actual model if using DataParallel
            model_to_save = model.module if hasattr(model, 'module') else model
            save_dict = {
                'model_state_dict': model_to_save.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'hyperparameters': {
                    'batch_size': args.batch_size,
                    'max_length': args.max_length,
                    'epochs': args.epochs,
                    'learning_rate': args.lr,
                    'train_samples': args.train_samples,
                    'val_samples': args.val_samples,
                    'test_samples': args.test_samples,
                    'training_mode': 'triplet' if args.triplet else 'classification',
                    'gpu_strategy': args.gpu_strategy,
                    'world_size': args.world_size
                }
            }

            if val_accuracies is not None:
                save_dict['val_accuracies'] = val_accuracies
            if test_accuracy is not None:
                save_dict['test_accuracy'] = test_accuracy
            if args.triplet:
                save_dict['margin'] = args.margin
                save_dict['use_cosine'] = args.use_cosine

            torch.save(save_dict, model_save_path)

            print(f"\nModel saved as '{model_save_path}'")
            if test_accuracy is not None:
                print(f"Final Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
