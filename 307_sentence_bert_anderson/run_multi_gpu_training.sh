#!/bin/bash

# Multi-GPU Sentence BERT Training Script
# This script provides different strategies for training with 8 GPUs

echo "🚀 Sentence BERT Multi-GPU Training Script"
echo "=========================================="

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits

# Set environment variables for optimal performance
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0

# Strategy 1: Distributed Data Parallel (DDP) - RECOMMENDED for 8 GPUs
echo ""
echo "🔥 Strategy 1: Distributed Data Parallel (DDP)"
echo "==============================================="
echo "Best performance and scalability for 8 GPUs"
echo "Each GPU gets its own process and gradients are synchronized"
echo ""

# DDP Training command
python train_sentence_bert.py \
    --gpu_strategy ddp \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --batch_size 64 \
    --epochs 5 \
    --lr 2e-5 \
    --max_length 128 \
    --use_amp \
    --triplet \
    --margin 1.0 \
    --train_samples 50000 \
    --val_samples 5000 \
    --test_samples 5000

echo ""
echo "✅ DDP Training completed!"

# Optional: Strategy 2 - DataParallel (DP)
echo ""
echo "📦 Strategy 2: DataParallel (DP) - Alternative approach"
echo "======================================================="
echo "Simpler but potentially slower than DDP"
echo "Single process manages all GPUs"
echo ""

# Uncomment to run DataParallel training
# python train_sentence_bert.py \
#     --gpu_strategy dp \
#     --gpu_ids "0,1,2,3,4,5,6,7" \
#     --batch_size 32 \
#     --epochs 3 \
#     --lr 2e-5 \
#     --max_length 128 \
#     --use_amp \
#     --triplet \
#     --margin 1.0 \
#     --train_samples 20000 \
#     --val_samples 2000 \
#     --test_samples 2000

echo ""
echo "🎯 Training Tips for 8 GPUs:"
echo "=============================="
echo "1. DDP is usually faster than DP for 8 GPUs"
echo "2. Increase batch size proportionally (8x base batch size)"
echo "3. Consider gradient accumulation for larger effective batch sizes"
echo "4. Mixed precision (AMP) saves memory and speeds up training"
echo "5. Monitor GPU utilization with nvidia-smi or TensorBoard"
echo ""

echo "📊 Performance Monitoring:"
echo "=========================="
echo "1. TensorBoard: tensorboard --logdir=runs/"
echo "2. GPU Monitor: watch -n 1 nvidia-smi"
echo "3. NCCL Debug: Check logs for communication efficiency"
echo ""

echo "🎉 Multi-GPU training script completed!" 