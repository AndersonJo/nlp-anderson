#!/usr/bin/env python3
"""
Preprocessing script for SimCSE
미리 데이터를 토크나이징하여 저장하는 스크립트
"""

import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from data_loader import load_snli_dataset, create_sample_data
import argparse
import time

def preprocess_and_save(sentences, tokenizer, max_length, output_dir, split_name):
    """
    문장들을 토크나이징하여 저장
    """
    print(f"Preprocessing {len(sentences)} sentences for {split_name}...")
    
    # 결과를 저장할 리스트들
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    
    # 배치 단위로 토크나이징 (메모리 효율성)
    batch_size = 1000
    
    for i in tqdm(range(0, len(sentences), batch_size), desc=f"Tokenizing {split_name}"):
        batch_sentences = sentences[i:i + batch_size]
        
        # 배치 토크나이징
        encoded_batch = tokenizer(
            batch_sentences,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # CPU 텐서로 변환하여 저장
        input_ids_list.append(encoded_batch['input_ids'].cpu())
        attention_mask_list.append(encoded_batch['attention_mask'].cpu())
        
        # token_type_ids 처리
        if 'token_type_ids' in encoded_batch:
            token_type_ids_list.append(encoded_batch['token_type_ids'].cpu())
        else:
            token_type_ids_list.append(torch.zeros_like(encoded_batch['input_ids']).cpu())
    
    # 모든 배치를 연결
    all_input_ids = torch.cat(input_ids_list, dim=0)
    all_attention_mask = torch.cat(attention_mask_list, dim=0)
    all_token_type_ids = torch.cat(token_type_ids_list, dim=0)
    
    print(f"Final shapes - Input IDs: {all_input_ids.shape}, "
          f"Attention Mask: {all_attention_mask.shape}, "
          f"Token Type IDs: {all_token_type_ids.shape}")
    
    # 저장할 데이터
    preprocessed_data = {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'token_type_ids': all_token_type_ids,
        'sentences': sentences,  # 원본 문장도 함께 저장 (디버깅용)
        'max_length': max_length,
        'tokenizer_name': tokenizer.name_or_path
    }
    
    # 저장
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{split_name}_preprocessed.pkl')
    
    print(f"Saving preprocessed data to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    # 메모리 정보
    file_size = os.path.getsize(save_path) / 1024**2  # MB
    print(f"Saved {len(sentences)} samples to {save_path} ({file_size:.2f} MB)")
    
    return save_path

def load_preprocessed_data(file_path):
    """
    전처리된 데이터 로드
    """
    print(f"Loading preprocessed data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data['sentences'])} preprocessed samples")
    print(f"Max length: {data['max_length']}")
    print(f"Tokenizer: {data['tokenizer_name']}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Preprocess SimCSE data')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Model name for tokenizer')
    parser.add_argument('--max_length', type=int, default=64,
                       help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='./preprocessed_data',
                       help='Output directory for preprocessed data')
    parser.add_argument('--use_snli', action='store_true', default=True,
                       help='Use SNLI dataset')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to preprocess')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SimCSE Data Preprocessing")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Max length: {args.max_length}")
    print(f"Output directory: {args.output_dir}")
    print(f"Use SNLI: {args.use_snli}")
    print("=" * 60)
    
    # 토크나이저 로드
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 데이터 로드
    start_time = time.time()
    
    if args.use_snli:
        print("Loading SNLI dataset...")
        train_sentences, val_data, test_data = load_snli_dataset()
        
        if args.max_samples and len(train_sentences) > args.max_samples:
            train_sentences = train_sentences[:args.max_samples]
            print(f"Limited to {len(train_sentences)} samples")
    else:
        print("Creating sample data...")
        train_sentences = create_sample_data(args.max_samples or 1000)
    
    data_load_time = time.time() - start_time
    print(f"Data loading completed in {data_load_time:.2f} seconds")
    
    # 전처리 및 저장
    preprocessing_start = time.time()
    
    save_path = preprocess_and_save(
        train_sentences, 
        tokenizer, 
        args.max_length, 
        args.output_dir, 
        'train'
    )
    
    preprocessing_time = time.time() - preprocessing_start
    print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
    
    # 검증용 로드 테스트
    print("\nTesting data loading...")
    load_start = time.time()
    loaded_data = load_preprocessed_data(save_path)
    load_time = time.time() - load_start
    print(f"Data loading test completed in {load_time:.4f} seconds")
    
    # 샘플 확인
    print("\nSample verification:")
    sample_idx = 0
    original_sentence = loaded_data['sentences'][sample_idx]
    input_ids = loaded_data['input_ids'][sample_idx]
    decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
    
    print(f"Original: {original_sentence}")
    print(f"Decoded:  {decoded}")
    print(f"Input IDs shape: {input_ids.shape}")
    
    print(f"\nPreprocessing completed successfully!")
    print(f"Preprocessed data saved to: {save_path}")

if __name__ == "__main__":
    main() 