import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Optional
import random
import pickle
import os
from datasets import load_dataset

class PreprocessedTextDataset(Dataset):
    """
    전처리된 데이터를 사용하는 Dataset
    토크나이징이 이미 완료되어 있어 매우 빠름
    """
    def __init__(self, preprocessed_data_path: str):
        print(f"Loading preprocessed data from {preprocessed_data_path}...")
        
        with open(preprocessed_data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.input_ids = self.data['input_ids']
        self.attention_mask = self.data['attention_mask']
        self.token_type_ids = self.data['token_type_ids']
        self.sentences = self.data['sentences']
        self.max_length = self.data['max_length']
        
        print(f"Loaded {len(self.sentences)} preprocessed samples")
        print(f"Data shapes: input_ids={self.input_ids.shape}, "
              f"attention_mask={self.attention_mask.shape}")
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        # 이미 전처리된 데이터를 단순히 반환 (매우 빠름)
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'token_type_ids': self.token_type_ids[idx]
        }
    
    def get_sentence(self, idx):
        """디버깅용 - 원본 문장 반환"""
        return self.sentences[idx]

def load_snli_dataset():
    """
    Load SNLI dataset and filter for entailment (label=0) only
    """
    print("Loading SNLI dataset...")
    dataset = load_dataset('snli')
    
    # Filter for entailment (label=0) only
    train_data = dataset['train'].filter(lambda x: x['label'] == 0)
    validation_data = dataset['validation'].filter(lambda x: x['label'] == 0) 
    test_data = dataset['test'].filter(lambda x: x['label'] == 0)
    
    print(f"Filtered SNLI dataset:")
    print(f"  Train: {len(train_data)} entailment pairs")
    print(f"  Validation: {len(validation_data)} entailment pairs") 
    print(f"  Test: {len(test_data)} entailment pairs")
    
    # Extract sentences from premise and hypothesis
    train_sentences = []
    
    # Add premises
    for item in train_data:
        if item['premise'] and item['premise'].strip():
            train_sentences.append(item['premise'].strip())
    
    # Add hypotheses  
    for item in train_data:
        if item['hypothesis'] and item['hypothesis'].strip():
            train_sentences.append(item['hypothesis'].strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sentences = []
    for sentence in train_sentences:
        if sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
    
    print(f"Extracted {len(unique_sentences)} unique sentences from SNLI entailment data")
    
    return unique_sentences, validation_data, test_data

def create_sample_data(num_samples: int = 1000) -> List[str]:
    """
    Create sample sentences for demonstration
    """
    sample_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of training data.",
        "Transformers have revolutionized the field of NLP.",
        "BERT is a bidirectional encoder representation from transformers.",
        "Contrastive learning learns representations by comparing similar and dissimilar samples.",
        "SimCSE uses dropout as a simple data augmentation technique.",
        "Sentence embeddings capture semantic meaning in vector space.",
        "PyTorch is a popular deep learning framework.",
        "The weather is beautiful today.",
        "I enjoy reading books in my free time.",
        "Technology continues to advance rapidly.",
        "Education is important for personal growth.",
        "Traveling broadens one's perspective on life.",
        "Music has the power to evoke emotions.",
        "Cooking is both an art and a science.",
        "Exercise is essential for maintaining good health.",
        "Friendship is one of life's greatest treasures.",
        "Learning new languages opens doors to different cultures."
    ]
    
    # Generate variations and repeat to reach desired number
    sentences = []
    for i in range(num_samples):
        base_sentence = sample_sentences[i % len(sample_sentences)]
        
        # Add some variation
        variations = [
            base_sentence,
            base_sentence.replace(".", "!"),
            base_sentence.capitalize(),
            f"Today, {base_sentence.lower()}",
            f"In my opinion, {base_sentence.lower()}",
        ]
        
        sentences.append(random.choice(variations))
    
    return sentences

def create_dataloader(
    preprocessed_data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """
    전처리된 데이터를 사용하는 DataLoader 생성
    """
    dataset = PreprocessedTextDataset(preprocessed_data_path)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return dataloader

def prepare_data(config, preprocessed_data_path=None):
    """
    전처리된 데이터를 사용하는 데이터 준비
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    if preprocessed_data_path and os.path.exists(preprocessed_data_path):
        print(f"Using preprocessed data from {preprocessed_data_path}")
        
        # 전처리된 데이터로 DataLoader 생성
        train_dataloader = create_dataloader(
            preprocessed_data_path,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=getattr(config, 'dataloader_num_workers', 0),
            pin_memory=getattr(config, 'pin_memory', torch.cuda.is_available())
        )
        
        return train_dataloader, tokenizer
    else:
        raise FileNotFoundError(
            f"Preprocessed data not found at {preprocessed_data_path}. "
            "Please run 'python preprocess.py' first to create preprocessed data."
        ) 