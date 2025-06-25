import torch
import numpy as np
import random
import os
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import json

def set_seed(seed: int):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, tokenizer, output_dir: str, step: int = None):
    """
    Save model and tokenizer
    """
    if step is not None:
        save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    else:
        save_dir = output_dir
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model.simcse.bert.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save model configuration
    config_dict = {
        'pooler_type': model.simcse.pooler_type,
        'temp': model.simcse.temp,
        'model_name': tokenizer.name_or_path
    }
    
    with open(os.path.join(save_dir, 'simcse_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Model saved to {save_dir}")

def load_model(model_path: str, device):
    """
    Load trained SimCSE model
    """
    from model import SimCSE
    from transformers import AutoTokenizer
    
    # Load configuration
    config_path = os.path.join(model_path, 'simcse_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model = SimCSE(
            model_name=model_path,
            pooler_type=config.get('pooler_type', 'cls'),
            temp=config.get('temp', 0.05)
        )
    else:
        model = SimCSE(model_name=model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    return model, tokenizer

def encode_sentences(model, tokenizer, sentences: List[str], device, batch_size: int = 32, max_length: int = 128):
    """
    Encode sentences into embeddings
    """
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_sentences,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            # Get embeddings
            if hasattr(model, 'encode'):
                embeddings = model.encode(**encoded)
            else:
                embeddings = model(**encoded)
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.concatenate(all_embeddings, axis=0)

def evaluate_similarity(model, tokenizer, device, test_pairs: List[Tuple[str, str, float]] = None):
    """
    Evaluate model on sentence similarity task
    test_pairs: List of (sentence1, sentence2, similarity_score)
    """
    if test_pairs is None:
        # Create simple test pairs for demonstration
        test_pairs = [
            ("The cat is sleeping.", "A cat is taking a nap.", 0.8),
            ("I love programming.", "I hate coding.", 0.2),
            ("The weather is nice today.", "Today has beautiful weather.", 0.9),
            ("Machine learning is fun.", "Deep learning is challenging.", 0.6),
            ("I went to the store.", "She bought groceries.", 0.4),
        ]
    
    sentences1 = [pair[0] for pair in test_pairs]
    sentences2 = [pair[1] for pair in test_pairs]
    true_scores = [pair[2] for pair in test_pairs]
    
    # Get embeddings
    embeddings1 = encode_sentences(model, tokenizer, sentences1, device)
    embeddings2 = encode_sentences(model, tokenizer, sentences2, device)
    
    # Compute cosine similarities
    similarities = []
    for emb1, emb2 in zip(embeddings1, embeddings2):
        sim = cosine_similarity([emb1], [emb2])[0][0]
        similarities.append(sim)
    
    # Compute Spearman correlation
    correlation, p_value = spearmanr(true_scores, similarities)
    
    print(f"Spearman correlation: {correlation:.4f} (p-value: {p_value:.4f})")
    print("\nExample similarities:")
    for i, (s1, s2, true_sim, pred_sim) in enumerate(zip(sentences1, sentences2, true_scores, similarities)):
        print(f"{i+1}. True: {true_sim:.2f}, Predicted: {pred_sim:.4f}")
        print(f"   Sentence 1: {s1}")
        print(f"   Sentence 2: {s2}")
        print()
    
    return correlation, similarities

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def format_time(elapsed):
    """
    Format time in human readable format
    """
    elapsed_rounded = int(round(elapsed))
    return f"{elapsed_rounded // 3600:02d}:{(elapsed_rounded % 3600) // 60:02d}:{elapsed_rounded % 60:02d}"

def create_optimizer_and_scheduler(model, config, num_training_steps):
    """
    Create optimizer and learning rate scheduler
    """
    from transformers import get_linear_schedule_with_warmup
    
    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    
    # Prepare scheduler
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler 