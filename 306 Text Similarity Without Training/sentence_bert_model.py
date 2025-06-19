import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel


class SentenceBERT(nn.Module):
    """
    Sentence BERT model implementation
    Paper: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
    """
    
    def __init__(self, model_name='bert-base-uncased', embedding_dim=768):
        super(SentenceBERT, self).__init__()
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Pooling layer for sentence embedding
        self.pooling = nn.Linear(embedding_dim, embedding_dim)
        
        # Classification head (SNLI: 3 classes)
        # 3 = u + v + |u-v| (paper implementation)
        self.classifier = nn.Linear(embedding_dim * 3, 3)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Mean pooling: calculate average considering attention mask
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Forward pass for sentence pair classification
        """
        # Encode first sentence
        outputs_1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)
        embeddings_1 = self.mean_pooling(outputs_1.last_hidden_state, attention_mask_1)
        embeddings_1 = self.pooling(embeddings_1)
        embeddings_1 = self.dropout(embeddings_1)
        
        # Encode second sentence
        outputs_2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2)
        embeddings_2 = self.mean_pooling(outputs_2.last_hidden_state, attention_mask_2)
        embeddings_2 = self.pooling(embeddings_2)
        embeddings_2 = self.dropout(embeddings_2)
        
        # Calculate absolute difference between sentence pairs (SNLI paper method: u, v, |u-v|)
        diff = torch.abs(embeddings_1 - embeddings_2)
        
        # Concatenate for classification (paper implementation: u, v, |u-v|)
        combined = torch.cat([embeddings_1, embeddings_2, diff], dim=1)
        
        # Classification (input dimension: 3n → k)
        logits = self.classifier(combined)
        
        return logits, embeddings_1, embeddings_2
    
    def forward_triplet(self, input_ids_a, attention_mask_a, 
                       input_ids_p, attention_mask_p,
                       input_ids_n, attention_mask_n):
        """
        Forward pass for triplet loss
        Paper: max(||sa − sp|| − ||sa − sn|| + ε, 0)
        
        Args:
            input_ids_a, attention_mask_a: anchor sentence
            input_ids_p, attention_mask_p: positive sentence  
            input_ids_n, attention_mask_n: negative sentence
        """
        # Anchor sentence encoding
        outputs_a = self.bert(input_ids=input_ids_a, attention_mask=attention_mask_a)
        embeddings_a = self.mean_pooling(outputs_a.last_hidden_state, attention_mask_a)
        embeddings_a = self.pooling(embeddings_a)
        embeddings_a = self.dropout(embeddings_a)
        
        # Positive sentence encoding
        outputs_p = self.bert(input_ids=input_ids_p, attention_mask=attention_mask_p)
        embeddings_p = self.mean_pooling(outputs_p.last_hidden_state, attention_mask_p)
        embeddings_p = self.pooling(embeddings_p)
        embeddings_p = self.dropout(embeddings_p)
        
        # Negative sentence encoding
        outputs_n = self.bert(input_ids=input_ids_n, attention_mask=attention_mask_n)
        embeddings_n = self.mean_pooling(outputs_n.last_hidden_state, attention_mask_n)
        embeddings_n = self.pooling(embeddings_n)
        embeddings_n = self.dropout(embeddings_n)
        
        return embeddings_a, embeddings_p, embeddings_n
    
    def encode(self, input_ids, attention_mask):
        """
        Encode single sentence to embedding vector (for inference)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        embeddings = self.pooling(embeddings)
        return embeddings


class SNLIDataset(Dataset):
    """
    Custom Dataset class for SNLI dataset
    """
    
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Label mapping
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Tokenize sentence pair
        sentence1 = str(item['premise'])
        sentence2 = str(item['hypothesis'])
        
        # Tokenization
        encoding1 = self.tokenizer(
            sentence1,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            sentence2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Label processing
        label = item['label']
        if label == -1:  # Invalid label in SNLI
            label = 'neutral'  # Set as default
        
        label_id = self.label_map.get(label, 2)  # Default is neutral
        
        return {
            'input_ids_1': encoding1['input_ids'].squeeze(0),
            'attention_mask_1': encoding1['attention_mask'].squeeze(0),
            'input_ids_2': encoding2['input_ids'].squeeze(0),
            'attention_mask_2': encoding2['attention_mask'].squeeze(0),
            'label': torch.tensor(label_id, dtype=torch.long)
        }


class TripletDataset(Dataset):
    """
    Dataset for triplet loss training
    Generate Anchor, Positive, Negative sentence pairs
    """
    
    def __init__(self, dataset, tokenizer, max_length=128, num_triplets_per_anchor=3):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_triplets_per_anchor = num_triplets_per_anchor
        
        # Group sentences by label
        self.sentences_by_label = {}
        for item in dataset:
            label = item['label']
            if label not in self.sentences_by_label:
                self.sentences_by_label[label] = []
            self.sentences_by_label[label].append(item['premise'])
        
        # Available labels
        self.labels = list(self.sentences_by_label.keys())
        
    def __len__(self):
        return len(self.dataset) * self.num_triplets_per_anchor
    
    def __getitem__(self, idx):
        # Select anchor from original dataset
        anchor_item = self.dataset[idx % len(self.dataset)]
        anchor_label = anchor_item['label']
        anchor_sentence = str(anchor_item['premise'])
        
        # Positive: different sentence with same label
        positive_sentences = [s for s in self.sentences_by_label[anchor_label] 
                            if s != anchor_sentence]
        if positive_sentences:
            positive_sentence = np.random.choice(positive_sentences)
        else:
            positive_sentence = anchor_sentence  # fallback
        
        # Negative: sentence with different label
        other_labels = [l for l in self.labels if l != anchor_label]
        if other_labels:
            negative_label = np.random.choice(other_labels)
            negative_sentence = np.random.choice(self.sentences_by_label[negative_label])
        else:
            negative_sentence = anchor_sentence  # fallback
        
        # Tokenization
        anchor_encoding = self.tokenizer(
            anchor_sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        positive_encoding = self.tokenizer(
            positive_sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        negative_encoding = self.tokenizer(
            negative_sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids_a': anchor_encoding['input_ids'].squeeze(0),
            'attention_mask_a': anchor_encoding['attention_mask'].squeeze(0),
            'input_ids_p': positive_encoding['input_ids'].squeeze(0),
            'attention_mask_p': positive_encoding['attention_mask'].squeeze(0),
            'input_ids_n': negative_encoding['input_ids'].squeeze(0),
            'attention_mask_n': negative_encoding['attention_mask'].squeeze(0),
            'anchor_label': anchor_label
        }


def triplet_loss(embeddings_a, embeddings_p, embeddings_n, margin=1.0):
    """
    Calculate triplet loss
    Paper: max(||sa − sp|| − ||sa − sn|| + ε, 0)
    
    Args:
        embeddings_a: anchor sentence embeddings
        embeddings_p: positive sentence embeddings
        embeddings_n: negative sentence embeddings
        margin: margin ε (set to 1 in paper)
    
    Returns:
        triplet_loss: calculated triplet loss
    """
    # Calculate Euclidean distance
    # ||sa - sp||
    dist_pos = torch.norm(embeddings_a - embeddings_p, p=2, dim=1)
    
    # ||sa - sn||
    dist_neg = torch.norm(embeddings_a - embeddings_n, p=2, dim=1)
    
    # max(||sa − sp|| − ||sa − sn|| + ε, 0)
    triplet_loss = torch.clamp(dist_pos - dist_neg + margin, min=0.0)
    
    return triplet_loss.mean()


def cosine_triplet_loss(embeddings_a, embeddings_p, embeddings_n, margin=0.5):
    """
    Triplet loss using cosine distance (alternative)
    Some implementations use cosine distance
    """
    # Calculate cosine distance (1 - cosine similarity)
    cos_sim_pos = F.cosine_similarity(embeddings_a, embeddings_p, dim=1)
    cos_sim_neg = F.cosine_similarity(embeddings_a, embeddings_n, dim=1)
    
    # Distance = 1 - similarity
    dist_pos = 1 - cos_sim_pos
    dist_neg = 1 - cos_sim_neg
    
    # max(dist_pos - dist_neg + margin, 0)
    triplet_loss = torch.clamp(dist_pos - dist_neg + margin, min=0.0)
    
    return triplet_loss.mean()


def compute_similarity(embeddings1, embeddings2):
    """
    Calculate cosine similarity
    """
    return F.cosine_similarity(embeddings1, embeddings2, dim=1)


# TensorBoard utility functions
if __name__ == "__main__":
    # Print help when script is run directly
    print("Sentence BERT Model")
    print("=" * 50)
    print("Available components:")
    print("- SentenceBERT: Main model class")
    print("- SNLIDataset: Dataset for SNLI classification")
    print("- TripletDataset: Dataset for triplet loss training")
    print("- triplet_loss: Triplet loss function")
    print("- cosine_triplet_loss: Cosine triplet loss function")
    print("- compute_similarity: Cosine similarity function")
    print()
    print("This module contains only model-related code.")
    print("Training functions are in train_sentence_bert.py") 