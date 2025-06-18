import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class SentenceBERT(nn.Module):
    """
    Sentence BERT 모델 구현
    논문: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
    """
    
    def __init__(self, model_name='bert-base-uncased', embedding_dim=768):
        super(SentenceBERT, self).__init__()
        
        # BERT 인코더
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Sentence embedding을 위한 pooling layer
        self.pooling = nn.Linear(embedding_dim, embedding_dim)
        
        # 분류를 위한 head (SNLI: 3클래스)
        self.classifier = nn.Linear(embedding_dim * 3, 3)  # 3 = entailment, contradiction, neutral
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Mean pooling: attention mask를 고려한 평균 계산
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Forward pass for sentence pair classification
        """
        # 첫 번째 문장 인코딩
        outputs_1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)
        embeddings_1 = self.mean_pooling(outputs_1.last_hidden_state, attention_mask_1)
        embeddings_1 = self.pooling(embeddings_1)
        embeddings_1 = self.dropout(embeddings_1)
        
        # 두 번째 문장 인코딩
        outputs_2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2)
        embeddings_2 = self.mean_pooling(outputs_2.last_hidden_state, attention_mask_2)
        embeddings_2 = self.pooling(embeddings_2)
        embeddings_2 = self.dropout(embeddings_2)
        
        # 문장 쌍의 차이와 곱 계산 (SNLI 논문 방식)
        diff = torch.abs(embeddings_1 - embeddings_2)
        product = embeddings_1 * embeddings_2
        
        # Concatenate for classification
        combined = torch.cat([embeddings_1, embeddings_2, diff], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits, embeddings_1, embeddings_2
    
    def encode(self, input_ids, attention_mask):
        """
        단일 문장을 인코딩하는 메서드 (추론 시 사용)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        embeddings = self.pooling(embeddings)
        return embeddings


class SNLIDataset(Dataset):
    """
    SNLI 데이터셋을 위한 커스텀 Dataset 클래스
    """
    
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 레이블 매핑
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 문장 쌍 토크나이징
        sentence1 = str(item['premise'])
        sentence2 = str(item['hypothesis'])
        
        # 토크나이징
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
        
        # 레이블 처리
        label = item['label']
        if label == -1:  # SNLI에서 -1은 무효한 레이블
            label = 'neutral'  # 기본값으로 설정
        
        label_id = self.label_map.get(label, 2)  # 기본값은 neutral
        
        return {
            'input_ids_1': encoding1['input_ids'].squeeze(0),
            'attention_mask_1': encoding1['attention_mask'].squeeze(0),
            'input_ids_2': encoding2['input_ids'].squeeze(0),
            'attention_mask_2': encoding2['attention_mask'].squeeze(0),
            'label': torch.tensor(label_id, dtype=torch.long)
        }


def train_sentence_bert(model, train_loader, val_loader, device, epochs=3, lr=2e-5):
    """
    Sentence BERT 모델 학습 함수
    """
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Training loop
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in train_pbar:
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
            
            train_pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
        
        train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for batch in val_pbar:
                input_ids_1 = batch['input_ids_1'].to(device)
                attention_mask_1 = batch['attention_mask_1'].to(device)
                input_ids_2 = batch['input_ids_2'].to(device)
                attention_mask_2 = batch['attention_mask_2'].to(device)
                labels = batch['label'].to(device)
                
                logits, _, _ = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                loss = criterion(logits, labels)
                
                total_val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'loss': loss.item(), 'acc': val_correct/val_total})
        
        val_loss = total_val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        print('-' * 50)
    
    return train_losses, val_losses, val_accuracies


def plot_training_history(train_losses, val_losses, val_accuracies):
    """
    학습 히스토리를 시각화하는 함수
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(val_accuracies, label='Val Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def compute_similarity(embeddings1, embeddings2):
    """
    코사인 유사도 계산
    """
    return F.cosine_similarity(embeddings1, embeddings2, dim=1)


if __name__ == "__main__":
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 초기화
    model = SentenceBERT()
    print("Model initialized successfully!")
    
    # 데이터셋 로드 (이 부분은 별도로 구현 필요)
    print("Please load your SNLI dataset and create DataLoaders") 