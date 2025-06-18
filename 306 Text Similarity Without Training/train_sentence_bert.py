import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_bert_model import SentenceBERT, SNLIDataset, train_sentence_bert, plot_training_history
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_prepare_snli_data():
    """
    SNLI 데이터를 로드하고 전처리하는 함수
    """
    print("Loading SNLI dataset...")
    
    # SNLI 데이터셋 로드
    snli = load_dataset("snli")
    
    # 데이터 필터링 (유효한 레이블만)
    def filter_valid_data(example):
        return example['label'] != -1
    
    # 각 스플릿에서 유효한 데이터만 필터링
    train_data = snli['train'].filter(filter_valid_data)
    validation_data = snli['validation'].filter(filter_valid_data)
    test_data = snli['test'].filter(filter_valid_data)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(validation_data)}")
    print(f"Test samples: {len(test_data)}")
    
    return train_data, validation_data, test_data

def create_data_loaders(train_data, val_data, test_data, batch_size=16, max_length=128):
    """
    DataLoader 생성
    """
    # 모델 초기화 (토크나이저를 위해)
    model = SentenceBERT()
    tokenizer = model.tokenizer
    
    # Dataset 생성
    train_dataset = SNLIDataset(train_data, tokenizer, max_length)
    val_dataset = SNLIDataset(val_data, tokenizer, max_length)
    test_dataset = SNLIDataset(test_data, tokenizer, max_length)
    
    # DataLoader 생성
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
    테스트 데이터에서 모델 성능 평가
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
            
            # 클래스별 정확도 계산
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

def main():
    """
    메인 학습 함수
    """
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 하이퍼파라미터
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    print(f"Hyperparameters:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Max Length: {MAX_LENGTH}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print("-" * 50)
    
    # 데이터 로드
    train_data, val_data, test_data = load_and_prepare_snli_data()
    
    # DataLoader 생성
    train_loader, val_loader, test_loader, model = create_data_loaders(
        train_data, val_data, test_data, BATCH_SIZE, MAX_LENGTH
    )
    
    print(f"DataLoaders created successfully!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print("-" * 50)
    
    # 모델 학습
    print("Starting training...")
    train_losses, val_losses, val_accuracies = train_sentence_bert(
        model, train_loader, val_loader, device, EPOCHS, LEARNING_RATE
    )
    
    # 학습 히스토리 시각화
    plot_training_history(train_losses, val_losses, val_accuracies)
    
    # 테스트 성능 평가
    print("\nEvaluating on test set...")
    test_accuracy = evaluate_model(model, test_loader, device)
    
    # 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_accuracy,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'max_length': MAX_LENGTH,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE
        }
    }, 'sentence_bert_snli_model.pth')
    
    print(f"\nModel saved as 'sentence_bert_snli_model.pth'")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main() 