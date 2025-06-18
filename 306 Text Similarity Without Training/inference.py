import torch
import torch.nn.functional as F
from sentence_bert_model import SentenceBERT
import numpy as np

class SentenceBERTInference:
    """
    학습된 Sentence BERT 모델을 사용한 추론 클래스
    """
    
    def __init__(self, model_path='sentence_bert_snli_model.pth', model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 초기화
        self.model = SentenceBERT(model_name)
        
        # 저장된 모델 로드
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Test accuracy: {checkpoint.get('test_accuracy', 'N/A')}")
    
    def predict_relationship(self, sentence1, sentence2):
        """
        두 문장의 관계를 예측 (entailment, contradiction, neutral)
        """
        # 토크나이징
        encoding1 = self.model.tokenizer(
            sentence1,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        encoding2 = self.model.tokenizer(
            sentence2,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # 디바이스로 이동
        input_ids_1 = encoding1['input_ids'].to(self.device)
        attention_mask_1 = encoding1['attention_mask'].to(self.device)
        input_ids_2 = encoding2['input_ids'].to(self.device)
        attention_mask_2 = encoding2['attention_mask'].to(self.device)
        
        # 예측
        with torch.no_grad():
            logits, embedding1, embedding2 = self.model(
                input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
            )
            
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 결과 매핑
        label_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
        relationship = label_map[predicted_class]
        
        return {
            'relationship': relationship,
            'confidence': confidence,
            'probabilities': {
                'entailment': probabilities[0][0].item(),
                'contradiction': probabilities[0][1].item(),
                'neutral': probabilities[0][2].item()
            },
            'embeddings': {
                'sentence1': embedding1.cpu().numpy(),
                'sentence2': embedding2.cpu().numpy()
            }
        }
    
    def compute_similarity(self, sentence1, sentence2):
        """
        두 문장의 코사인 유사도 계산
        """
        result = self.predict_relationship(sentence1, sentence2)
        
        embedding1 = result['embeddings']['sentence1']
        embedding2 = result['embeddings']['sentence2']
        
        # 코사인 유사도 계산
        similarity = F.cosine_similarity(
            torch.tensor(embedding1), 
            torch.tensor(embedding2), 
            dim=1
        ).item()
        
        return {
            'similarity': similarity,
            'relationship': result['relationship'],
            'confidence': result['confidence']
        }
    
    def encode_sentence(self, sentence):
        """
        단일 문장을 임베딩 벡터로 인코딩
        """
        encoding = self.model.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode(input_ids, attention_mask)
        
        return embedding.cpu().numpy()


def demo_inference():
    """
    추론 데모 함수
    """
    print("Loading Sentence BERT model...")
    
    try:
        # 모델 로드 (모델이 학습되어 있다면)
        inference_model = SentenceBERTInference()
    except FileNotFoundError:
        print("Trained model not found. Please train the model first.")
        print("Creating a new model for demonstration...")
        inference_model = SentenceBERTInference(model_path=None)
    
    # 예제 문장 쌍들
    example_pairs = [
        ("A soccer game with multiple males playing.", "Some men are playing a sport."),
        ("A black race car starts up in front of a crowd of people.", "A man is driving down a lonely road."),
        ("An older and younger man smiling.", "Two men are smiling and laughing at the cats playing on the floor."),
        ("A man inspects the uniform of a figure in some East Asian country.", "The man is sleeping."),
        ("A person on a horse jumps over a broken down airplane.", "A person is training his horse for a competition.")
    ]
    
    print("\n=== Sentence Relationship Prediction ===")
    for i, (sentence1, sentence2) in enumerate(example_pairs, 1):
        print(f"\nExample {i}:")
        print(f"Sentence 1: {sentence1}")
        print(f"Sentence 2: {sentence2}")
        
        result = inference_model.predict_relationship(sentence1, sentence2)
        
        print(f"Relationship: {result['relationship']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.4f}")
        
        # 유사도도 계산
        similarity_result = inference_model.compute_similarity(sentence1, sentence2)
        print(f"Cosine Similarity: {similarity_result['similarity']:.4f}")
        print("-" * 80)
    
    # 단일 문장 인코딩 예제
    print("\n=== Single Sentence Encoding ===")
    test_sentence = "A man is playing guitar on stage."
    embedding = inference_model.encode_sentence(test_sentence)
    print(f"Sentence: {test_sentence}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 values): {embedding[0][:10]}")


if __name__ == "__main__":
    demo_inference() 