import torch
import argparse
from utils import load_model, encode_sentences, evaluate_similarity
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Evaluate SimCSE model')
    parser.add_argument('--model_path', type=str, default='./output', 
                       help='Path to trained model')
    parser.add_argument('--sentences', type=str, nargs='+', 
                       default=['The cat is sleeping.', 'A cat is taking a nap.', 'The dog is running.'],
                       help='Sentences to encode and compare')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    try:
        model, tokenizer = load_model(args.model_path, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Encode sentences
    print(f"\nEncoding {len(args.sentences)} sentences...")
    embeddings = encode_sentences(model, tokenizer, args.sentences, device)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Compute pairwise similarities
    print("\nPairwise cosine similarities:")
    similarities = cosine_similarity(embeddings)
    
    for i, sent1 in enumerate(args.sentences):
        for j, sent2 in enumerate(args.sentences):
            if i <= j:
                sim = similarities[i][j]
                print(f"Sentence {i+1} <-> Sentence {j+1}: {sim:.4f}")
                print(f"  '{sent1}'")
                if i != j:
                    print(f"  '{sent2}'")
                print()
    
    # Evaluate on predefined similarity task
    print("Evaluating on predefined similarity pairs...")
    evaluate_similarity(model, tokenizer, device)

if __name__ == "__main__":
    main() 