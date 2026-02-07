#!/usr/bin/env python3
"""
SimCSE Demo Script
Interactive demonstration of the trained SimCSE model
"""

import torch
import numpy as np
from transformers import AutoTokenizer
from utils import load_model, encode_sentences
import argparse
import os

def load_trained_model(model_path, device):
    """
    Load trained SimCSE model
    """
    print(f"Loading model from {model_path}...")
    
    # Use utils.load_model function
    model, tokenizer = load_model(model_path, device)
    
    print("Model loaded successfully!")
    return model, tokenizer

def compute_similarity_matrix(sentences, model, tokenizer, device):
    """
    Compute similarity matrix for given sentences
    """
    embeddings = encode_sentences(model, tokenizer, sentences, device)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            cos_sim = torch.nn.functional.cosine_similarity(
                torch.tensor(embeddings[i]).unsqueeze(0),
                torch.tensor(embeddings[j]).unsqueeze(0)
            ).item()
            similarity_matrix[i][j] = cos_sim
    
    return similarity_matrix

def demo_similarity(model, tokenizer, device):
    """
    Interactive similarity demo
    """
    print("\n" + "="*60)
    print("🔍 Sentence Similarity Demo")
    print("="*60)
    print("Enter sentences to compute their similarities.")
    print("Type 'quit' to exit this demo.")
    
    sentences = []
    
    while True:
        sentence = input(f"\nEnter sentence {len(sentences)+1} (or 'quit'): ").strip()
        
        if sentence.lower() == 'quit':
            break
        
        if sentence:
            sentences.append(sentence)
            print(f"Added: {sentence}")
        
        if len(sentences) >= 2:
            choice = input("\nAdd more sentences? (y/n): ").strip().lower()
            if choice != 'y':
                break
    
    if len(sentences) < 2:
        print("Need at least 2 sentences for similarity computation.")
        return
    
    print(f"\nComputing similarities for {len(sentences)} sentences...")
    similarity_matrix = compute_similarity_matrix(sentences, model, tokenizer, device)
    
    # Display results
    print("\n" + "="*60)
    print("📊 Similarity Matrix")
    print("="*60)
    
    # Print header
    print(f"{'':>3}", end="")
    for i in range(len(sentences)):
        print(f"{i+1:>8}", end="")
    print()
    
    # Print matrix
    for i in range(len(sentences)):
        print(f"{i+1:>2}:", end="")
        for j in range(len(sentences)):
            print(f"{similarity_matrix[i][j]:>8.3f}", end="")
        print()
    
    # Print sentences
    print("\nSentences:")
    for i, sentence in enumerate(sentences):
        print(f"{i+1}. {sentence}")
    
    # Highlight most similar pairs
    print("\n🔥 Most Similar Pairs:")
    pairs = []
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            pairs.append((i, j, similarity_matrix[i][j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, (idx1, idx2, sim) in enumerate(pairs[:3]):
        print(f"{i+1}. Similarity: {sim:.3f}")
        print(f"   '{sentences[idx1]}'")
        print(f"   '{sentences[idx2]}'")

def demo_clustering(model, tokenizer, device):
    """
    Sentence clustering demo
    """
    print("\n" + "="*60)
    print("🎯 Sentence Clustering Demo")
    print("="*60)
    
    # Predefined example sentences
    example_sentences = [
        "The cat is sleeping on the couch.",
        "A feline is resting on the sofa.",
        "The dog is running in the park.",
        "A canine is jogging in the garden.",
        "I love machine learning.",
        "Artificial intelligence is fascinating.",
        "The weather is beautiful today.",
        "It's a gorgeous day outside.",
        "Python is a programming language.",
        "Coding in Python is enjoyable."
    ]
    
    print("Using example sentences for clustering:")
    for i, sentence in enumerate(example_sentences):
        print(f"{i+1:>2}. {sentence}")
    
    print("\nComputing embeddings...")
    embeddings = encode_sentences(model, tokenizer, example_sentences, device)
    
    # Simple clustering based on similarity threshold
    threshold = 0.7
    clusters = []
    used = set()
    
    for i in range(len(example_sentences)):
        if i in used:
            continue
        
        cluster = [i]
        used.add(i)
        
        for j in range(i+1, len(example_sentences)):
            if j in used:
                continue
            
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(embeddings[i]).unsqueeze(0),
                torch.tensor(embeddings[j]).unsqueeze(0)
            ).item()
            
            if similarity > threshold:
                cluster.append(j)
                used.add(j)
        
        clusters.append(cluster)
    
    print(f"\n📊 Clusters (similarity threshold: {threshold}):")
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i+1}:")
        for idx in cluster:
            print(f"  • {example_sentences[idx]}")

def main():
    parser = argparse.ArgumentParser(description='SimCSE Demo')
    parser.add_argument('--model_path', type=str, default='./output',
                       help='Path to trained model')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🚀 SimCSE Demo")
    print("="*50)
    print(f"Device: {device}")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found at {args.model_path}")
        print("Please train a model first by running: python train.py")
        return
    
    # Load model
    try:
        model, tokenizer = load_trained_model(args.model_path, device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Main demo loop
    while True:
        print("\n" + "="*50)
        print("🎮 Demo Options")
        print("="*50)
        print("1. Sentence Similarity")
        print("2. Sentence Clustering")
        print("3. Quick Test")
        print("4. Quit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            demo_similarity(model, tokenizer, device)
        elif choice == '2':
            demo_clustering(model, tokenizer, device)
        elif choice == '3':
            # Quick test with predefined sentences
            test_sentences = [
                "The cat is sleeping.",
                "A cat is taking a nap.",
                "The dog is running.",
                "I love programming.",
                "Coding is fun."
            ]
            
            print("\n🧪 Quick Test with predefined sentences:")
            similarity_matrix = compute_similarity_matrix(test_sentences, model, tokenizer, device)
            
            for i, sentence in enumerate(test_sentences):
                print(f"{i+1}. {sentence}")
            
            print(f"\nHighest similarity: {np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]):.3f}")
            print(f"Lowest similarity: {np.min(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]):.3f}")
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Please try again.")

if __name__ == "__main__":
    main() 