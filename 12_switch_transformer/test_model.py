import torch
import pickle
import math
from src.model import SwitchTransformerLM
from config import hyperparameters as hp

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load vocab and model
    with open(hp.VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    
    ntokens = len(vocab)
    model = SwitchTransformerLM(
        ntokens, hp.D_MODEL, hp.NHEAD, hp.D_FF, 
        hp.NUM_EXPERTS, hp.NUM_LAYERS, hp.DROPOUT
    )
    
    # Load state dict and handle torch.compile prefixes
    state_dict = torch.load(hp.MODEL_PATH, map_location=device, weights_only=True)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            cleaned_key = key[len('_orig_mod.'):]
            cleaned_state_dict[cleaned_key] = value
        else:
            cleaned_state_dict[key] = value
    
    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {ntokens}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test different prompts
    test_prompts = [
        "The",
        "I am",
        "It was",
        "She said",
        "They went"
    ]
    
    print("\n" + "="*50)
    print("TESTING DIFFERENT PROMPTS")
    print("="*50)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        try:
            # Tokenize input
            input_ids = torch.tensor([vocab[token] for token in prompt.split()], dtype=torch.long).unsqueeze(1).to(device)
            
            with torch.no_grad():
                output, lb_loss = model(input_ids)
                
                # Get logits for last token
                last_word_logits = output[-1, 0, :]
                
                # Get top 5 predictions
                top_k = 5
                top_probs, top_indices = torch.topk(torch.softmax(last_word_logits, dim=0), top_k)
                
                print(f"  Load balancing loss: {lb_loss.item():.4f}")
                print(f"  Top {top_k} next word predictions:")
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    word = vocab.get_itos()[idx.item()]
                    print(f"    {i+1}. '{word}' (prob: {prob.item():.4f})")
                    
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test model capacity
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE INFO")
    print("="*50)
    
    # Count parameters by component
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if 'expert' in name:
            print(f"{name}: {param_count:,} params")
    
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test expert utilization
    print("\n" + "="*50)
    print("EXPERT UTILIZATION TEST")
    print("="*50)
    
    test_text = "The quick brown fox jumps over the lazy dog"
    try:
        input_ids = torch.tensor([vocab[token] for token in test_text.split()], dtype=torch.long).unsqueeze(1).to(device)
        
        with torch.no_grad():
            output, lb_loss = model(input_ids)
            print(f"Input: '{test_text}'")
            print(f"Load balancing loss: {lb_loss.item():.4f}")
            print(f"Output shape: {output.shape}")
            
    except Exception as e:
        print(f"Error in expert utilization test: {e}")

if __name__ == "__main__":
    test_model()