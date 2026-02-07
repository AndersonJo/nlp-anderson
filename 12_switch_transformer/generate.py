import torch
import pickle
from src.model import SwitchTransformerLM
from src.vocab import Vocab # Import the Vocab class
from config import hyperparameters as hp

# --- Load Model and Vocab ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(hp.VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)

ntokens = len(vocab)

model = SwitchTransformerLM(
    ntokens, hp.D_MODEL, hp.NHEAD, hp.D_FF, 
    hp.NUM_EXPERTS, hp.NUM_LAYERS, hp.DROPOUT
)

# Load state dict and handle torch.compile prefixes
state_dict = torch.load(hp.MODEL_PATH, map_location=device, weights_only=True)

# Remove '_orig_mod.' prefix from keys if present (from torch.compile)
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

# --- Generate Text ---
input_ids = torch.tensor([vocab[token] for token in hp.PROMPT.split()], dtype=torch.long).unsqueeze(1).to(device)
generated_words = list(hp.PROMPT.split())

with torch.no_grad():
    for _ in range(hp.MAX_WORDS):
        output, _ = model(input_ids)
        # Get the logits for the last token
        last_word_logits = output[-1, 0, :]
        
        # Apply temperature scaling
        scaled_logits = last_word_logits / hp.TEMPERATURE
        
        # Get probabilities with softmax
        probabilities = torch.softmax(scaled_logits, dim=0)
        
        # Sample the next word
        next_word_id = torch.multinomial(probabilities, 1).item()
        
        # Stop if <eos> token is generated
        if vocab.get_itos()[next_word_id] == '<eos>':
            break
            
        # Add the new word to the sequence
        generated_words.append(vocab.get_itos()[next_word_id])
        
        # Append the new word id to the input for the next iteration
        next_word_tensor = torch.tensor([[next_word_id]], dtype=torch.long).to(device)
        input_ids = torch.cat([input_ids, next_word_tensor], dim=0)

print("Generated Text:")
print(" ".join(generated_words))
