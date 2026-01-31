import torch
from datasets import load_dataset
from collections import Counter, OrderedDict
import pickle
from config import hyperparameters as hp
from src.vocab import Vocab

def get_tokenizer(dataset):
    counter = Counter()
    for text in dataset['train']['text']:
        if text.strip():
            counter.update(text.strip().split())
    
    # Sort by frequency, then alphabetically for stability
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    
    v = Vocab(ordered_dict, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    return v

def data_process(raw_text_iter, vocab):
    data = [torch.tensor([vocab[token] for token in item.strip().split()], dtype=torch.long) for item in raw_text_iter if item.strip()]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data, bsz, device):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def main():
    print("Loading wikitext-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    print("Building vocabulary...")
    vocab_obj = get_tokenizer(dataset)
    
    print("Processing train, validation, and test sets...")
    train_text = [item['text'] for item in dataset['train']]
    val_text = [item['text'] for item in dataset['validation']]
    test_text = [item['text'] for item in dataset['test']]

    train_data = data_process(train_text, vocab_obj)
    val_data = data_process(val_text, vocab_obj)
    test_data = data_process(test_text, vocab_obj)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Batchifying data...")
    train_data_b = batchify(train_data, hp.BATCH_SIZE, device)
    val_data_b = batchify(val_data, hp.BATCH_SIZE, device) # Use same batch size for simplicity
    test_data_b = batchify(test_data, hp.BATCH_SIZE, device)
    
    print("Saving processed data and vocab...")
    torch.save(train_data_b, hp.TRAIN_DATA_PATH)
    torch.save(val_data_b, hp.VALID_DATA_PATH)
    torch.save(test_data_b, hp.TEST_DATA_PATH)
    
    with open(hp.VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab_obj, f)
        
    print("Data preparation finished.")
    print(f"Vocabulary size: {len(vocab_obj)}")
    print(f"Train data shape: {train_data_b.shape}")
    print(f"Validation data shape: {val_data_b.shape}")
    print(f"Test data shape: {test_data_b.shape}")


if __name__ == "__main__":
    main()
