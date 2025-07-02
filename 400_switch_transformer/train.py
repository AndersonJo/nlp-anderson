import math
import pickle
import time

import torch
import torch.nn as nn

from config import hyperparameters as hp
from src.model import SwitchTransformerLM


def get_batch(source, i, bptt):
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int
    Returns:
        tuple (data, target) where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def train_one_epoch(model, train_data, criterion, optimizer, scheduler, scaler, ntokens, epoch):
    """Trains the model for one epoch with AMP."""
    model.train()
    total_loss = 0.
    total_main_loss = 0.
    total_lb_loss = 0.
    start_time = time.time()
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, hp.BPTT)):
        data, targets = get_batch(train_data, i, hp.BPTT)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            output, lb_loss = model(data)
            main_loss = criterion(output.view(-1, ntokens), targets)
            loss = main_loss + hp.LOAD_BALANCING_LOSS_WEIGHT * lb_loss
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_lb_loss += lb_loss.item()

        if batch % hp.LOG_INTERVAL == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / hp.LOG_INTERVAL
            cur_loss = total_loss / hp.LOG_INTERVAL
            cur_main_loss = total_main_loss / hp.LOG_INTERVAL
            cur_lb_loss = total_lb_loss / hp.LOG_INTERVAL
            ppl = math.exp(cur_main_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_data) // hp.BPTT:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | main_loss {cur_main_loss:5.2f} | lb_loss {cur_lb_loss:5.2f} | '
                  f'ppl {ppl:8.2f}')
            total_loss = 0
            total_main_loss = 0
            total_lb_loss = 0
            start_time = time.time()

def evaluate(model, data_source, criterion, ntokens):
    """Evaluates the model on the given data source."""
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, hp.BPTT):
            data, targets = get_batch(data_source, i, hp.BPTT)
            with torch.amp.autocast('cuda'):
                output, _ = model(data)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

def main():
    """Main function to run the training and evaluation."""
    # Load data
    with open(hp.VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    # These tensors are already on the correct device from prepare_data.py
    train_data = torch.load(hp.TRAIN_DATA_PATH, weights_only=True)
    val_data = torch.load(hp.VALID_DATA_PATH, weights_only=True)
    test_data = torch.load(hp.TEST_DATA_PATH, weights_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ntokens = len(vocab)

    model = SwitchTransformerLM(ntokens, hp.D_MODEL, hp.NHEAD, hp.D_FF, hp.NUM_EXPERTS, hp.NUM_LAYERS, hp.DROPOUT).to(device)
    model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(1, hp.EPOCHS + 1):
        epoch_start_time = time.time()
        train_one_epoch(model, train_data, criterion, optimizer, scheduler, scaler, ntokens, epoch)
        val_loss = evaluate(model, val_data, criterion, ntokens)
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {math.exp(val_loss):8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, hp.MODEL_PATH)

        scheduler.step()

    # Test the best model
    if best_model_state:
        model.load_state_dict(best_model_state) # Load best model weights
        test_loss = evaluate(model, test_data, criterion, ntokens)
        print('=' * 89)
        print(f'| End of training | test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):8.2f}')
        print('=' * 89)

if __name__ == '__main__':
    main()

