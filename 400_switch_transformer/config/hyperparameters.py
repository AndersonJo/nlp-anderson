# --- Model Hyperparameters ---
D_MODEL = 512
NHEAD = 8
D_FF = 2048
NUM_LAYERS = 6
NUM_EXPERTS = 8
DROPOUT = 0.2

# --- Training Hyperparameters ---
LEARNING_RATE = 0.0003
EPOCHS = 5
BATCH_SIZE = 180
BPTT = 35  # Sequence length
LOG_INTERVAL = 200
LOAD_BALANCING_LOSS_WEIGHT = 0.01

# --- Generation Hyperparameters ---
PROMPT = "The king of France"
MAX_WORDS = 50
TEMPERATURE = 0.8

# --- File Paths ---
MODEL_PATH = 'model.pt'
VOCAB_PATH = 'data/vocab.pkl'
TRAIN_DATA_PATH = 'data/train_data.pt'
VALID_DATA_PATH = 'data/val_data.pt'
TEST_DATA_PATH = 'data/test_data.pt'
