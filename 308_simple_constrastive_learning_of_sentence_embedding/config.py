import torch

class Config:
    """SimCSE training configuration (optimized for preprocessed data)"""
    
    # Model configuration
    model_name = "bert-base-uncased"
    pooler_type = "cls"  # cls, avg, avg_top2, avg_first_last
    temp = 0.05  # Temperature for contrastive loss
    
    # Training configuration
    batch_size = 256  # Optimized for preprocessed data
    max_seq_length = 64  # Optimized sequence length
    learning_rate = 5e-5
    num_epochs = 1
    warmup_ratio = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0
    gradient_accumulation_steps = 1
    
    # Optimization settings
    fp16 = True  # Mixed precision training
    compile_model = True  # PyTorch 2.0+ compilation
    
    # DataLoader optimization
    dataloader_num_workers = 4
    pin_memory = True
    
    # Data configuration
    use_snli = True
    max_train_samples = None  # Use all available data
    
    # Hardware configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    use_data_parallel = True if num_gpus > 1 else False
    
    # Output configuration
    output_dir = "./output"
    logging_steps = 50
    save_steps = 500
    save_total_limit = 3
    
    # Preprocessing configuration
    preprocessed_data_path = "./preprocessed_data/train_preprocessed.pkl"
    
    # Random seed
    seed = 42
    
    def __post_init__(self):
        """Post initialization checks and adjustments"""
        if self.num_gpus > 1 and self.use_data_parallel:
            print(f"Using DataParallel with {self.num_gpus} GPUs")
            # Adjust batch size for multi-GPU
            self.effective_batch_size = self.batch_size * self.num_gpus
        else:
            self.effective_batch_size = self.batch_size
        
        # Ensure FP16 is disabled on CPU
        if self.device.type == "cpu":
            self.fp16 = False
            self.compile_model = False
        
        print(f"Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Effective batch size: {self.effective_batch_size}")
        print(f"  Mixed precision: {self.fp16}")
        print(f"  Model compilation: {self.compile_model}")
        print(f"  DataLoader workers: {self.dataloader_num_workers}")

# Create default config instance
config = Config()
config.__post_init__() 