# SimCSE: Simple Contrastive Learning of Sentence Embeddings

PyTorch implementation of SimCSE with optimized preprocessing for fast training.

## Features

- ✅ **Optimized Preprocessing**: Pre-tokenize data for 2.79x faster training
- ✅ **SNLI Dataset Support**: Automatic filtering for entailment pairs  
- ✅ **Mixed Precision Training**: FP16 for memory efficiency
- ✅ **PyTorch 2.0 Compilation**: Automatic model optimization
- ✅ **Multi-GPU Support**: DataParallel for faster training
- ✅ **Interactive Demo**: Test trained models easily

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data

```bash
# Preprocess SNLI dataset (automatic download)
python preprocess.py --max_samples 50000 --max_length 64

# Or use all available data
python preprocess.py
```

### 3. Train Model

```bash
# Train with preprocessed data
python train.py

# Custom settings
python train.py --batch_size 128 --epochs 3
```

### 4. Test Model

```bash
# Interactive demo
python demo.py

# Evaluation
python evaluate.py
```

## Project Structure

```
📁 SimCSE/
├── 📄 config.py           # Training configuration
├── 📄 model.py            # SimCSE model implementation
├── 📄 preprocess.py       # Data preprocessing script
├── 📄 data_loader.py      # Optimized data loading
├── 📄 train.py            # Training script
├── 📄 evaluate.py         # Evaluation script
├── 📄 demo.py             # Interactive demo
├── 📄 utils.py            # Utility functions
└── 📁 preprocessed_data/  # Preprocessed data storage
```

## Configuration

Key settings in `config.py`:

```python
class Config:
    # Model
    model_name = "bert-base-uncased"
    pooler_type = "cls"
    temp = 0.05
    
    # Training (optimized)
    batch_size = 256
    max_seq_length = 64
    learning_rate = 5e-5
    fp16 = True
    compile_model = True
    
    # Data
    preprocessed_data_path = "./preprocessed_data/train_preprocessed.pkl"
```

## Performance Optimization

### Preprocessing Benefits

- **DataLoader Speed**: 2.79x faster than on-demand tokenization
- **CPU Usage**: Significantly reduced during training  
- **Memory Efficiency**: Optimized tensor storage and access
- **Scalability**: Better performance with larger datasets

### When to Use Preprocessing

✅ **Recommended for**:
- Large datasets (50K+ samples)
- Multiple epochs (3+)
- Repeated experiments
- CPU-limited environments

⚠️ **Standard approach for**:
- Small datasets (< 20K samples)
- Single epoch training
- One-time experiments

## Advanced Usage

### Custom Data Preprocessing

```bash
# Different sequence length
python preprocess.py --max_length 128

# Limit samples
python preprocess.py --max_samples 100000

# Different model
python preprocess.py --model_name roberta-base
```

### Training Options

```bash
# Custom preprocessed data
python train.py --preprocessed_data ./custom_data.pkl

# Different output directory
python train.py --output_dir ./my_model

# Override batch size
python train.py --batch_size 512
```

## Model Performance

The trained model achieves competitive performance on sentence similarity tasks:

- **Spearman Correlation**: ~0.60-0.80 on test sets
- **Training Speed**: Optimized with preprocessing
- **Memory Usage**: Efficient with mixed precision

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.20+
- Datasets
- NumPy
- scikit-learn
- tqdm

## Citation

This implementation is based on the SimCSE paper:

```bibtex
@inproceedings{gao2021simcse,
   title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
   author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2021}
}
``` 