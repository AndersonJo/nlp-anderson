# CLIP from Scratch - PyTorch Implementation

A minimal, faithful PyTorch implementation of [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., 2021).

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │              CLIP Model (151.3M)             │
                    │                                              │
  Image ──────►     │  ┌──────────────────┐  ┌──────────────────┐  │
  (224×224)         │  │Vision Transformer│  │ Text Transformer │  │    ◄──── Text
                    │  │   (ViT-B/32)     │  │  (Masked Attn)   │  │         (77 tokens)
                    │  │                  │  │                  │  │
                    │  │ Patch 32×32      │  │ BPE vocab 49408  │  │
                    │  │ 12 layers        │  │ 12 layers        │  │
                    │  │ 768 width        │  │ 512 width        │  │
                    │  │ 12 heads         │  │ 8 heads          │  │
                    │  │                  │  │                  │  │
                    │  │ [CLS] → proj     │  │ [EOS] → proj     │  │
                    │  └───────┬──────────┘  └────────┬─────────┘  │
                    │          │                      │            │
                    │          ▼                      ▼            │
                    │     image_embed (512)      text_embed (512)  │
                    │          │                      │            │
                    │          └──────┐    ┌──────────┘            │
                    │                 ▼    ▼                       │
                    │          cosine similarity × τ               │
                    │          (learnable temperature)             │
                    │                   │                          │
                    │                   ▼                          │
                    │        symmetric InfoNCE loss                │
                    └──────────────────────────────────────────────┘
```

## Paper Details Reflected in Code

| Paper Section | Detail | Implementation |
|---|---|---|
| Table 1 | ViT-B/32 config | `VisionTransformer`: 12L, 768W, 12H, patch 32 |
| Table 1 | Text encoder config | `TextTransformer`: 12L, 512W, 8H, ctx 77 |
| Sec 2.3 | Pre-LayerNorm | `TransformerBlock` applies LN before attn/FFN |
| Sec 2.3 | QuickGELU | `x * sigmoid(1.702 * x)` activation |
| Sec 2.4 | Causal text mask | Lower-triangular attention mask in `TextTransformer` |
| Sec 2.4 | [EOS] as text feature | `text.argmax(dim=-1)` to find [EOT] position |
| Sec 2.4 | [CLS] as image feature | `x[:, 0, :]` after transformer |
| Sec 2.5 | Learnable temperature | `logit_scale = log(1/0.07)`, clamped to max 100 |
| Sec 2.5 | Symmetric loss | Average of image→text and text→image cross-entropy |
| Sec 2.5 | AdamW | β₁=0.9, β₂=0.98, ε=1e-6, weight decay=0.2 |
| Sec 2.5 | Cosine schedule | Linear warmup 2000 steps, then cosine decay |
| Sec 3.1.4 | Zero-shot eval | Template ensemble with prompt engineering |

## Project Structure

```
.
├── clip_model.py      # CLIP model (ViT + Text Transformer + contrastive loss)
├── train.py           # Training pipeline (data, optimization, evaluation)
├── requirements.txt   # Dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

### Dependencies

- `torch >= 2.0` - Core framework
- `torchvision >= 0.15` - Image transforms
- `transformers >= 4.30` - CLIP BPE tokenizer
- `datasets >= 2.14` - HuggingFace dataset loading
- `Pillow >= 9.0` - Image I/O
- `requests >= 2.28` - Image URL downloading (for CC3M)

## Quick Start

### Test the pipeline (no data needed)

```bash
# Verify model builds correctly
python clip_model.py

# Train on synthetic data to verify pipeline
python train.py --dataset dummy --model small --epochs 3 --batch-size 64
```

### Train on real data

```bash
# Conceptual Captions 3M (recommended)
python train.py --dataset cc3m --epochs 32 --batch-size 256

# COCO Captions
python train.py --dataset coco --epochs 32 --batch-size 128

# Custom dataset
python train.py --dataset folder --data-dir /path/to/data --epochs 32
```

### Custom dataset format

```
data_dir/
  images/
    img_001.jpg
    img_002.jpg
    ...
  captions.txt        # tab-separated: filename<TAB>caption
```

## Recommended Datasets

| Dataset | Size | Pros | Cons |
|---|---|---|---|
| **CC3M** | 3.3M pairs | Large, diverse web images | URLs may expire over time |
| **COCO Captions** | 118K × 5 captions | High quality, easy download | Small scale |
| **Flickr30k** | 30K × 5 captions | Clean, good for prototyping | Very small |
| **LAION-400M** | 400M pairs | Closest to original WIT | Requires significant compute |

For meaningful results, CC3M is the best balance of scale and accessibility. COCO Captions works well for validating the implementation with limited resources.

## Training Hyperparameters

All defaults match the original paper (Section 2.5):

```
--lr 5e-4              # Peak learning rate
--weight-decay 0.2     # AdamW decoupled weight decay
--beta1 0.9            # Adam β₁
--beta2 0.98           # Adam β₂ (higher for training stability)
--eps 1e-6             # Adam ε
--warmup-steps 2000    # Linear warmup steps
--epochs 32            # Total training epochs
--batch-size 256       # Per-GPU (paper uses 32768 across GPUs)
```

## Model Variants

| Model | Params | VRAM (bs=256) | Use case |
|---|---|---|---|
| `vit-b/32` | 151.3M | ~12 GB | Full paper reproduction |
| `small` | 29.4M | ~4 GB | Quick experiments, debugging |

```bash
python train.py --model vit-b/32    # Default, paper config
python train.py --model small       # Lightweight
```

## Note on Unsloth

[Unsloth](https://github.com/unslothai/unsloth) is designed for fine-tuning large language models (LLaMA, Mistral, etc.) with LoRA/QLoRA. It does not support custom vision-language architectures like CLIP.

This implementation uses equivalent PyTorch-native optimizations:

| Unsloth Feature | PyTorch Equivalent Used |
|---|---|
| FP16/BF16 training | `torch.amp.autocast` + `GradScaler` |
| Memory optimization | Gradient checkpointing (`--grad-checkpoint`) |
| Efficient data loading | `pin_memory`, `num_workers`, `prefetch_factor` |

## Usage Examples

### Encode images and text

```python
from clip_from_the_scratch.clip_model import clip_vit_b32
from clip_from_the_scratch.train import SimpleTokenizer, get_image_transform
from PIL import Image
import torch

device = "cuda"
model = clip_vit_b32().to(device)
model.load_state_dict(torch.load("clip_from_the_scratch/checkpoints/clip_best.pt")["model_state_dict"])
model.eval()

tokenizer = SimpleTokenizer()
transform = get_image_transform(224, is_train=False)

# Encode
image = transform(Image.open("cat.jpg")).unsqueeze(0).to(device)
text = tokenizer(["a photo of a cat", "a photo of a dog"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    similarity = (image_features @ text_features.T).softmax(dim=-1)

print(similarity)  # [[0.95, 0.05]] → "cat" matches
```

### Zero-shot classification

```python
from clip_from_the_scratch.train import zero_shot_eval, SimpleTokenizer
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

tokenizer = SimpleTokenizer()
dataset = CIFAR100(root="./data", train=False, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=256)

accuracy = zero_shot_eval(
    model, tokenizer, loader,
    class_names=dataset.classes,
    device=device,
)
```

## References

- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) - Original CLIP paper
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Vision Transformer (ViT)
- [OpenAI CLIP](https://github.com/openai/CLIP) - Official implementation
