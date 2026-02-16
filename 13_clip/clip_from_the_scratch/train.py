"""
CLIP Training Pipeline
======================
Train CLIP from scratch using PyTorch with native optimizations.

Recommended datasets (in order of preference):
1. Conceptual Captions 3M (CC3M) - 3M image-text pairs, academic use
2. COCO Captions 2017 - 118K images × 5 captions, easy to obtain
3. Flickr30k - 30K images × 5 captions, good for quick experiments
4. LAION-400M - 400M pairs, closest to original CLIP's WIT dataset

This script uses HuggingFace `datasets` for CC3M / COCO, with automatic
image downloading and caching.

Optimizations (replacing Unsloth, which is LLM-specific):
- torch.cuda.amp (FP16 mixed precision)
- Gradient checkpointing option
- Efficient data loading with prefetching
- Cosine LR schedule with linear warmup (as in paper)
- AdamW with decoupled weight decay (as in paper)

Usage:
    python train.py                          # Train on COCO Captions (default)
    python train.py --dataset cc3m           # Train on CC3M
    python train.py --model small --epochs 5 # Quick experiment
"""

import os
import io
import math
import argparse
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from clip_model import CLIP, clip_vit_b32, clip_small

# Optional imports with graceful fallback
try:
    from datasets import load_dataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False
    print("Warning: `datasets` not installed. Install with: pip install datasets")

try:
    from transformers import CLIPTokenizerFast
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Simple BPE Tokenizer (fallback when transformers is not installed)
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """
    Minimal tokenizer for CLIP.
    Uses HuggingFace CLIPTokenizerFast when available, otherwise falls back
    to a basic character-level tokenizer (for testing only).

    The original CLIP uses byte-pair encoding with 49,408 vocab size,
    operating on lower-cased byte-pair encoded text, with [SOT] and [EOT]
    special tokens. Context length is 77 tokens.
    """

    def __init__(self, context_length: int = 77):
        self.context_length = context_length

        if HAS_TOKENIZER:
            self._tokenizer = CLIPTokenizerFast.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.vocab_size = self._tokenizer.vocab_size
            self.use_hf = True
        else:
            print("Warning: Using basic fallback tokenizer. "
                  "Install transformers for proper BPE tokenization: "
                  "pip install transformers")
            # Basic char-level tokenizer for testing
            self.vocab_size = 49408
            self.use_hf = False

    def __call__(self, texts, return_tensors="pt"):
        """Tokenize a batch of strings."""
        if isinstance(texts, str):
            texts = [texts]

        if self.use_hf:
            encoded = self._tokenizer(
                texts,
                max_length=self.context_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return encoded["input_ids"]
        else:
            return self._basic_tokenize(texts)

    def _basic_tokenize(self, texts):
        """Fallback character-level tokenizer."""
        SOT_TOKEN = 49406
        EOT_TOKEN = 49407

        batch = []
        for text in texts:
            text = text.lower().strip()
            tokens = [SOT_TOKEN]
            for ch in text[: self.context_length - 2]:
                tokens.append(min(ord(ch), 49405))  # Clamp to vocab
            tokens.append(EOT_TOKEN)
            # Pad to context_length
            tokens += [0] * (self.context_length - len(tokens))
            batch.append(tokens)

        return torch.tensor(batch, dtype=torch.long)


# ---------------------------------------------------------------------------
# Image Preprocessing (following the paper)
# ---------------------------------------------------------------------------

def get_image_transform(image_size: int = 224, is_train: bool = True):
    """
    Image preprocessing as described in the CLIP paper:
    - Resize to image_size (shorter side)
    - Center crop to image_size × image_size
    - Normalize with ImageNet mean/std (as used by CLIP)

    Training adds random resized crop and horizontal flip.
    """
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],   # CLIP's normalization
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class CC3MDataset(Dataset):
    """
    Conceptual Captions 3M dataset via HuggingFace.

    CC3M contains ~3.3M image-text pairs with diverse web images.
    Images are provided as URLs and downloaded on-the-fly.
    """

    def __init__(self, split="train", image_size=224, max_samples=None):
        assert HAS_HF_DATASETS, "Install datasets: pip install datasets"
        assert HAS_REQUESTS, "Install requests: pip install requests"

        print(f"Loading CC3M {split} split...")
        self.dataset = load_dataset(
            "conceptual_captions", split=split,
        )
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        self.transform = get_image_transform(image_size, is_train=(split == "train"))
        print(f"CC3M {split}: {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        caption = item["caption"]
        image_url = item["image_url"]

        try:
            response = requests.get(image_url, timeout=5)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image = self.transform(image)
        except Exception:
            # Return a black image on failure (will be filtered in collate)
            image = torch.zeros(3, 224, 224)

        return image, caption


class COCOCaptionsDataset(Dataset):
    """
    COCO Captions via HuggingFace datasets (jxie/coco_captions).

    567K training samples (118K images × ~5 captions each, pre-flattened).
    Each row has one image and one caption.
    """

    def __init__(self, split="train", image_size=224, max_samples=None):
        assert HAS_HF_DATASETS, "Install datasets: pip install datasets"

        print(f"Loading COCO Captions {split} split...")
        hf_split = "train" if split == "train" else "validation"
        self.dataset = load_dataset(
            "jxie/coco_captions",
            split=hf_split,
        )
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        self.transform = get_image_transform(image_size, is_train=(split == "train"))
        print(f"COCO {split}: {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        caption = item["caption"]
        image = self.transform(image)
        return image, caption


class ImageTextFolderDataset(Dataset):
    """
    Generic dataset: reads images from a folder with a captions file.

    Expected structure:
        data_dir/
            images/
                img_001.jpg
                img_002.jpg
                ...
            captions.txt   (format: "filename\tcaption" per line)

    This allows training on any custom dataset.
    """

    def __init__(self, data_dir: str, image_size=224, is_train=True, max_samples=None):
        self.data_dir = data_dir
        self.transform = get_image_transform(image_size, is_train=is_train)

        captions_file = os.path.join(data_dir, "captions.txt")
        self.samples = []

        with open(captions_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    self.samples.append((parts[0], parts[1]))

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} image-text pairs from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, caption = self.samples[idx]
        image_path = os.path.join(self.data_dir, "images", filename)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, caption




# ---------------------------------------------------------------------------
# Collate function (handles failed image downloads)
# ---------------------------------------------------------------------------

def clip_collate_fn(batch):
    """Filter out samples where image download failed (all-zero images)."""
    valid = [(img, cap) for img, cap in batch if img.sum() != 0]
    if len(valid) == 0:
        return None
    images = torch.stack([img for img, _ in valid])
    captions = [cap for _, cap in valid]
    return images, captions


# ---------------------------------------------------------------------------
# Learning rate schedule (from paper)
# ---------------------------------------------------------------------------

class CosineWarmupScheduler:
    """
    Cosine annealing with linear warmup, as described in the CLIP paper.

    The paper uses:
    - Linear warmup for the first 2000 steps
    - Cosine decay to 0 for the remaining steps
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr_scale = self._get_lr_scale()
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * lr_scale

    def _get_lr_scale(self) -> float:
        if self.step_count <= self.warmup_steps:
            return self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: CLIP,
    dataloader: DataLoader,
    tokenizer: SimpleTokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    device: torch.device,
    epoch: int,
    amp_dtype: torch.dtype = torch.bfloat16,
    log_interval: int = 50,
):
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        if batch is None:  # All images in batch failed
            continue

        images, captions = batch
        images = images.to(device, non_blocking=True)
        text_tokens = tokenizer(captions).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass (BF16)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            loss, logits_i, logits_t = model(images, text_tokens)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]
            logit_scale = model.logit_scale.exp().item()
            samples_per_sec = (num_batches * dataloader.batch_size) / elapsed
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                f"Logit scale: {logit_scale:.2f} | "
                f"{samples_per_sec:.0f} samples/s | "
                f"Time: {elapsed:.1f}s"
            )

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


@torch.no_grad()
def validate(
    model: CLIP,
    dataloader: DataLoader,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        if batch is None:
            continue

        images, captions = batch
        images = images.to(device, non_blocking=True)
        text_tokens = tokenizer(captions).to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            loss, _, _ = model(images, text_tokens)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


# ---------------------------------------------------------------------------
# Zero-shot classification evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def zero_shot_eval(
    model: CLIP,
    tokenizer: SimpleTokenizer,
    dataloader: DataLoader,
    class_names: list,
    templates: list = None,
    device: torch.device = torch.device("cpu"),
):
    """
    Zero-shot classification as described in Section 3.1.4 of the paper.

    For each class, create text prompts using templates like
    "a photo of a {class}" and average the text embeddings.
    Then classify images by finding the nearest text embedding.
    """
    if templates is None:
        # Paper uses 80 templates; here we use a representative subset
        templates = [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a photo of the large {}.",
            "a photo of the small {}.",
            "a photo of a {} in the wild.",
            "a bright photo of a {}.",
            "a dark photo of a {}.",
            "a close-up photo of a {}.",
        ]

    model.eval()

    # Build text classifier weights (ensemble of templates)
    print("Building zero-shot classifier...")
    text_features_list = []
    for class_name in class_names:
        texts = [template.format(class_name) for template in templates]
        text_tokens = tokenizer(texts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features.mean(dim=0)  # Average over templates
        text_features = F.normalize(text_features, dim=-1)
        text_features_list.append(text_features)

    text_classifier = torch.stack(text_features_list, dim=0)  # (num_classes, embed_dim)

    # Classify images
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images)  # (B, embed_dim)

        # Cosine similarity
        similarity = image_features @ text_classifier.t()  # (B, num_classes)
        predictions = similarity.argmax(dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.shape[0]

    accuracy = correct / max(1, total)
    print(f"Zero-shot accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import glob
import re


def find_best_checkpoint(save_dir):
    """Find checkpoint with minimum loss by parsing filenames like clip_ep05_3.3415.pt."""
    pattern = os.path.join(save_dir, "clip_ep*_*.pt")
    best_path, best_loss = None, float("inf")
    for path in glob.glob(pattern):
        m = re.search(r"clip_ep(\d+)_(\d+\.\d+)\.pt$", path)
        if m:
            loss = float(m.group(2))
            if loss < best_loss:
                best_loss = loss
                best_path = path
    return best_path, best_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP from scratch")

    # Model
    parser.add_argument("--model", type=str, default="vit-b/32",
                        choices=["vit-b/32", "small"],
                        help="Model size (default: vit-b/32)")

    # Data
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=["cc3m", "coco", "folder"],
                        help="Dataset to use (default: coco)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (for 'folder' dataset)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to use (for quick experiments)")
    parser.add_argument("--image-size", type=int, default=224)

    # Training hyperparameters (from paper)
    parser.add_argument("--epochs", type=int, default=32,
                        help="Number of training epochs (paper uses 32)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size (paper uses 32768, we use 1024 for single GPU)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Peak learning rate (paper: 5e-4 for ViT-B/32)")
    parser.add_argument("--weight-decay", type=float, default=0.2,
                        help="Weight decay (paper: 0.2)")
    parser.add_argument("--warmup-steps", type=int, default=2000,
                        help="Linear warmup steps (paper: 2000)")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam beta1 (paper: 0.9)")
    parser.add_argument("--beta2", type=float, default=0.98,
                        help="Adam beta2 (paper: 0.98, for stability)")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="Adam epsilon (paper: 1e-6)")

    # Optimization
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use mixed precision training (BF16)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--compile", action="store_true", default=True,
                        help="Use torch.compile for kernel fusion (default: on)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing to save memory")

    # Logging & saving
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, nargs="?", const="auto",
                        help="Resume from checkpoint. 'auto' or omit value to load best, or pass a path")

    return parser.parse_args()


def main():
    args = parse_args()
    use_amp = args.amp and not args.no_amp
    use_compile = args.compile and not args.no_compile

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable TF32 for matmuls and cuDNN (large speedup, negligible precision loss)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    print(f"\nCreating CLIP model: {args.model}")
    if args.model == "vit-b/32":
        model = clip_vit_b32()
    elif args.model == "small":
        model = clip_small()

    model = model.to(device)

    if use_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")

    # -----------------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------------
    tokenizer = SimpleTokenizer(context_length=model.context_length)

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == "cc3m":
        train_dataset = CC3MDataset(
            split="train", image_size=args.image_size,
            max_samples=args.max_samples,
        )
    elif args.dataset == "coco":
        train_dataset = COCOCaptionsDataset(
            split="train", image_size=args.image_size,
            max_samples=args.max_samples,
        )
    elif args.dataset == "folder":
        assert args.data_dir, "--data-dir required for 'folder' dataset"
        train_dataset = ImageTextFolderDataset(
            data_dir=args.data_dir, image_size=args.image_size,
            max_samples=args.max_samples,
        )

    # Validation dataset
    val_dataset = None
    if args.dataset == "coco":
        val_dataset = COCOCaptionsDataset(
            split="val", image_size=args.image_size,
        )
    elif args.dataset == "cc3m":
        val_dataset = CC3MDataset(
            split="validation", image_size=args.image_size,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=clip_collate_fn,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=clip_collate_fn,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )

    # -----------------------------------------------------------------------
    # Optimizer & scheduler (paper Section 2.5)
    # -----------------------------------------------------------------------
    # The paper uses AdamW with decoupled weight decay
    # "We use a very large minibatch size of 32,768"
    # "trained for 32 epochs"
    # "cosine schedule ... linear warmup over the first 2000 updates"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )

    # BF16 doesn't need GradScaler (no inf/nan from reduced exponent range)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # -----------------------------------------------------------------------
    # Resume from checkpoint
    # -----------------------------------------------------------------------
    start_epoch = 1
    best_loss = float("inf")

    if args.resume is not None:
        if args.resume == "auto":
            ckpt_path, _ = find_best_checkpoint(args.save_dir)
        else:
            ckpt_path = args.resume

        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"Resuming from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
            # Strip "_orig_mod." prefix from torch.compile() saved checkpoints
            state_dict = {k.removeprefix("_orig_mod."): v
                          for k, v in checkpoint["model_state_dict"].items()}
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint.get("loss", float("inf"))
            # Restore scheduler state
            scheduler.step_count = checkpoint["epoch"] * len(train_loader)
            print(f"  Resumed from epoch {checkpoint['epoch']}, loss {best_loss:.4f}")
        else:
            print(f"Warning: no checkpoint found, training from scratch")

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Training CLIP")
    print(f"{'=' * 60}")
    print(f"  Model:        {args.model}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Epochs:       {start_epoch}-{args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Learning rate:{args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Total steps:  {total_steps}")
    print(f"  Mixed prec:   {amp_dtype}")
    print(f"  torch.compile:{use_compile}")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            amp_dtype=amp_dtype,
            log_interval=args.log_interval,
        )

        # Validation
        val_loss = None
        if val_loader is not None:
            val_loss = validate(model, val_loader, tokenizer, device, amp_dtype)

        epoch_time = time.time() - epoch_start
        val_str = f" | Val Loss: {val_loss:.4f}" if val_loss is not None else ""
        print(
            f"Epoch {epoch}/{args.epochs} completed | "
            f"Train Loss: {avg_loss:.4f}{val_str} | Time: {epoch_time:.1f}s"
        )

        # Save checkpoint (use val loss if available, else train loss)
        check_loss = val_loss if val_loss is not None else avg_loss
        if epoch % args.save_every == 0 or check_loss < best_loss:
            if check_loss < best_loss:
                best_loss = check_loss
            save_path = os.path.join(
                args.save_dir, f"clip_ep{epoch:02d}_{check_loss:.4f}.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": check_loss,
                "args": vars(args),
            }, save_path)
            print(f"  Saved checkpoint: {save_path}")

    best_path, best_loss = find_best_checkpoint(args.save_dir)
    print(f"\nTraining complete! Best: {best_path} (loss: {best_loss:.4f})")


if __name__ == "__main__":
    main()
