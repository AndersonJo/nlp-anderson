"""
CLIP (Contrastive Language-Image Pre-training) - PyTorch Implementation
========================================================================
Faithful implementation of: "Learning Transferable Visual Models From Natural
Language Supervision" (Radford et al., 2021)

Architecture: ViT-B/32 image encoder + Masked Transformer text encoder
Loss: Symmetric InfoNCE (contrastive) loss with learnable temperature
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class QuickGELU(nn.Module):
    """Approximation of GELU used in the original CLIP implementation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional causal mask."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.in_proj = nn.Linear(d_model, 3 * d_model)  # QKV projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, is_causal: bool = False):
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_mask: (seq_len, seq_len) additive mask (0 or -inf)
            is_causal: use causal mask (faster than explicit attn_mask)
        """
        B, L, D = x.shape
        qkv = self.in_proj(x)  # (B, L, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_heads, L, head_dim)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's optimized SDPA (FlashAttention / memory-efficient)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
        )

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block (as used in CLIP paper)."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(mlp_width, d_model)),
        ]))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, is_causal: bool = False):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask, is_causal=is_causal)
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Vision Transformer (ViT-B/32) - Image Encoder
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """
    ViT-B/32 as described in the CLIP paper (Table 1).

    - Patch embedding via Conv2d (kernel=stride=patch_size)
    - Prepend learnable [CLS] token
    - Learned positional embeddings
    - Pre-LayerNorm Transformer
    - Final LayerNorm → linear projection
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 32,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 7*7 = 49

        # Patch embedding: Conv2d acts as linear projection of flattened patches
        self.patch_embed = nn.Conv2d(
            3, width, kernel_size=patch_size, stride=patch_size, bias=False
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_patches + 1, width)
        )

        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.Sequential(
            *[TransformerBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)

        # Projection to shared embedding space
        self.proj = nn.Parameter(scale * torch.randn(width, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, image_size, image_size)
        Returns:
            (batch, embed_dim) - L2-normalized image embeddings
        """
        B = x.shape[0]

        # Patch embedding: (B, 3, H, W) → (B, width, grid, grid) → (B, num_patches, width)
        x = self.patch_embed(x)  # (B, width, 7, 7)
        x = x.flatten(2).transpose(1, 2)  # (B, 49, width)

        # Prepend [CLS] token
        cls = self.class_embedding.unsqueeze(0).expand(B, -1).unsqueeze(1)  # (B, 1, width)
        x = torch.cat([cls, x], dim=1)  # (B, 50, width)

        # Add positional embeddings
        x = x + self.positional_embedding

        x = self.ln_pre(x)

        # Transformer blocks
        for block in self.transformer:
            x = block(x)

        # Take [CLS] token output
        x = self.ln_post(x[:, 0, :])  # (B, width)

        # Project to shared embedding space
        x = x @ self.proj  # (B, embed_dim)

        return x


# ---------------------------------------------------------------------------
# Text Transformer - Text Encoder
# ---------------------------------------------------------------------------

class TextTransformer(nn.Module):
    """
    Masked (causal) Transformer text encoder as described in the CLIP paper.

    - Byte-pair encoding tokenizer (external, vocab_size=49408)
    - Token + positional embeddings
    - Causal self-attention mask (autoregressive)
    - [EOS] token activation as text representation
    - Final LayerNorm → linear projection
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        context_length: int = 77,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(
            torch.empty(context_length, width)
        )

        self.transformer = nn.ModuleList(
            [TransformerBlock(width, heads) for _ in range(layers)]
        )
        self.ln_final = nn.LayerNorm(width)

        # Projection to shared embedding space
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.ln_final.normalized_shape[0] ** -0.5)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text: (batch, context_length) - tokenized text with padding
        Returns:
            (batch, embed_dim) - L2-normalized text embeddings
        """
        x = self.token_embedding(text)  # (B, L, width)
        x = x + self.positional_embedding

        for block in self.transformer:
            x = block(x, is_causal=True)

        x = self.ln_final(x)

        # Take features from the [EOS] token position
        # [EOS] is the highest-numbered token in each sequence
        eos_indices = text.argmax(dim=-1)  # (B,)
        x = x[torch.arange(x.shape[0], device=x.device), eos_indices]  # (B, width)

        # Project to shared embedding space
        x = x @ self.text_projection  # (B, embed_dim)

        return x


# ---------------------------------------------------------------------------
# CLIP Model
# ---------------------------------------------------------------------------

class CLIP(nn.Module):
    """
    CLIP: Contrastive Language-Image Pre-training

    Combines a Vision Transformer (image encoder) and a masked Transformer
    (text encoder) trained with a symmetric contrastive loss (InfoNCE).

    Key paper details reflected:
    - Learnable temperature parameter (logit_scale), initialized to ln(1/0.07)
    - Symmetric cross-entropy loss over cosine similarity matrix
    - L2 normalization of both image and text embeddings before computing similarity
    - Pre-LayerNorm Transformer architecture

    Config sizes from paper (Table 1):
    ┌──────────┬────────┬───────┬───────┬──────┬────────────┐
    │ Model    │ Layers │ Width │ Heads │ Patch│ Embed dim  │
    ├──────────┼────────┼───────┼───────┼──────┼────────────┤
    │ ViT-B/32 │   12   │  768  │  12   │  32  │    512     │
    │ Text     │   12   │  512  │   8   │  -   │    512     │
    └──────────┴────────┴───────┴───────┴──────┴────────────┘
    """

    def __init__(
        self,
        # Vision
        image_size: int = 224,
        patch_size: int = 32,
        vision_width: int = 768,
        vision_layers: int = 12,
        vision_heads: int = 12,
        # Text
        vocab_size: int = 49408,
        context_length: int = 77,
        text_width: int = 512,
        text_layers: int = 12,
        text_heads: int = 8,
        # Shared
        embed_dim: int = 512,
    ):
        super().__init__()

        self.context_length = context_length

        self.visual = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            embed_dim=embed_dim,
        )

        self.text = TextTransformer(
            vocab_size=vocab_size,
            context_length=context_length,
            width=text_width,
            layers=text_layers,
            heads=text_heads,
            embed_dim=embed_dim,
        )

        # Learnable temperature parameter
        # Paper: "initialized to the equivalent of np.log(1/0.07) ≈ 2.6593"
        # Clipped to prevent scaling the logits by more than 100 (paper Section 2.5)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode images to L2-normalized embeddings."""
        x = self.visual(image)
        return F.normalize(x, dim=-1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to L2-normalized embeddings."""
        x = self.text(text)
        return F.normalize(x, dim=-1)

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        """
        Compute CLIP contrastive loss.

        Args:
            image: (batch, 3, H, W) - preprocessed images
            text:  (batch, context_length) - tokenized text

        Returns:
            loss: scalar - symmetric InfoNCE loss
            logits_per_image: (batch, batch) - image-to-text similarity
            logits_per_text: (batch, batch) - text-to-image similarity
        """
        image_features = self.encode_image(image)  # (B, embed_dim)
        text_features = self.encode_text(text)      # (B, embed_dim)

        # Clamp logit_scale as described in the paper (max 100)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # Cosine similarity as logits (scaled by temperature)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # Symmetric cross-entropy loss (InfoNCE)
        # Ground truth: each image matches its corresponding text (diagonal)
        labels = torch.arange(len(image), device=image.device)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2.0

        return loss, logits_per_image, logits_per_text


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def clip_vit_b32(**kwargs) -> CLIP:
    """Create CLIP ViT-B/32 (default config from the paper)."""
    return CLIP(
        image_size=224, patch_size=32,
        vision_width=768, vision_layers=12, vision_heads=12,
        vocab_size=49408, context_length=77,
        text_width=512, text_layers=12, text_heads=8,
        embed_dim=512,
        **kwargs,
    )


def clip_vit_b16(**kwargs) -> CLIP:
    """Create CLIP ViT-B/16 (higher resolution patches)."""
    return CLIP(
        image_size=224, patch_size=16,
        vision_width=768, vision_layers=12, vision_heads=12,
        vocab_size=49408, context_length=77,
        text_width=512, text_layers=12, text_heads=8,
        embed_dim=512,
        **kwargs,
    )


def clip_small(**kwargs) -> CLIP:
    """Smaller CLIP for debugging / resource-constrained training."""
    return CLIP(
        image_size=224, patch_size=32,
        vision_width=384, vision_layers=6, vision_heads=6,
        vocab_size=49408, context_length=77,
        text_width=256, text_layers=6, text_heads=4,
        embed_dim=256,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = clip_vit_b32().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"CLIP ViT-B/32 total parameters: {total_params / 1e6:.1f}M")

    # Dummy forward pass
    images = torch.randn(4, 3, 224, 224, device=device)
    texts = torch.randint(0, 49408, (4, 77), device=device)

    loss, logits_i, logits_t = model(images, texts)
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits_i.shape}")
    print(f"Logit scale: {model.logit_scale.exp().item():.2f}")
