import torch
from torch import nn
from .moe_layer import MoELayer

class SwitchTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_experts, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.moe_layer = MoELayer(d_model, d_ff, num_experts)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Multi-head self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # MoE layer
        src2, load_balancing_loss = self.moe_layer(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src, load_balancing_loss

class SwitchTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        total_load_balancing_loss = 0.0
        
        for mod in self.layers:
            output, load_balancing_loss = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            total_load_balancing_loss += load_balancing_loss
            
        return output, total_load_balancing_loss / self.num_layers
