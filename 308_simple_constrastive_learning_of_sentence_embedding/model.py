import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional

class SimCSE(nn.Module):
    def __init__(self, model_name: str, pooler_type: str = "cls", temp: float = 0.05):
        super(SimCSE, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.pooler_type = pooler_type
        self.temp = temp
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass for SimCSE
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] (optional)
        Returns:
            sentence_embeddings: [batch_size, hidden_size]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Pooling
        sentence_embeddings = self.pooling(outputs, attention_mask)
        
        return sentence_embeddings
    
    def pooling(self, outputs, attention_mask):
        """
        Pooling operation to get sentence embeddings
        """
        if self.pooler_type == "cls":
            return outputs.last_hidden_state[:, 0]  # [CLS] token
        elif self.pooler_type == "cls_before_pooler":
            return outputs.last_hidden_state[:, 0]
        elif self.pooler_type == "avg":
            return self.avg_pooling(outputs.last_hidden_state, attention_mask)
        elif self.pooler_type == "avg_top2":
            second_last_hidden = outputs.hidden_states[-2]
            last_hidden = outputs.hidden_states[-1]
            pooled_result = (self.avg_pooling(last_hidden, attention_mask) + 
                           self.avg_pooling(second_last_hidden, attention_mask)) / 2
            return pooled_result
        elif self.pooler_type == "avg_first_last":
            first_hidden = outputs.hidden_states[1]  # Skip embedding layer
            last_hidden = outputs.hidden_states[-1]
            pooled_result = (self.avg_pooling(last_hidden, attention_mask) + 
                           self.avg_pooling(first_hidden, attention_mask)) / 2
            return pooled_result
        else:
            raise NotImplementedError(f"Pooler type {self.pooler_type} not implemented")
    
    def avg_pooling(self, hidden_states, attention_mask):
        """
        Average pooling with attention mask
        """
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]
        
        # Expand attention mask to match hidden states dimensions
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Apply attention mask and compute mean
        sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def simcse_loss(self, z1, z2):
        """
        Compute SimCSE loss using InfoNCE
        Args:
            z1: [batch_size, hidden_size] - first representations
            z2: [batch_size, hidden_size] - second representations (dropout augmented)
        Returns:
            loss: contrastive loss
        """
        # Normalize embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # Concatenate z1 and z2
        z = torch.cat([z1, z2], dim=0)  # [2*batch_size, hidden_size]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.t()) / self.temp  # [2*batch_size, 2*batch_size]
        
        batch_size = z1.size(0)
        
        # Create labels for positive pairs
        # For i-th sample in z1, its positive is (i+batch_size)-th sample in z2
        # For i-th sample in z2, its positive is (i-batch_size)-th sample in z1
        labels = torch.arange(2 * batch_size, device=z.device)
        labels[:batch_size] += batch_size  # z1's positives are in z2
        labels[batch_size:] -= batch_size  # z2's positives are in z1
        
        # Mask out diagonal (self-similarity)
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class SimCSEForTraining(nn.Module):
    """
    Wrapper for training SimCSE with contrastive learning
    """
    def __init__(self, model_name: str, pooler_type: str = "cls", temp: float = 0.05):
        super(SimCSEForTraining, self).__init__()
        self.simcse = SimCSE(model_name, pooler_type, temp)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass with dropout-based augmentation
        """
        # First forward pass
        z1 = self.simcse(input_ids, attention_mask, token_type_ids)
        
        # Second forward pass (dropout will create different representations)
        z2 = self.simcse(input_ids, attention_mask, token_type_ids)
        
        # Compute contrastive loss
        loss = self.simcse.simcse_loss(z1, z2)
        
        return {
            'loss': loss,
            'embeddings_1': z1,
            'embeddings_2': z2
        }
    
    def encode(self, input_ids, attention_mask, token_type_ids=None):
        """
        Encode sentences for evaluation (no dropout)
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.simcse(input_ids, attention_mask, token_type_ids)
        return embeddings 