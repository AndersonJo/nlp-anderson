import torch
import torch.nn.functional as F
from torch import nn

class Router(nn.Module):
    """
    The router module determines which expert each token is sent to.
    It also computes the load balancing loss.
    """
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        logits = self.gate(x)
        # logits: (batch_size, seq_len, num_experts)
        
        # Get the top-1 expert for each token
        top1_logits, top1_indices = logits.max(dim=-1)
        # top1_indices: (batch_size, seq_len)
        
        # Create a one-hot encoding of the expert indices
        # This will be used to dispatch tokens to the correct expert
        expert_mask = F.one_hot(top1_indices, self.num_experts).float()
        # expert_mask: (batch_size, seq_len, num_experts)
        
        # Calculate the load balancing loss
        # This loss encourages all experts to be used equally.
        
        # Count how many tokens are sent to each expert
        tokens_per_expert = expert_mask.sum(dim=(0, 1)) # (num_experts)
        # Total number of tokens
        total_tokens = x.size(0) * x.size(1)
        
        # Calculate the fraction of tokens sent to each expert
        fraction_tokens_per_expert = tokens_per_expert / total_tokens
        
        # Calculate the expert probabilities from the logits
        expert_probs = F.softmax(logits, dim=-1).mean(dim=(0, 1)) # (num_experts)
        
        # The load balancing loss is the dot product of these two quantities
        load_balancing_loss = self.num_experts * torch.dot(fraction_tokens_per_expert, expert_probs)
        
        return expert_mask, load_balancing_loss
