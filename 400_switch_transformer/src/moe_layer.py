import torch
from torch import nn
from .expert import Expert
from .router import Router

class MoELayer(nn.Module):
    """
    The Mixture-of-Experts (MoE) layer, which replaces the FFN layer in a standard Transformer.
    """
    def __init__(self, d_model, d_ff, num_experts, capacity_factor=1.25):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.router = Router(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # Get the expert mask and load balancing loss from the router
        expert_mask, load_balancing_loss = self.router(x)
        # expert_mask: (batch_size, seq_len, num_experts)
        
        # Determine the capacity of each expert
        # Capacity is the max number of tokens an expert can process.
        # (Number of tokens / number of experts) * capacity_factor
        capacity = int((seq_len / self.num_experts) * self.capacity_factor)
        
        # Reshape x for easier processing
        x_flat = x.view(-1, d_model) # (batch_size * seq_len, d_model)
        expert_mask_flat = expert_mask.view(-1, self.num_experts) # (batch_size * seq_len, num_experts)

        # Dispatch tokens to experts
        final_output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            # Get the indices of the tokens that should go to this expert
            expert_indices = torch.where(expert_mask_flat[:, i] > 0)[0]
            
            # If an expert is assigned more tokens than its capacity, drop the excess tokens.
            if expert_indices.shape[0] > capacity:
                expert_indices = expert_indices[:capacity]

            if expert_indices.shape[0] > 0:
                # Select the tokens for the current expert
                expert_input = x_flat[expert_indices]
                
                # Process the tokens with the expert
                expert_output = expert(expert_input)
                
                # Ensure the output is in the same dtype as the final_output before adding
                expert_output = expert_output.to(final_output.dtype)

                # Scatter the expert output back to the correct positions
                final_output.index_add_(0, expert_indices, expert_output)

        final_output = final_output.view(batch_size, seq_len, d_model)
        
        return final_output, load_balancing_loss
