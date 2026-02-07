import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
from toolformer import ToolformerModel, APICallEncoder


class ToolformerLoss(nn.Module):
    """
    Toolformer loss function implementing the exact methodology from the paper.
    
    The loss combines:
    1. Standard language modeling loss (cross-entropy)
    2. API call filtering based on loss improvement
    3. Self-supervised learning signal for tool usage
    """
    
    def __init__(self, model: ToolformerModel, filter_threshold: float = 0.0):
        super().__init__()
        self.model = model
        self.filter_threshold = filter_threshold
        self.encoder = model.encoder
        self.tokenizer = model.tokenizer
        
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Toolformer loss
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels [batch_size, seq_len]
            
        Returns:
            Dictionary containing loss components
        """
        batch_size, seq_len = input_ids.shape
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']  # [batch_size, seq_len, vocab_size]
        
        language_modeling_loss = self._compute_language_modeling_loss(logits, labels, attention_mask)
        
        api_loss, api_metrics = self._compute_api_call_loss(input_ids, logits, labels, attention_mask)
        
        total_loss = language_modeling_loss + api_loss
        
        return {
            'loss': total_loss,
            'language_modeling_loss': language_modeling_loss,
            'api_loss': api_loss,
            'perplexity': torch.exp(language_modeling_loss),
            **api_metrics
        }
    
    def _compute_language_modeling_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                                      attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute standard language modeling cross-entropy loss"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        
        loss = self.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        loss = loss.view(shift_labels.shape)
        
        masked_loss = loss * shift_attention_mask.float()
        
        total_loss = masked_loss.sum()
        total_tokens = shift_attention_mask.sum()
        
        return total_loss / (total_tokens + 1e-8)
    
    def _compute_api_call_loss(self, input_ids: torch.Tensor, logits: torch.Tensor, 
                             labels: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss component related to API calls
        
        This implements the filtering mechanism from the paper where API calls
        are only kept if they reduce the language modeling loss.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        total_api_loss = torch.tensor(0.0, device=device)
        beneficial_calls = 0
        total_calls = 0
        
        for batch_idx in range(batch_size):
            sequence_ids = input_ids[batch_idx]
            sequence_text = self.tokenizer.decode(sequence_ids, skip_special_tokens=False)
            
            api_calls = self.encoder.decode_calls(sequence_text)
            
            if not api_calls:
                continue
            
            for call in api_calls:
                total_calls += 1
                
                original_text = self._remove_api_call_from_text(sequence_text, call)
                
                if self._is_api_call_beneficial(original_text, sequence_text):
                    beneficial_calls += 1
                    
                    api_positions = self._find_api_positions(sequence_ids, call)
                    if api_positions:
                        api_token_loss = self._compute_api_token_loss(
                            logits[batch_idx], labels[batch_idx], api_positions
                        )
                        total_api_loss += api_token_loss
        
        api_loss = total_api_loss / max(beneficial_calls, 1)
        
        metrics = {
            'beneficial_calls': beneficial_calls,
            'total_calls': total_calls,
            'call_acceptance_rate': beneficial_calls / max(total_calls, 1)
        }
        
        return api_loss, metrics
    
    def _remove_api_call_from_text(self, text: str, call: Dict[str, str]) -> str:
        """Remove a specific API call from text"""
        if call['result']:
            call_text = self.encoder.encode_call_with_result(
                call['tool_name'], call['input'], call['result']
            )
        else:
            call_text = self.encoder.encode_call(call['tool_name'], call['input'])
        
        return text.replace(call_text, '').strip()
    
    def _is_api_call_beneficial(self, original_text: str, augmented_text: str) -> bool:
        """
        Check if API call is beneficial according to Toolformer criteria
        
        An API call is beneficial if it reduces the language modeling loss
        on the remaining text sequence.
        """
        try:
            original_tokens = self.tokenizer.encode(original_text, return_tensors='pt')
            augmented_tokens = self.tokenizer.encode(augmented_text, return_tensors='pt')
            
            if original_tokens.shape[1] == 0 or augmented_tokens.shape[1] == 0:
                return False
            
            with torch.no_grad():
                original_outputs = self.model(original_tokens, labels=original_tokens)
                augmented_outputs = self.model(augmented_tokens, labels=augmented_tokens)
                
                if original_outputs['loss'] is None or augmented_outputs['loss'] is None:
                    return False
                
                improvement = original_outputs['loss'].item() - augmented_outputs['loss'].item()
                return improvement > self.filter_threshold
                
        except Exception:
            return False
    
    def _find_api_positions(self, sequence_ids: torch.Tensor, call: Dict[str, str]) -> Optional[List[int]]:
        """Find positions of API call tokens in the sequence"""
        sequence_text = self.tokenizer.decode(sequence_ids, skip_special_tokens=False)
        
        if call['result']:
            call_text = self.encoder.encode_call_with_result(
                call['tool_name'], call['input'], call['result']
            )
        else:
            call_text = self.encoder.encode_call(call['tool_name'], call['input'])
        
        call_tokens = self.tokenizer.encode(call_text, add_special_tokens=False)
        
        positions = []
        sequence_list = sequence_ids.tolist()
        
        for i in range(len(sequence_list) - len(call_tokens) + 1):
            if sequence_list[i:i+len(call_tokens)] == call_tokens:
                positions.extend(range(i, i + len(call_tokens)))
                break
        
        return positions if positions else None
    
    def _compute_api_token_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                              positions: List[int]) -> torch.Tensor:
        """Compute loss specifically for API call tokens"""
        if not positions:
            return torch.tensor(0.0, device=logits.device)
        
        api_losses = []
        for pos in positions:
            if pos < logits.shape[0] - 1:
                token_logits = logits[pos]
                token_label = labels[pos + 1]  # Next token prediction
                
                if token_label != -100:  # Valid label
                    token_loss = F.cross_entropy(
                        token_logits.unsqueeze(0), 
                        token_label.unsqueeze(0)
                    )
                    api_losses.append(token_loss)
        
        if api_losses:
            return torch.stack(api_losses).mean()
        else:
            return torch.tensor(0.0, device=logits.device)
    
    def compute_loss_improvement(self, original_text: str, augmented_text: str) -> float:
        """
        Compute loss improvement from adding API calls (used during training)
        
        This is the core filtering mechanism from the Toolformer paper.
        """
        try:
            original_tokens = self.tokenizer.encode(original_text, return_tensors='pt')
            augmented_tokens = self.tokenizer.encode(augmented_text, return_tensors='pt')
            
            with torch.no_grad():
                original_outputs = self.model(original_tokens, labels=original_tokens)
                augmented_outputs = self.model(augmented_tokens, labels=augmented_tokens)
                
                if original_outputs['loss'] is None or augmented_outputs['loss'] is None:
                    return 0.0
                
                return original_outputs['loss'].item() - augmented_outputs['loss'].item()
                
        except Exception as e:
            return 0.0
    
    def filter_api_calls_by_loss(self, text: str, candidate_calls: List[str]) -> List[str]:
        """Filter API calls based on loss improvement criterion"""
        beneficial_calls = []
        
        for call_text in candidate_calls:
            improvement = self.compute_loss_improvement(text, text + " " + call_text)
            if improvement > self.filter_threshold:
                beneficial_calls.append(call_text)
        
        return beneficial_calls


class ToolformerMetrics:
    """Metrics tracking for Toolformer training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.language_modeling_loss = 0.0
        self.api_loss = 0.0
        self.perplexity = 0.0
        self.beneficial_calls = 0
        self.total_calls = 0
        self.num_batches = 0
    
    def update(self, loss_dict: Dict[str, torch.Tensor]):
        """Update metrics with loss dictionary"""
        self.total_loss += loss_dict['loss'].item()
        self.language_modeling_loss += loss_dict['language_modeling_loss'].item()
        self.api_loss += loss_dict['api_loss'].item()
        self.perplexity += loss_dict['perplexity'].item()
        self.beneficial_calls += loss_dict['beneficial_calls']
        self.total_calls += loss_dict['total_calls']
        self.num_batches += 1
    
    def compute_averages(self) -> Dict[str, float]:
        """Compute average metrics"""
        if self.num_batches == 0:
            return {}
        
        return {
            'avg_total_loss': self.total_loss / self.num_batches,
            'avg_language_modeling_loss': self.language_modeling_loss / self.num_batches,
            'avg_api_loss': self.api_loss / self.num_batches,
            'avg_perplexity': self.perplexity / self.num_batches,
            'total_beneficial_calls': self.beneficial_calls,
            'total_calls': self.total_calls,
            'call_acceptance_rate': self.beneficial_calls / max(self.total_calls, 1)
        }
    
    def log_metrics(self, step: int, logger):
        """Log metrics to logger"""
        metrics = self.compute_averages()
        for key, value in metrics.items():
            logger.info(f"Step {step} - {key}: {value:.4f}")