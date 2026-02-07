import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import re
import json
from abc import ABC, abstractmethod
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from dataclasses import dataclass


@dataclass
class ToolformerConfig:
    """Configuration for Toolformer model"""
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int = 3072
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    
    api_start_token: str = "<API>"
    api_end_token: str = "</API>"
    api_result_token: str = "→"
    
    filter_threshold: float = 0.0
    top_k_candidates: int = 10
    temperature: float = 1.0


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def name(self) -> str:
        """Return the tool name"""
        pass
    
    @abstractmethod
    def execute(self, input_str: str) -> str:
        """Execute the tool with given input and return result"""
        pass
    
    @abstractmethod
    def description(self) -> str:
        """Return tool description for prompting"""
        pass


class Calculator(Tool):
    """Calculator tool for arithmetic operations"""
    
    def name(self) -> str:
        return "Calculator"
    
    def execute(self, input_str: str) -> str:
        try:
            result = eval(input_str.strip())
            return str(result)
        except Exception:
            return "Error: Invalid calculation"
    
    def description(self) -> str:
        return "Calculator(expression) - Evaluates mathematical expressions"


class QASystem(Tool):
    """Simple QA system tool"""
    
    def name(self) -> str:
        return "QA"
    
    def execute(self, input_str: str) -> str:
        return f"Answer to '{input_str.strip()}': [QA Result Placeholder]"
    
    def description(self) -> str:
        return "QA(question) - Answers questions using knowledge base"


class APICallEncoder:
    """Encodes and decodes API calls to/from text sequences"""
    
    def __init__(self, config: ToolformerConfig):
        self.config = config
        self.api_call_pattern = re.compile(
            rf'{re.escape(config.api_start_token)}\s*(\w+)\s*\((.*?)\)\s*(?:{re.escape(config.api_result_token)}\s*(.*?))?\s*{re.escape(config.api_end_token)}'
        )
    
    def encode_call(self, tool_name: str, input_str: str) -> str:
        """Encode API call without result"""
        return f"{self.config.api_start_token} {tool_name}({input_str}) {self.config.api_end_token}"
    
    def encode_call_with_result(self, tool_name: str, input_str: str, result: str) -> str:
        """Encode API call with result"""
        return f"{self.config.api_start_token} {tool_name}({input_str}) {self.config.api_result_token} {result} {self.config.api_end_token}"
    
    def decode_calls(self, text: str) -> List[Dict[str, str]]:
        """Extract all API calls from text"""
        matches = self.api_call_pattern.findall(text)
        calls = []
        for match in matches:
            tool_name, input_str, result = match
            calls.append({
                'tool_name': tool_name.strip(),
                'input': input_str.strip(),
                'result': result.strip() if result else None
            })
        return calls
    
    def remove_results(self, text: str) -> str:
        """Remove results from API calls, keeping only the call part"""
        def replace_func(match):
            full_match = match.group(0)
            if self.config.api_result_token in full_match:
                call_part = full_match.split(self.config.api_result_token)[0].strip()
                return call_part + f" {self.config.api_end_token}"
            return full_match
        
        return self.api_call_pattern.sub(replace_func, text)


class ToolformerModel(nn.Module):
    """Main Toolformer model combining language model with tool integration"""
    
    def __init__(self, config: ToolformerConfig, tools: List[Tool]):
        super().__init__()
        self.config = config
        self.tools = {tool.name(): tool for tool in tools}
        
        gpt2_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.n_positions,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_inner=config.n_inner,
            activation_function=config.activation_function,
            resid_pdrop=config.resid_pdrop,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
            layer_norm_epsilon=config.layer_norm_epsilon,
            initializer_range=config.initializer_range,
        )
        
        self.language_model = GPT2LMHeadModel(gpt2_config)
        self.encoder = APICallEncoder(config)
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        special_tokens = [
            config.api_start_token,
            config.api_end_token,
            config.api_result_token
        ]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        # Set pad token to eos token to avoid padding issues
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        
        self.api_start_token_id = self.tokenizer.convert_tokens_to_ids(config.api_start_token)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return {
            'logits': outputs.logits,
            'loss': outputs.loss,
            'hidden_states': outputs.hidden_states if outputs.hidden_states is not None else None
        }
    
    def compute_api_start_probabilities(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute probabilities of API start token at each position"""
        with torch.no_grad():
            outputs = self.forward(input_ids)
            logits = outputs['logits']
            
            api_start_logits = logits[:, :, self.api_start_token_id]
            api_start_probs = torch.softmax(api_start_logits, dim=-1)
            
            return api_start_probs
    
    def sample_api_candidates(self, text: str, top_k: int = None) -> List[int]:
        """Sample candidate positions for API calls"""
        if top_k is None:
            top_k = self.config.top_k_candidates
            
        tokens = self.tokenizer.encode(text, return_tensors='pt')
        api_probs = self.compute_api_start_probabilities(tokens)
        
        # Handle case where sequence is shorter than top_k
        seq_len = api_probs.squeeze().shape[0] if api_probs.squeeze().dim() > 0 else 1
        k = min(top_k, seq_len)
        
        if k > 0:
            positions = torch.topk(api_probs.squeeze(), k).indices.tolist()
            # Handle case where topk returns a single value (not a list)
            if not isinstance(positions, list):
                positions = [positions]
        else:
            positions = []
            
        return positions
    
    def execute_tool(self, tool_name: str, input_str: str) -> str:
        """Execute a specific tool"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(input_str)
    
    def generate_with_tools(self, input_text: str, max_length: int = 512,
                           temperature: float = None) -> str:
        """Generate text with tool integration"""
        if temperature is None:
            temperature = self.config.temperature
            
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        generated_ids = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            with torch.no_grad():
                outputs = self.forward(generated_ids)
                logits = outputs['logits'][:, -1, :] / temperature
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                current_text = self.tokenizer.decode(generated_ids[0])
                
                if self.config.api_result_token in current_text:
                    api_calls = self.encoder.decode_calls(current_text)
                    for call in api_calls:
                        if call['result'] is None:
                            result = self.execute_tool(call['tool_name'], call['input'])
                            
                            partial_call = self.encoder.encode_call(call['tool_name'], call['input'])
                            full_call = self.encoder.encode_call_with_result(
                                call['tool_name'], call['input'], result
                            )
                            
                            current_text = current_text.replace(partial_call, full_call)
                            generated_ids = self.tokenizer.encode(current_text, return_tensors='pt')
                
                if generated_ids.size(1) >= max_length:
                    break
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    
    def compute_loss_improvement(self, original_text: str, augmented_text: str) -> float:
        """Compute loss improvement from adding API calls"""
        original_tokens = self.tokenizer.encode(original_text, return_tensors='pt')
        augmented_tokens = self.tokenizer.encode(augmented_text, return_tensors='pt')
        
        with torch.no_grad():
            original_outputs = self.forward(original_tokens, labels=original_tokens)
            augmented_outputs = self.forward(augmented_tokens, labels=augmented_tokens)
            
            original_loss = original_outputs['loss'].item()
            augmented_loss = augmented_outputs['loss'].item()
            
            return original_loss - augmented_loss
    
    def filter_api_calls(self, text: str, candidate_calls: List[str]) -> List[str]:
        """Filter API calls based on loss improvement"""
        filtered_calls = []
        
        for call in candidate_calls:
            improvement = self.compute_loss_improvement(text, call)
            if improvement > self.config.filter_threshold:
                filtered_calls.append(call)
        
        return filtered_calls