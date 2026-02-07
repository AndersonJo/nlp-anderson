#!/usr/bin/env python3
"""
Inference script for Toolformer model.
Provides methods for text generation with tool integration.
"""

import torch
import re
from typing import List, Dict, Optional, Tuple
import logging

from toolformer import ToolformerModel, ToolformerConfig, APICallEncoder
from tools import get_all_tools, ToolRegistry


class ToolformerInference:
    """Inference engine for Toolformer model"""
    
    def __init__(self, model: ToolformerModel, temperature: float = 1.0, max_length: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_length = max_length
        self.encoder = model.encoder
        self.tokenizer = model.tokenizer
        self.tool_registry = ToolRegistry()
        
        self.logger = logging.getLogger(__name__)
        
        self.model.eval()
    
    def generate(self, prompt: str, do_sample: bool = True, num_return_sequences: int = 1) -> List[str]:
        """
        Generate text with tool integration
        
        Args:
            prompt: Input text prompt
            do_sample: Whether to use sampling or greedy decoding
            num_return_sequences: Number of sequences to return
        
        Returns:
            List of generated text sequences
        """
        results = []
        
        for _ in range(num_return_sequences):
            generated_text = self._generate_single(prompt, do_sample)
            results.append(generated_text)
        
        return results
    
    def _generate_single(self, prompt: str, do_sample: bool = True) -> str:
        """Generate a single text sequence with tool integration"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        generated_ids = input_ids.clone()
        
        for step in range(self.max_length - input_ids.size(1)):
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs['logits'][:, -1, :] / self.temperature
                
                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                current_text = self.tokenizer.decode(generated_ids[0])
                
                if self._should_execute_tools(current_text):
                    current_text = self._execute_pending_tools(current_text)
                    generated_ids = self.tokenizer.encode(current_text, return_tensors='pt')
                
                if self._should_stop_generation(current_text, next_token):
                    break
        
        final_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        return self._post_process_output(final_text)
    
    def _should_execute_tools(self, text: str) -> bool:
        """Check if there are pending tool calls to execute"""
        api_calls = self.encoder.decode_calls(text)
        
        for call in api_calls:
            if call['result'] is None or call['result'] == '':
                call_without_result = self.encoder.encode_call(call['tool_name'], call['input'])
                if call_without_result in text and self.model.config.api_result_token not in text.split(call_without_result)[-1]:
                    return True
        
        return self.model.config.api_result_token in text
    
    def _execute_pending_tools(self, text: str) -> str:
        """Execute any pending tool calls in the text"""
        modified_text = text
        
        api_calls = self.encoder.decode_calls(text)
        
        for call in api_calls:
            if call['result'] is None or call['result'] == '':
                call_without_result = self.encoder.encode_call(call['tool_name'], call['input'])
                
                if call_without_result in modified_text:
                    result = self.tool_registry.execute_tool(call['tool_name'], call['input'])
                    call_with_result = self.encoder.encode_call_with_result(
                        call['tool_name'], call['input'], result
                    )
                    
                    modified_text = modified_text.replace(call_without_result, call_with_result)
                    self.logger.debug(f"Executed {call['tool_name']}({call['input']}) = {result}")
        
        arrow_pattern = rf"({re.escape(self.model.config.api_start_token)}[^{re.escape(self.model.config.api_end_token)}]*?){re.escape(self.model.config.api_result_token)}"
        
        def execute_arrow_call(match):
            call_start = match.group(1)
            
            call_match = re.search(rf'(\w+)\s*\((.*?)\)', call_start)
            if call_match:
                tool_name, input_str = call_match.groups()
                result = self.tool_registry.execute_tool(tool_name.strip(), input_str.strip())
                return f"{call_start} {result} {self.model.config.api_end_token}"
            
            return match.group(0)
        
        modified_text = re.sub(arrow_pattern, execute_arrow_call, modified_text)
        
        return modified_text
    
    def _should_stop_generation(self, text: str, last_token: torch.Tensor) -> bool:
        """Check if generation should stop"""
        if self.tokenizer.eos_token_id and last_token.item() == self.tokenizer.eos_token_id:
            return True
        
        if text.endswith(('.', '!', '?')) and len(text.split()) > 10:
            return True
        
        return False
    
    def _post_process_output(self, text: str) -> str:
        """Post-process the generated output"""
        text = text.strip()
        
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(rf'{re.escape(self.model.config.api_start_token)}\s*(\w+)\s*\((.*?)\)\s*{re.escape(self.model.config.api_result_token)}\s*(.*?)\s*{re.escape(self.model.config.api_end_token)}',
                     r'<API> \1(\2) → \3 </API>', text)
        
        return text
    
    def complete_with_tools(self, text: str) -> str:
        """Complete text by executing any existing tool calls"""
        return self._execute_pending_tools(text)
    
    def explain_tool_usage(self, text: str) -> Dict[str, List[Dict]]:
        """Explain what tools were used in the generated text"""
        api_calls = self.encoder.decode_calls(text)
        
        tool_usage = {}
        for call in api_calls:
            tool_name = call['tool_name']
            if tool_name not in tool_usage:
                tool_usage[tool_name] = []
            
            tool_usage[tool_name].append({
                'input': call['input'],
                'result': call['result'],
                'description': self.tool_registry.get_tool(tool_name).description() if self.tool_registry.get_tool(tool_name) else "Unknown tool"
            })
        
        return tool_usage
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[List[str]]:
        """Generate text for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results
    
    def interactive_demo(self):
        """Run an interactive demo"""
        print("Toolformer Interactive Demo")
        print("Type 'quit' to exit, 'tools' to see available tools")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nPrompt: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'tools':
                    print("\nAvailable tools:")
                    for name, description in self.tool_registry.get_tool_descriptions().items():
                        print(f"  - {name}: {description}")
                    continue
                elif not user_input:
                    continue
                
                print("\nGenerating...")
                results = self.generate(user_input, num_return_sequences=1)
                
                for i, result in enumerate(results):
                    print(f"\nGenerated text {i+1}:")
                    print(result)
                    
                    tool_usage = self.explain_tool_usage(result)
                    if tool_usage:
                        print("\nTools used:")
                        for tool_name, calls in tool_usage.items():
                            print(f"  {tool_name}:")
                            for call in calls:
                                print(f"    Input: {call['input']}")
                                print(f"    Result: {call['result']}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nDemo ended.")


def create_inference_examples():
    """Create example prompts for testing inference"""
    return [
        "Calculate the area of a rectangle with length 12 and width 8.",
        "What is the capital of Germany? Also, what is 5 times 7?",
        "Today is 2024-01-15. What day of the week is it?",
        "How do you say 'thank you' in French?",
        "If I have 100 dollars and spend 25% on food, how much do I have left?",
        "What is the square root of 225?",
        "Tell me about Albert Einstein and calculate 1879 + 76.",
        "The temperature is 25°C. What would that be in Fahrenheit using the formula F = C * 9/5 + 32?",
        "Search for information about machine learning.",
        "What is 7 factorial?"
    ]


def main():
    """Test inference functionality"""
    logging.basicConfig(level=logging.INFO)
    
    from main import ToolformerTrainingPipeline, ToolformerConfig, TrainingConfig
    
    config = ToolformerConfig()
    training_config = TrainingConfig(batch_size=1)
    
    pipeline = ToolformerTrainingPipeline(config, training_config)
    inference = ToolformerInference(pipeline.model, temperature=0.8)
    
    print("Testing Toolformer Inference")
    print("=" * 40)
    
    examples = create_inference_examples()
    
    for i, prompt in enumerate(examples[:3]):  # Test first 3 examples
        print(f"\nExample {i+1}: {prompt}")
        results = inference.generate(prompt, num_return_sequences=1)
        
        for result in results:
            print(f"Generated: {result}")
            
            tool_usage = inference.explain_tool_usage(result)
            if tool_usage:
                print("Tools used:")
                for tool_name, calls in tool_usage.items():
                    for call in calls:
                        print(f"  {tool_name}({call['input']}) = {call['result']}")
    
    print(f"\nStarting interactive demo...")
    inference.interactive_demo()


if __name__ == "__main__":
    main()