#!/usr/bin/env python3
"""
Simple demo to show Toolformer functionality.
"""

from toolformer import ToolformerModel, ToolformerConfig
from tools import Calculator, QASystem, Calendar
from inference import ToolformerInference

def main():
    print("🤖 Toolformer Demo")
    print("=" * 40)
    
    # Create a small model for demo
    config = ToolformerConfig(
        n_embd=128,
        n_layer=2,
        n_head=2,
        n_positions=256
    )
    
    # Initialize with basic tools
    tools = [Calculator(), QASystem(), Calendar()]
    model = ToolformerModel(config, tools)
    
    print(f"✅ Model initialized with {len(tools)} tools:")
    for tool in tools:
        print(f"   - {tool.name()}: {tool.description()}")
    
    print("\n📋 Testing Tools Directly:")
    print("-" * 30)
    
    calc = Calculator()
    print(f"Calculator(2+3*4): {calc.execute('2+3*4')}")
    print(f"Calculator(sqrt(16)): {calc.execute('4**0.5')}")  # Since eval doesn't have sqrt
    
    qa = QASystem()
    print(f"QA(capital of France): {qa.execute('What is the capital of France?')}")
    
    calendar = Calendar()
    print(f"Calendar(today): {calendar.execute('today')}")
    
    print("\n🔧 Testing API Call Encoding:")
    print("-" * 30)
    
    encoder = model.encoder
    call = encoder.encode_call("Calculator", "5*7")
    print(f"Encoded call: {call}")
    
    call_with_result = encoder.encode_call_with_result("Calculator", "5*7", "35")
    print(f"Call with result: {call_with_result}")
    
    decoded = encoder.decode_calls(call_with_result)
    print(f"Decoded: {decoded}")
    
    print("\n🎯 Testing Model Components:")
    print("-" * 30)
    
    # Test tokenization
    text = "What is 2+3?"
    tokens = model.tokenizer.encode(text, return_tensors='pt')
    print(f"Text: '{text}'")
    print(f"Tokens shape: {tokens.shape}")
    
    # Test forward pass
    outputs = model(tokens)
    print(f"Model output logits shape: {outputs['logits'].shape}")
    
    # Test API candidate sampling
    candidates = model.sample_api_candidates(text, top_k=3)
    print(f"API candidate positions: {candidates}")
    
    print("\n🧠 Testing Inference Engine:")
    print("-" * 30)
    
    inference = ToolformerInference(model, temperature=1.0, max_length=50)
    
    # Test basic generation (will be random with untrained model)
    result = inference.generate("Hello", num_return_sequences=1)[0]
    print(f"Generated text: {result[:100]}...")  # First 100 chars
    
    # Test tool execution in text
    test_text = "The answer is <API> Calculator(3*8) </API>."
    completed = inference.complete_with_tools(test_text)
    print(f"Before: {test_text}")
    print(f"After: {completed}")
    
    # Explain tool usage
    usage = inference.explain_tool_usage(completed)
    if usage:
        print(f"Tools used: {list(usage.keys())}")
    
    print("\n✨ Demo Complete!")
    print("The model architecture is ready for training.")
    print("Run 'python main.py --mode train' to start training.")
    print("Run 'python main.py --mode interactive' for interactive mode.")


if __name__ == "__main__":
    main()