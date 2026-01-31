# Toolformer Implementation

A complete PyTorch implementation of Toolformer from the paper "Toolformer: Language Models Can Teach Themselves to Use Tools" (https://arxiv.org/abs/2302.04761).

## Overview

**What is Toolformer?**

Toolformer is a language model that learns to use external tools in a self-supervised manner. Unlike traditional approaches that require extensive human annotation, Toolformer autonomously learns when to call APIs, which APIs to call, what arguments to pass, and how to incorporate the results into future token prediction.

**Key Features:**
- **Self-supervised learning**: No manual annotation of tool usage required
- **API call filtering**: Only keeps API calls that improve language modeling performance
- **Multiple tools**: Supports calculator, QA system, search, calendar, translation, and more
- **Production-ready**: Complete training and inference pipeline

## Model Architecture

The implementation is based on GPT-2 architecture with the following key components:

1. **Base Language Model**: GPT-2 transformer with tool integration capabilities
2. **API Call Encoder**: Encodes tool calls as special text sequences: `<API> tool_name(input) → result </API>`
3. **Tool Interface**: Abstract base class for implementing different tools
4. **Self-supervised Learning**: Automatically generates training data with beneficial API calls

## Loss Function

The loss function combines:
1. **Standard Language Modeling Loss**: Cross-entropy loss for next token prediction
2. **API Call Filtering**: Only keeps API calls that reduce overall model loss
3. **Tool Usage Optimization**: Learns optimal tool usage patterns

The filtering mechanism is the core innovation - API calls are only retained if they improve future token prediction performance.

## Directory Structure

```
312_toolformer/
├── toolformer.py       # Core model architecture
├── tools.py           # Tool implementations
├── training.py        # Training pipeline and dataset
├── loss.py           # Loss function implementation
├── inference.py      # Inference engine
├── main.py          # Main training/inference script
├── test_toolformer.py # Comprehensive tests
├── requirements.txt  # Dependencies
└── README.md        # This file
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd 006_toolformer

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train with default settings
python t01_gpt_oss_inference.py --mode train --epochs 3 --batch-size 8

# Train with custom data
python t01_gpt_oss_inference.py --mode train --data-path data.txt --epochs 5
```

### Inference

```bash
# Interactive mode
python t01_gpt_oss_inference.py --mode interactive --model-path checkpoints/toolformer.pt

# Generate text
python t01_gpt_oss_inference.py --mode generate --model-path checkpoints/toolformer.pt
```

### Testing

```bash
# Run all tests
python test_toolformer.py

# Run inference demo
python inference.py
```

## Usage Examples

### Basic Training

```python
from toolformer import ToolformerModel, ToolformerConfig
from tools import get_all_tools
from training import ToolformerTrainer, TrainingConfig

# Initialize model
config = ToolformerConfig()
tools = get_all_tools()
model = ToolformerModel(config, tools)

# Setup training
train_config = TrainingConfig(batch_size=8, num_epochs=3)
trainer = ToolformerTrainer(model, train_config)

# Train
training_data = ["What is 2+3?", "Calculate the area..."]
trainer.train(training_data)
```

### Custom Tools

```python
from tools import Tool

class CustomTool(Tool):
    def name(self) -> str:
        return "CustomTool"
    
    def execute(self, input_str: str) -> str:
        # Your custom logic here
        return f"Result: {input_str}"
    
    def description(self) -> str:
        return "CustomTool(input) - Does something custom"

# Add to model
tools = get_all_tools()
tools.append(CustomTool())
model = ToolformerModel(config, tools)
```

### Inference

```python
from inference import ToolformerInference

# Initialize inference
inference = ToolformerInference(model)

# Generate text
result = inference.generate("What is 15% of 200?")
print(result)
# Output: "What is 15% of 200? <API> Calculator(200 * 0.15) → 30 </API> The answer is 30."

# Explain tool usage
usage = inference.explain_tool_usage(result)
print(usage)
# Output: {'Calculator': [{'input': '200 * 0.15', 'result': '30', 'description': '...'}]}
```

## Available Tools

The implementation includes the following tools as specified in the paper:

1. **Calculator**: Evaluates mathematical expressions
   - Example: `Calculator(2+3*4)` → `14`

2. **QA System**: Answers factual questions
   - Example: `QA(What is the capital of France?)` → `Paris`

3. **Search Engine**: Searches for information
   - Example: `Search(machine learning)` → `Machine learning is...`

4. **Calendar**: Handles date/time operations
   - Example: `Calendar(today)` → `2024-01-15`

5. **Translator**: Translates between languages
   - Example: `Translator(hello to spanish)` → `hola`

6. **Wikipedia**: Searches Wikipedia
   - Example: `Wikipedia(Albert Einstein)` → `Albert Einstein was...`

## Training Details

### Self-supervised Learning Process

1. **Candidate Generation**: Sample potential API call positions using `<API>` token probability
2. **API Call Generation**: Generate candidate API calls for each position
3. **Execution**: Execute API calls and collect results
4. **Filtering**: Keep only API calls that reduce language modeling loss
5. **Training**: Fine-tune on augmented dataset with beneficial API calls

### Loss Function Components

The loss function implements the exact methodology from the paper:

```python
total_loss = language_modeling_loss + api_loss

# Where:
# - language_modeling_loss: Standard cross-entropy loss
# - api_loss: Loss for beneficial API call tokens
```

### Filtering Criterion

API calls are kept if and only if:
```
loss_improvement = loss(original_text) - loss(text_with_api_call) > threshold
```

## Configuration

### Model Configuration

```python
config = ToolformerConfig(
    vocab_size=50257,      # GPT-2 vocabulary size
    n_positions=1024,      # Maximum sequence length
    n_embd=768,           # Embedding dimension
    n_layer=12,           # Number of transformer layers
    n_head=12,            # Number of attention heads
    
    # Tool-specific settings
    api_start_token="<API>",
    api_end_token="</API>",
    api_result_token="→",
    filter_threshold=0.0,  # Loss improvement threshold
    top_k_candidates=10    # API candidate positions to consider
)
```

### Training Configuration

```python
training_config = TrainingConfig(
    batch_size=8,
    learning_rate=5e-5,
    num_epochs=3,
    warmup_steps=1000,
    
    # Augmentation settings
    augmentation_samples=5,        # Augmentations per text
    min_improvement_threshold=0.01, # Minimum loss improvement
    max_api_calls_per_text=3       # Maximum API calls per example
)
```

## Evaluation

The implementation includes comprehensive evaluation metrics:

- **Language Modeling Metrics**: Perplexity, cross-entropy loss
- **Tool Usage Metrics**: API call acceptance rate, beneficial calls ratio
- **Performance Metrics**: Training time, inference speed

Run evaluation:
```bash
python t01_gpt_oss_inference.py --mode evaluate --model-path checkpoints/toolformer.pt --data-path test_data.txt
```

## Testing

Comprehensive test suite covering:

- ✅ API call encoding/decoding
- ✅ Individual tool functionality
- ✅ Model forward pass and training
- ✅ Loss function computation
- ✅ Self-supervised learning pipeline
- ✅ Inference and generation
- ✅ End-to-end integration

Run tests:
```bash
python test_toolformer.py
```

## Performance Notes

### Small Model for Development

The default configuration uses a smaller model (768 dimensions, 12 layers) for faster development and testing. For production use, consider:

- GPT-J 6B parameters (as in the paper)
- Larger embedding dimensions (4096+)
- More transformer layers (28+)

### Training Tips

1. **Data Quality**: Use diverse text with natural opportunities for tool usage
2. **Filtering Threshold**: Adjust `filter_threshold` based on your data
3. **Tool Selection**: Include tools relevant to your domain
4. **Batch Size**: Larger batches improve training stability

## Implementation Accuracy

This implementation closely follows the Toolformer paper:

✅ **Exact API Call Format**: `<API> tool(input) → result </API>`  
✅ **Self-supervised Learning**: No manual annotation required  
✅ **Loss-based Filtering**: Only beneficial API calls are kept  
✅ **GPT-based Architecture**: Built on transformer language model  
✅ **Multiple Tools**: Calculator, QA, Search, Calendar, Translation  
✅ **Training Pipeline**: Complete end-to-end training process  

## Limitations

- **Tool Execution**: Tools are simplified implementations (not real APIs)
- **Model Size**: Uses smaller model for development (not 6B+ parameters)
- **Training Data**: Includes sample data (replace with domain-specific data)
- **Performance**: Not optimized for large-scale production use

## Future Improvements

1. **Real API Integration**: Connect to actual APIs (Wolfram Alpha, Google Search, etc.)
2. **Larger Models**: Support for 6B+ parameter models
3. **Advanced Tools**: More sophisticated tool implementations
4. **Distributed Training**: Multi-GPU training support
5. **Evaluation Suite**: Comprehensive benchmark evaluations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{schick2023toolformer,
  title={Toolformer: Language Models Can Teach Themselves to Use Tools},
  author={Schick, Timo and Dwivedi-Yu, Jane and Dess{\`i}, Roberto and Raileanu, Roberta and Lomeli, Maria and Zettlemoyer, Luke and Cancedda, Nicola and Scialom, Thomas},
  journal={arXiv preprint arXiv:2302.04761},
  year={2023}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.