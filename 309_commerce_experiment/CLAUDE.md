# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a commerce POC (Proof of Concept) that demonstrates product comparison using the Kimi-K2-Instruct local LLM model. The application searches through Korean product data and uses AI to provide intelligent product comparisons and recommendations.

## Commands

### Development Setup
```bash
# Install dependencies (PyTorch, Transformers, etc.)
pip install -r requirements.txt

# Run the main application
python commerce_poc.py

# Run with specific model
python commerce_poc.py "moonshotai/Kimi-K2-Instruct"
```

### System Requirements
- **Memory**: Minimum 250GB RAM/VRAM for 4-bit quantized model
- **GPU**: NVIDIA GPU with CUDA support recommended
- **Storage**: 100GB+ for model download

## Architecture

### Core Components

1. **Product**: Dataclass representing product information (ID, name, category, price, description, features, pros/cons)

2. **ProductSearcher**: Handles CSV-based product search with keyword matching
   - Loads products from `products.csv` (20 Korean products)
   - Implements simple scoring-based search algorithm

3. **LocalLLMClient**: Manages local Kimi-K2-Instruct model
   - Auto-downloads model from Hugging Face on first run
   - Uses 4-bit quantization for memory optimization
   - Supports both GPU and CPU execution
   - Implements chat template formatting for Kimi model

4. **CommercePOC**: Main application class that orchestrates the terminal interface
   - Interactive search loop
   - Integrates product search with LLM analysis

### Model Integration

The application uses the Kimi-K2-Instruct model (1 trillion parameters, 32B activated) with:
- 4-bit quantization via BitsAndBytesConfig
- Automatic device mapping (GPU/CPU)
- Chat template formatting for proper model interaction
- Korean language support for product analysis

### Data Flow

1. User enters search query in Korean/English
2. ProductSearcher finds matching products via keyword scoring
3. LocalLLMClient formats products and query into structured prompt
4. Kimi model analyzes products and provides recommendations
5. Response formatted with customer intent analysis, product comparison, ranking, and advice

### Error Handling

- Graceful fallback when model loading fails
- CSV parsing error handling
- Memory availability checks
- Import dependency validation