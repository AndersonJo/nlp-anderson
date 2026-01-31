# Kepler 442c Sci-Fi RAG with LangChain + FAISS

Korean sci-fi world RAG example using:
- **LangChain** for RAG orchestration
- **FAISS** for vector storage
- **Local gpt-oss:20b** model
- **Korean embeddings** (ko-sroberta-multitask)
- **Kepler 442c sci-fi dataset** (completely fictional world)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the example
python rag_langchain_faiss.py
```

## Files

- `rag_langchain_faiss.py` - Main LangChain + FAISS RAG implementation
- `kepler_442c_data.csv` - Korean sci-fi world knowledge dataset
- `requirements.txt` - Python dependencies

## Features

- ✅ **LangChain RetrievalQA** chains
- ✅ **FAISS vector store** for fast similarity search
- ✅ **Korean text embeddings** optimized for Korean language
- ✅ **Local LLM** (no API calls)
- ✅ **8-bit quantization** for memory efficiency
- ✅ **Fallback mode** (retrieval-only if LLM fails)

## How it works

1. **Document Loading**: CSV data → LangChain Documents
2. **Text Splitting**: RecursiveCharacterTextSplitter chunks documents
3. **Embeddings**: Korean text → vectors using ko-sroberta-multitask
4. **Vector Store**: FAISS index for fast similarity search
5. **Retrieval**: Query → similar document retrieval
6. **Generation**: Retrieved context + query → LLM response

## Dataset - Kepler 442c Sci-Fi World

Completely fictional alien world with unique characteristics:
- **Physics**: 2.3x Earth gravity, 4.7x atmospheric pressure, 47-hour days
- **Life Forms**: Crystal Trees, Plasma Birds, Gravity Mollusks, Gellatian aliens
- **Technology**: Dimensional jump devices, gravity wave communication, time crystals
- **Environment**: Dual suns, xenon aurora, electromagnetic storms, methane seas
- **Culture**: Gellatian civilization, resonance festivals, telepathic communication

Perfect for testing RAG with completely novel information that LLMs weren't trained on!