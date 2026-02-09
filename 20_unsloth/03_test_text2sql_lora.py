"""
Test Script for Text-to-SQL LoRA Model

This script tests the fine-tuned LoRA model for text-to-SQL generation.
It loads the base model with the trained LoRA adapters and runs inference
on sample SQL generation tasks.

Usage:
    python 03_test_text2sql_lora.py
"""
from unsloth import FastLanguageModel

print("Loading model...")

from transformers import TextStreamer


def load_model_with_lora(
    base_model: str = "unsloth/gpt-oss-20b",
    lora_path: str = "./text2sql_lora_model",
    max_seq_length: int = 4096
):
    """
    Load base model and apply trained LoRA weights.
    
    Args:
        base_model: Base model name
        lora_path: Path to saved LoRA weights
        max_seq_length: Maximum sequence length
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,  # Load directly from LoRA path (includes adapter config)
        dtype="bfloat16",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        device_map="cuda"
    )
    
    # Set to inference mode
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer


def generate_sql(
    model,
    tokenizer,
    schema: str,
    question: str,
    max_new_tokens: int = 512,
    stream: bool = True
) -> str:
    """
    Generate SQL from natural language question.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        schema: Database schema description
        question: Natural language question
        max_new_tokens: Maximum tokens to generate
        stream: Whether to stream output
    
    Returns:
        Generated SQL query
    """
    system_prompt = """You are an expert SQL assistant. Given a database schema and a natural language question, generate the correct SQL query.

Always respond with valid SQL that can be executed directly.
Use proper SQL formatting and follow best practices."""

    user_content = f"""### Schema:
{schema}

### Question:
{question}

### SQL Query:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="low",
    ).to("cuda")
    
    streamer = TextStreamer(tokenizer) if stream else None
    
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        use_cache=True
    )
    
    # Decode the generated tokens (excluding input tokens)
    generated_tokens = output[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text


# =============================================================================
# Test Cases
# =============================================================================

TEST_CASES = [
    {
        "name": "Simple JOIN - Employees and Departments",
        "schema": """
CREATE TABLE employees (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    department_id INT,
    salary DECIMAL(10, 2)
);

CREATE TABLE departments (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    location VARCHAR(100)
);
""",
        "question": "Find all employees and their department names where the department is located in New York."
    },
    {
        "name": "Simple JOIN - Orders and Customers",
        "schema": """
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(100),
    email VARCHAR(100)
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    total_amount DECIMAL(10, 2)
);
""",
        "question": "List all orders with customer names for orders placed in 2024."
    },
    {
        "name": "Simple JOIN - Products and Categories",
        "schema": """
CREATE TABLE categories (
    category_id INT PRIMARY KEY,
    category_name VARCHAR(50)
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(100),
    category_id INT,
    price DECIMAL(10, 2),
    stock INT
);
""",
        "question": "Show all products with their category names where stock is less than 10."
    },
]


def main():
    print("=" * 60)
    print("Text-to-SQL LoRA Model Test")
    print("=" * 60)
    
    # Load model with LoRA weights
    print("\n[1/2] Loading model with LoRA weights...")
    model, tokenizer = load_model_with_lora()
    print("Model loaded successfully!")
    
    # Run test cases
    print("\n[2/2] Running test cases...")
    print("=" * 60)
    
    results = []
    
    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['name']}")
        print("="*60)
        print(f"\nSchema:\n{test['schema']}")
        print(f"\nQuestion: {test['question']}")
        print(f"\nGenerated SQL:")
        print("-" * 40)
        
        generated_sql = generate_sql(
            model=model,
            tokenizer=tokenizer,
            schema=test['schema'],
            question=test['question'],
            stream=True
        )
        
        results.append({
            "test_name": test['name'],
            "question": test['question'],
            "generated_sql": generated_sql
        })
        
        print("-" * 40)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['test_name']}")
        print(f"   Q: {result['question'][:50]}...")
        sql_preview = result['generated_sql'].strip().replace('\n', ' ')[:80]
        print(f"   SQL: {sql_preview}...")
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
