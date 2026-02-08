"""
Dataset preprocessing utilities for fine-tuning.

These functions format raw datasets into the chat format expected by the model.

Usage:
    from gpt_oss import GptOssModel, preprocess_sql_dataset
    
    model = GptOssModel(...)
    dataset = preprocess_sql_dataset(
        raw_dataset['train'],
        model.formatter,
        complexity_filter='single join'
    )
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset
    from .formatter import ChatFormatter


def format_sql_example(example: dict, formatter: "ChatFormatter") -> dict:
    """
    Convert a SQL dataset row into chat format for fine-tuning.
    
    Output format (SQL query only, no explanation):
    - System: Provides domain context and instructs the model
    - User: Database schema + natural language question
    - Assistant: SQL query only
    
    Args:
        example: Raw dataset row with keys:
            - domain: SQL domain (e.g., 'e-commerce', 'healthcare')
            - sql_context: Database schema
            - sql_prompt: Natural language question
            - sql: Target SQL query
        formatter: ChatFormatter instance
    
    Returns:
        Dict with 'text' key containing formatted prompt
    """
    system_prompt = (
        f"You are an expert SQL assistant specializing in {example['domain']}.\n"
        f"Given a database schema and a natural language question, "
        f"generate the correct SQL query.\n"
        f"Respond with only the SQL query, no explanations."
    )

    user_message = (
        f"Database Schema:\n{example['sql_context']}\n"
        f"Question: {example['sql_prompt']}"
    )

    # SQL query only (target output for training)
    assistant_message = example['sql']

    formatted_text = formatter.format_for_training(
        system=system_prompt,
        user=user_message,
        assistant=assistant_message
    )

    return {"text": formatted_text}


def preprocess_sql_dataset(
    dataset: "Dataset",
    formatter: "ChatFormatter",
    complexity_filter: str | list[str] | None = None,
    max_samples: int | None = None
) -> "Dataset":
    """
    Preprocess the SQL dataset for fine-tuning.
    
    Args:
        dataset: Raw dataset from HuggingFace (gretelai/synthetic_text_to_sql)
        formatter: ChatFormatter instance for template formatting
        complexity_filter: Filter by sql_complexity 
            (e.g., 'single join', ['basic', 'single join'])
        max_samples: Limit number of samples (useful for testing)
    
    Returns:
        Formatted dataset ready for SFTTrainer with 'text' column
    
    Example:
        >>> from gpt_oss import GptOssModel, preprocess_sql_dataset
        >>> model = GptOssModel(model_config=ModelConfig())
        >>> dataset = preprocess_sql_dataset(
        ...     raw_dataset['train'],
        ...     model.formatter,
        ...     complexity_filter='single join',
        ...     max_samples=1000
        ... )
    """
    # Filter by complexity if specified
    if complexity_filter:
        if isinstance(complexity_filter, str):
            complexity_filter = [complexity_filter]
        dataset = dataset.filter(
            lambda x: x['sql_complexity'] in complexity_filter
        )
        print(f"Filtered to {len(dataset)} samples with complexity: {complexity_filter}")

    # Limit samples if specified
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} samples")

    # Apply formatting
    formatted_dataset = dataset.map(
        lambda x: format_sql_example(x, formatter),
        remove_columns=dataset.column_names,
        desc="Formatting dataset"
    )

    return formatted_dataset
