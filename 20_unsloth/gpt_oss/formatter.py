"""
Chat message formatting for training and inference.

Design Pattern:
- Single Responsibility: Only handles prompt formatting
- Decoupled from model: Can be tested without GPU
- Follows HuggingFace chat template conventions

Usage:
    formatter = ChatFormatter(tokenizer)
    
    # Training (includes target response)
    text = formatter.format_for_training(
        system="You are an SQL expert...",
        user="Schema: ... Question: ...",
        assistant="SELECT * FROM users"
    )
    
    # Inference (model generates response)
    text = formatter.format_for_inference(
        system="You are an SQL expert...",
        user="Schema: ... Question: ..."
    )
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class ChatFormatter:
    """
    Formats messages into chat template for model consumption.
    
    HuggingFace Convention:
    - Training: Include assistant message (the target output model should learn)
    - Inference: Omit assistant message + add generation prompt
    """
    
    def __init__(self, tokenizer: "PreTrainedTokenizer"):
        """
        Args:
            tokenizer: HuggingFace tokenizer with chat template support
        """
        self.tokenizer = tokenizer
    
    def format_for_training(
        self,
        system: str,
        user: str,
        assistant: str
    ) -> str:
        """
        Format messages for training (includes target response).
        
        Args:
            system: System instruction defining model behavior
            user: User's input/question
            assistant: Expected response (the target output)
        
        Returns:
            Formatted text ready for tokenization
            
        Example (Text-to-SQL):
            >>> formatter.format_for_training(
            ...     system="You are an SQL expert...",
            ...     user="Schema: CREATE TABLE users... Question: Get all users",
            ...     assistant="SELECT * FROM users"
            ... )
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
    def format_for_inference(
        self,
        system: str,
        user: str
    ) -> str:
        """
        Format messages for inference (model generates response).
        
        The generation prompt (e.g., "<|assistant|>\n") is automatically
        appended so the model knows to continue from there.
        
        Args:
            system: System instruction defining model behavior
            user: User's input/question
        
        Returns:
            Formatted text with generation prompt
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def format_messages(
        self,
        system: str,
        user: str,
        assistant: str | None = None
    ) -> str:
        """
        Unified formatting method (convenience wrapper).
        
        Automatically chooses training vs inference format based on
        whether assistant message is provided.
        
        Args:
            system: System instruction
            user: User input
            assistant: Target response (training) or None (inference)
        
        Returns:
            Formatted text
        """
        if assistant is not None:
            return self.format_for_training(system, user, assistant)
        return self.format_for_inference(system, user)
