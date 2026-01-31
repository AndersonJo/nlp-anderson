import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import random
import numpy as np
from tqdm import tqdm
import json
import logging
from dataclasses import dataclass

from toolformer import ToolformerModel, ToolformerConfig, Tool, APICallEncoder
import numpy as np


@dataclass
class TrainingConfig:
    """Configuration for Toolformer training"""
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    augmentation_samples: int = 5
    min_improvement_threshold: float = 0.01
    max_api_calls_per_text: int = 3


class ToolformerDataset(Dataset):
    """Dataset for Toolformer training with self-supervised API call augmentation"""
    
    def __init__(self, texts: List[str], model: ToolformerModel, config: TrainingConfig,
                 is_training: bool = True):
        self.texts = texts
        self.model = model
        self.config = config
        self.is_training = is_training
        self.encoder = model.encoder
        self.tokenizer = model.tokenizer
        
        if is_training:
            self.augmented_texts = self._generate_augmented_dataset()
        else:
            self.augmented_texts = texts
        
        self.tokenized_data = self._tokenize_data()
    
    def _generate_augmented_dataset(self) -> List[str]:
        """Generate augmented dataset with API calls using self-supervised learning"""
        logging.info("Generating augmented dataset with API calls...")
        augmented_texts = []
        
        for text in tqdm(self.texts, desc="Augmenting texts"):
            original_augmented = [text]
            
            for _ in range(self.config.augmentation_samples):
                augmented_text = self._augment_single_text(text)
                if augmented_text != text:
                    original_augmented.append(augmented_text)
            
            augmented_texts.extend(original_augmented)
        
        logging.info(f"Generated {len(augmented_texts)} augmented examples from {len(self.texts)} original texts")
        return augmented_texts
    
    def _augment_single_text(self, text: str) -> str:
        """Augment a single text with API calls using the Toolformer methodology"""
        candidate_positions = self.model.sample_api_candidates(text, top_k=10)
        
        if not candidate_positions:
            return text
        
        tokens = self.tokenizer.encode(text)
        candidate_calls = []
        
        for pos in candidate_positions[:self.config.max_api_calls_per_text]:
            if pos >= len(tokens) - 1:
                continue
            
            context_before = self.tokenizer.decode(tokens[:pos])
            context_after = self.tokenizer.decode(tokens[pos:])
            
            for tool_name in self.model.tools.keys():
                generated_call = self._generate_api_call_for_position(
                    context_before, context_after, tool_name
                )
                
                if generated_call:
                    candidate_calls.append((pos, generated_call))
        
        if not candidate_calls:
            return text
        
        filtered_calls = self._filter_beneficial_calls(text, candidate_calls)
        
        if not filtered_calls:
            return text
        
        return self._insert_api_calls(text, filtered_calls)
    
    def _generate_api_call_for_position(self, context_before: str, context_after: str, 
                                      tool_name: str) -> Optional[str]:
        """Generate API call for a specific position and tool"""
        if tool_name == "Calculator":
            math_expressions = self._extract_math_expressions(context_before + context_after)
            if math_expressions:
                expr = random.choice(math_expressions)
                result = self.model.execute_tool(tool_name, expr)
                return self.encoder.encode_call_with_result(tool_name, expr, result)
        
        elif tool_name == "QA":
            questions = self._extract_questions(context_before + context_after)
            if questions:
                question = random.choice(questions)
                result = self.model.execute_tool(tool_name, question)
                return self.encoder.encode_call_with_result(tool_name, question, result)
        
        return None
    
    def _extract_math_expressions(self, text: str) -> List[str]:
        """Extract potential mathematical expressions from text"""
        import re
        
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        numbers = re.findall(number_pattern, text)
        
        expressions = []
        if len(numbers) >= 2:
            for i in range(min(3, len(numbers) - 1)):
                for j in range(i + 1, min(i + 3, len(numbers))):
                    expressions.extend([
                        f"{numbers[i]} + {numbers[j]}",
                        f"{numbers[i]} - {numbers[j]}",
                        f"{numbers[i]} * {numbers[j]}",
                    ])
                    if float(numbers[j]) != 0:
                        expressions.append(f"{numbers[i]} / {numbers[j]}")
        
        return expressions[:5]
    
    def _extract_questions(self, text: str) -> List[str]:
        """Extract potential questions from text"""
        import re
        
        question_pattern = r'[A-Z][^.!?]*\?'
        questions = re.findall(question_pattern, text)
        
        if not questions and len(text.split()) > 5:
            words = text.split()
            sample_phrase = ' '.join(words[:min(10, len(words))])
            questions = [f"What is {sample_phrase}?"]
        
        return questions[:3]
    
    def _filter_beneficial_calls(self, original_text: str, 
                               candidate_calls: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        """Filter API calls based on loss improvement"""
        beneficial_calls = []
        
        for pos, call in candidate_calls:
            test_text = self._insert_single_call(original_text, pos, call)
            improvement = self.model.compute_loss_improvement(original_text, test_text)
            
            if improvement > self.config.min_improvement_threshold:
                beneficial_calls.append((pos, call))
        
        return beneficial_calls
    
    def _insert_single_call(self, text: str, position: int, call: str) -> str:
        """Insert a single API call at specified position"""
        tokens = self.tokenizer.encode(text)
        if position >= len(tokens):
            return text
        
        before_tokens = tokens[:position]
        after_tokens = tokens[position:]
        
        before_text = self.tokenizer.decode(before_tokens)
        after_text = self.tokenizer.decode(after_tokens)
        
        return f"{before_text} {call} {after_text}"
    
    def _insert_api_calls(self, text: str, calls: List[Tuple[int, str]]) -> str:
        """Insert multiple API calls into text"""
        calls = sorted(calls, key=lambda x: x[0], reverse=True)
        
        current_text = text
        for pos, call in calls:
            current_text = self._insert_single_call(current_text, pos, call)
        
        return current_text
    
    def _tokenize_data(self) -> List[Dict[str, torch.Tensor]]:
        """Tokenize the augmented dataset"""
        tokenized_data = []
        
        for text in self.augmented_texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            tokenized_data.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': encoding['input_ids'].squeeze(0).clone()
            })
        
        return tokenized_data
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]


class ToolformerTrainer:
    """Trainer for Toolformer model with self-supervised learning"""
    
    def __init__(self, model: ToolformerModel, train_config: TrainingConfig):
        self.model = model
        self.config = train_config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
        self.scheduler = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(self, train_texts: List[str], val_texts: Optional[List[str]] = None):
        """Main training loop"""
        self.logger.info("Starting Toolformer training...")
        
        train_dataset = ToolformerDataset(train_texts, self.model, self.config, is_training=True)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        val_loader = None
        if val_texts:
            val_dataset = ToolformerDataset(val_texts, self.model, self.config, is_training=False)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        total_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=self.config.warmup_steps
        )
        
        global_step = 0
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
                loss = self._training_step(batch)
                epoch_loss += loss
                global_step += 1
                
                if global_step % self.config.logging_steps == 0:
                    self.logger.info(f"Step {global_step}, Loss: {loss:.4f}")
                
                if val_loader and global_step % self.config.eval_steps == 0:
                    val_loss = self._evaluate(val_loader)
                    self.logger.info(f"Validation Loss: {val_loss:.4f}")
                    self.model.train()
                
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"checkpoint-{global_step}")
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        self.logger.info("Training completed!")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.optimizer.zero_grad()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        return loss.item()
    
    def _evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs['loss'].item()
        
        return total_loss / len(val_loader)
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config
        }, f"{checkpoint_name}.pt")
        self.logger.info(f"Checkpoint saved: {checkpoint_name}.pt")


def create_sample_training_data() -> List[str]:
    """Create sample training data for testing"""
    return [
        "The population of New York City is approximately 8.4 million people.",
        "If I have 25 apples and give away 7, how many do I have left?",
        "The distance from Earth to the Moon is about 384,400 kilometers.",
        "Calculate the area of a rectangle with length 15 meters and width 8 meters.",
        "What is the capital of France?",
        "The temperature today is 22 degrees Celsius, which equals how many Fahrenheit?",
        "There are 365 days in a regular year and 366 in a leap year.",
        "If a car travels at 60 mph for 2.5 hours, what distance does it cover?",
        "The largest ocean on Earth is the Pacific Ocean.",
        "What is 15% of 200?",
    ]


if __name__ == "__main__":
    from toolformer import Calculator, QASystem
    
    config = ToolformerConfig()
    tools = [Calculator(), QASystem()]
    model = ToolformerModel(config, tools)
    
    train_config = TrainingConfig(
        batch_size=2,
        num_epochs=1,
        learning_rate=5e-5
    )
    
    trainer = ToolformerTrainer(model, train_config)
    sample_data = create_sample_training_data()
    
    trainer.train(sample_data)