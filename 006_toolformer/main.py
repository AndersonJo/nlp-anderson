#!/usr/bin/env python3
"""
Main script for Toolformer training and inference.
Implements the complete Toolformer methodology from the paper.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
import json
import os
from typing import List, Dict, Optional
import random
import numpy as np

from toolformer import ToolformerModel, ToolformerConfig
from tools import get_all_tools, ToolRegistry
from training import ToolformerTrainer, TrainingConfig, ToolformerDataset
from loss import ToolformerLoss, ToolformerMetrics


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('toolformer.log'),
            logging.StreamHandler()
        ]
    )


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_data(data_path: str) -> List[str]:
    """Load training data from file"""
    if not os.path.exists(data_path):
        logging.warning(f"Training data file not found: {data_path}")
        return create_sample_training_data()
    
    with open(data_path, 'r', encoding='utf-8') as f:
        if data_path.endswith('.json'):
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'texts' in data:
                return data['texts']
        else:
            return [line.strip() for line in f if line.strip()]
    
    return []


def create_sample_training_data() -> List[str]:
    """Create sample training data for testing and demonstration"""
    return [
        "The population of New York City is approximately 8.4 million people, making it the largest city in the United States.",
        "If I have 25 apples and give away 7 to my friends, how many apples do I have left in total?",
        "The distance from Earth to the Moon is about 384,400 kilometers, which is roughly 238,855 miles.",
        "To calculate the area of a rectangle, multiply the length by the width. For example, a rectangle with length 15 meters and width 8 meters.",
        "What is the capital of France? This is one of the most basic geography questions.",
        "The temperature today is 22 degrees Celsius. I wonder what that would be in Fahrenheit.",
        "There are 365 days in a regular year and 366 in a leap year. February has 28 or 29 days.",
        "If a car travels at 60 mph for 2.5 hours, what distance does it cover during this journey?",
        "The largest ocean on Earth is the Pacific Ocean, covering about one-third of the planet's surface.",
        "What is 15% of 200? This type of percentage calculation is common in everyday math.",
        "Albert Einstein was born in 1879 and developed the theory of relativity. What year did he die?",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        "Today is Monday, January 15th, 2024. What day of the week will it be in exactly 10 days?",
        "How do you say 'hello' in Spanish? Learning basic greetings is important for communication.",
        "The boiling point of water at sea level is 100 degrees Celsius or 212 degrees Fahrenheit.",
        "If I invest $1000 at 5% annual interest, how much will I have after 3 years with compound interest?",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
        "The square root of 144 is 12, and 12 multiplied by itself gives us 144.",
        "What is the difference between 2023 and 1969? This calculation might be useful for age calculations.",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy for growth.",
        "If a pizza is cut into 8 equal slices and I eat 3 slices, what fraction of the pizza remains?",
        "The capital of Germany is Berlin, which became the capital after reunification in 1990.",
        "What is 7 factorial? Factorial calculations are important in combinatorics and probability.",
        "The human heart has 4 chambers: two atria and two ventricles that pump blood throughout the body.",
        "If it takes 3 hours to drive 180 miles, what is the average speed during this trip?",
        "William Shakespeare wrote many famous plays including Hamlet, Romeo and Juliet, and Macbeth.",
        "What is the area of a circle with radius 5 meters? Use the formula π × r².",
        "The freezing point of water is 0 degrees Celsius, which equals 32 degrees Fahrenheit.",
        "If I work 8 hours per day for 5 days a week, how many hours do I work in a month with 4 weeks?",
        "The Great Wall of China is approximately 21,196 kilometers long, making it one of the longest structures ever built."
    ]


class ToolformerTrainingPipeline:
    """Complete training pipeline for Toolformer"""
    
    def __init__(self, config: ToolformerConfig, training_config: TrainingConfig):
        self.config = config
        self.training_config = training_config
        self.tool_registry = ToolRegistry()
        self.logger = logging.getLogger(__name__)
        
        self.model = ToolformerModel(config, get_all_tools())
        self.loss_fn = ToolformerLoss(self.model)
        self.trainer = ToolformerTrainer(self.model, training_config)
        
        self.logger.info(f"Initialized Toolformer with {len(self.tool_registry.tools)} tools")
        for tool_name in self.tool_registry.list_tools():
            self.logger.info(f"  - {tool_name}")
    
    def train(self, train_data: List[str], val_data: Optional[List[str]] = None):
        """Run the complete training pipeline"""
        self.logger.info("Starting Toolformer training pipeline...")
        
        if val_data is None and len(train_data) > 10:
            split_idx = int(0.9 * len(train_data))
            val_data = train_data[split_idx:]
            train_data = train_data[:split_idx]
        
        self.trainer.train(train_data, val_data)
        self.logger.info("Training completed!")
    
    def evaluate(self, test_data: List[str]) -> Dict[str, float]:
        """Evaluate the model on test data"""
        self.logger.info("Evaluating model...")
        
        test_dataset = ToolformerDataset(test_data, self.model, self.training_config, is_training=False)
        test_loader = DataLoader(test_dataset, batch_size=self.training_config.batch_size, shuffle=False)
        
        self.model.eval()
        metrics = ToolformerMetrics()
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                
                loss_dict = self.loss_fn(input_ids, attention_mask, labels)
                metrics.update(loss_dict)
        
        results = metrics.compute_averages()
        self.logger.info("Evaluation results:")
        for key, value in results.items():
            self.logger.info(f"  {key}: {value:.4f}")
        
        return results
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_config': self.training_config,
            'tool_names': self.tool_registry.list_tools()
        }, save_path)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a trained model"""
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        
        self.logger.info(f"Model loaded from {load_path}")
    
    def generate_text(self, prompt: str, max_length: int = 200) -> str:
        """Generate text with tool usage"""
        self.model.eval()
        
        with torch.no_grad():
            result = self.model.generate_with_tools(prompt, max_length)
        
        return result
    
    def interactive_session(self):
        """Run an interactive session for testing"""
        self.logger.info("Starting interactive session. Type 'quit' to exit.")
        
        while True:
            try:
                prompt = input("\nEnter prompt: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                result = self.generate_text(prompt)
                print(f"\nGenerated text:\n{result}")
                
            except KeyboardInterrupt:
                break
        
        self.logger.info("Interactive session ended.")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Toolformer Training and Inference")
    parser.add_argument("--mode", choices=["train", "evaluate", "generate", "interactive"], 
                       default="train", help="Mode to run")
    parser.add_argument("--data-path", type=str, default="", 
                       help="Path to training data")
    parser.add_argument("--model-path", type=str, default="checkpoints/toolformer.pt",
                       help="Path to save/load model")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    set_seed(args.seed)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Toolformer in {args.mode} mode")
    
    config = ToolformerConfig()
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    pipeline = ToolformerTrainingPipeline(config, training_config)
    
    if args.mode == "train":
        train_data = load_training_data(args.data_path) if args.data_path else create_sample_training_data()
        logger.info(f"Loaded {len(train_data)} training examples")
        
        pipeline.train(train_data)
        pipeline.save_model(args.model_path)
        
    elif args.mode == "evaluate":
        if os.path.exists(args.model_path):
            pipeline.load_model(args.model_path)
        
        test_data = load_training_data(args.data_path) if args.data_path else create_sample_training_data()[:5]
        results = pipeline.evaluate(test_data)
        
        print("\nEvaluation Results:")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
    
    elif args.mode == "generate":
        if os.path.exists(args.model_path):
            pipeline.load_model(args.model_path)
        
        prompt = input("Enter prompt: ")
        result = pipeline.generate_text(prompt)
        print(f"\nGenerated text:\n{result}")
    
    elif args.mode == "interactive":
        if os.path.exists(args.model_path):
            pipeline.load_model(args.model_path)
        
        pipeline.interactive_session()


if __name__ == "__main__":
    main()