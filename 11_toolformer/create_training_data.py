#!/usr/bin/env python3
"""
Create a curated training dataset for Toolformer from multiple sources.
This script downloads and combines datasets that are ideal for tool usage.
"""

import json
import random
from datasets import load_dataset
from typing import List, Dict
import re
from tqdm import tqdm

def load_gsm8k_data(max_samples: int = 2000) -> List[str]:
    """Load GSM8K math word problems"""
    print("Loading GSM8K dataset...")
    try:
        dataset = load_dataset("gsm8k", "main")
        texts = []
        
        for item in dataset['train'][:max_samples]:
            question = item['question'].strip()
            answer = item['answer'].strip()
            
            # Extract the numerical answer
            answer_match = re.search(r'#### (\d+(?:\.\d+)?)', answer)
            if answer_match:
                numerical_answer = answer_match.group(1)
                text = f"{question} The answer is {numerical_answer}."
            else:
                text = f"{question} {answer}"
            
            texts.append(text)
        
        print(f"Loaded {len(texts)} GSM8K examples")
        return texts
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        return []

def load_natural_questions_data(max_samples: int = 1000) -> List[str]:
    """Load Natural Questions dataset"""
    print("Loading Natural Questions dataset...")
    try:
        dataset = load_dataset("natural_questions", "dev")  # Using dev set as it's smaller
        texts = []
        
        for item in dataset['validation'][:max_samples]:
            question = item['question']['text'].strip()
            
            # Get short answers if available
            annotations = item['annotations']
            if annotations and len(annotations) > 0:
                short_answers = annotations[0].get('short_answers', [])
                if short_answers:
                    answer = short_answers[0]['text'].strip()
                    text = f"{question} The answer is {answer}."
                else:
                    text = f"{question} This requires further research."
            else:
                text = f"{question} This is a factual question."
            
            texts.append(text)
        
        print(f"Loaded {len(texts)} Natural Questions examples")
        return texts
    except Exception as e:
        print(f"Error loading Natural Questions: {e}")
        return []

def load_mmlu_data(max_samples: int = 1500) -> List[str]:
    """Load MMLU dataset"""
    print("Loading MMLU dataset...")
    try:
        dataset = load_dataset("cais/mmlu", "all")
        texts = []
        
        subjects_with_math = [
            'high_school_mathematics', 'elementary_mathematics', 'college_mathematics',
            'high_school_physics', 'college_physics', 'high_school_chemistry',
            'high_school_statistics', 'machine_learning'
        ]
        
        for split in ['test', 'validation']:
            if split not in dataset:
                continue
                
            for item in dataset[split][:max_samples]:
                question = item['question'].strip()
                choices = item['choices']
                answer_idx = item['answer']
                correct_answer = choices[answer_idx]
                
                # Format as a natural question-answer pair
                choices_text = ', '.join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
                text = f"{question} Options: {choices_text} The correct answer is ({chr(65+answer_idx)}) {correct_answer}."
                
                texts.append(text)
        
        print(f"Loaded {len(texts)} MMLU examples")
        return texts
    except Exception as e:
        print(f"Error loading MMLU: {e}")
        return []

def create_synthetic_tool_data(num_samples: int = 500) -> List[str]:
    """Create synthetic data that naturally requires tool usage"""
    print("Creating synthetic tool-friendly data...")
    
    templates = [
        # Calculator examples
        "If I buy {a} items at ${b} each, what's the total cost?",
        "A rectangle has length {a} meters and width {b} meters. What is its area?",
        "What is {a}% of {b}?",
        "If I start with ${a} and spend ${b}, how much do I have left?",
        "What is {a} divided by {b}?",
        "Calculate {a} to the power of {b}.",
        "Find the square root of {a}.",
        
        # QA examples
        "What is the capital of {country}?",
        "Who was the president of {country} in {year}?",
        "What is the largest {category}?",
        "When did {event} happen?",
        
        # Calendar examples
        "What day of the week is {date}?",
        "How many days are there between {date1} and {date2}?",
        "What is today's date?",
        "If today is {date}, what date will it be in {days} days?",
        
        # Translation examples
        "How do you say '{phrase}' in {language}?",
        "Translate '{phrase}' to {language}.",
        
        # Mixed examples
        "The temperature today is {temp}°C. Convert this to Fahrenheit.",
        "If a train travels at {speed} mph for {time} hours, how far does it go?",
    ]
    
    # Sample data for templates
    countries = ['France', 'Germany', 'Italy', 'Spain', 'Japan', 'China', 'Brazil']
    languages = ['Spanish', 'French', 'German', 'Italian']
    phrases = ['hello', 'thank you', 'goodbye', 'good morning']
    categories = ['ocean', 'country', 'city', 'mountain', 'lake']
    events = ['World War II', 'the Renaissance', 'the Industrial Revolution']
    
    texts = []
    
    for _ in range(num_samples):
        template = random.choice(templates)
        
        try:
            if '{a}' in template or '{b}' in template:
                a, b = random.randint(1, 100), random.randint(1, 100)
                template = template.format(a=a, b=b)
            elif '{country}' in template:
                country = random.choice(countries)
                template = template.format(country=country)
                if '{year}' in template:
                    year = random.randint(1950, 2020)
                    template = template.format(year=year)
            elif '{category}' in template:
                category = random.choice(categories)
                template = template.format(category=category)
            elif '{date}' in template:
                date = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
                template = template.format(date=date)
                if '{date1}' in template or '{date2}' in template:
                    date1 = f"2024-{random.randint(1,6):02d}-{random.randint(1,28):02d}"
                    date2 = f"2024-{random.randint(7,12):02d}-{random.randint(1,28):02d}"
                    template = template.format(date1=date1, date2=date2)
                if '{days}' in template:
                    days = random.randint(1, 30)
                    template = template.format(days=days)
            elif '{phrase}' in template:
                phrase = random.choice(phrases)
                language = random.choice(languages)
                template = template.format(phrase=phrase, language=language)
            elif '{temp}' in template:
                temp = random.randint(0, 40)
                template = template.format(temp=temp)
            elif '{speed}' in template:
                speed = random.randint(30, 120)
                time = random.randint(1, 8)
                template = template.format(speed=speed, time=time)
            
            texts.append(template)
            
        except Exception:
            # Skip malformed templates
            continue
    
    print(f"Created {len(texts)} synthetic examples")
    return texts

def create_wikipedia_samples(max_samples: int = 1000) -> List[str]:
    """Create samples from Wikipedia that are likely to need tools"""
    print("Loading Wikipedia samples...")
    try:
        # Use a smaller Wikipedia subset
        dataset = load_dataset("wikipedia", "20220301.simple")  # Simple English Wikipedia
        texts = []
        
        # Look for articles that contain numbers, dates, or math
        math_keywords = ['calculate', 'total', 'population', 'area', 'distance', 'temperature', 'speed']
        fact_keywords = ['capital', 'president', 'founded', 'born', 'died', 'located']
        
        for item in dataset['train'][:max_samples*3]:  # Sample more to filter
            text = item['text']
            title = item['title']
            
            # Skip very short articles
            if len(text) < 200:
                continue
            
            # Look for articles with potential tool usage
            text_lower = text.lower()
            if any(keyword in text_lower for keyword in math_keywords + fact_keywords):
                # Extract first few sentences
                sentences = text.split('. ')[:3]
                summary = '. '.join(sentences) + '.'
                texts.append(f"{title}: {summary}")
                
                if len(texts) >= max_samples:
                    break
        
        print(f"Loaded {len(texts)} Wikipedia samples")
        return texts
    except Exception as e:
        print(f"Error loading Wikipedia: {e}")
        return []

def save_training_data(texts: List[str], output_file: str):
    """Save training data to file"""
    print(f"Saving {len(texts)} examples to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text.strip() + '\n')
    
    print(f"Training data saved to {output_file}")

def create_curated_dataset():
    """Create a curated dataset combining multiple sources"""
    print("Creating curated Toolformer training dataset...")
    print("=" * 50)
    
    all_texts = []
    
    # Load different data sources
    gsm8k_texts = load_gsm8k_data(max_samples=1000)
    all_texts.extend(gsm8k_texts)
    
    nq_texts = load_natural_questions_data(max_samples=500)
    all_texts.extend(nq_texts)
    
    mmlu_texts = load_mmlu_data(max_samples=800)
    all_texts.extend(mmlu_texts)
    
    synthetic_texts = create_synthetic_tool_data(num_samples=700)
    all_texts.extend(synthetic_texts)
    
    wiki_texts = create_wikipedia_samples(max_samples=500)
    all_texts.extend(wiki_texts)
    
    # Shuffle the combined dataset
    random.shuffle(all_texts)
    
    print(f"\nTotal dataset size: {len(all_texts)} examples")
    
    # Split into train/validation
    split_idx = int(0.9 * len(all_texts))
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    # Save datasets
    save_training_data(train_texts, "toolformer_train.txt")
    save_training_data(val_texts, "toolformer_val.txt")
    
    # Also save as JSON for easier loading
    train_data = {"texts": train_texts}
    val_data = {"texts": val_texts}
    
    with open("toolformer_train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open("toolformer_val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Dataset creation complete!")
    print(f"📊 Training set: {len(train_texts)} examples")
    print(f"📊 Validation set: {len(val_texts)} examples")
    print("\nFiles created:")
    print("- toolformer_train.txt")
    print("- toolformer_train.json")
    print("- toolformer_val.txt") 
    print("- toolformer_val.json")
    
    # Show sample examples
    print(f"\n📝 Sample examples:")
    for i, example in enumerate(train_texts[:3]):
        print(f"{i+1}. {example[:100]}...")

if __name__ == "__main__":
    create_curated_dataset()