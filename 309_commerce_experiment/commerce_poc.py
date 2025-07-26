#!/usr/bin/env python3
"""
Commerce POC using Kimi LLM for product comparison  
Hook test - fifth attempt with ../.claude location
"""

import csv
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


@dataclass
class Product:
    id: str
    name: str
    category: str
    price: int
    description: str
    features: str
    pros: str
    cons: str


class ProductSearcher:
    def __init__(self, csv_path: str = "products.csv"):
        self.products = self.load_products(csv_path)

    def load_products(self, csv_path: str) -> List[Product]:
        """Load products from CSV file"""
        products = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    product = Product(
                        id=row['id'],
                        name=row['name'],
                        category=row['category'],
                        price=int(row['price']),
                        description=row['description'],
                        features=row['features'],
                        pros=row['pros'],
                        cons=row['cons']
                    )
                    products.append(product)
        except FileNotFoundError:
            print(f"Error: {csv_path} 파일을 찾을 수 없습니다.")
        except Exception as e:
            print(f"Error loading products: {e}")
        return products

    def search_products(self, query: str, max_results: int = 10) -> List[Product]:
        """Search products based on query text"""
        query_lower = query.lower()
        scored_products = []

        for product in self.products:
            score = 0
            search_text = f"{product.name} {product.description} {product.features} {product.category}".lower()

            # Simple keyword matching
            for word in query_lower.split():
                if word in search_text:
                    score += 1

            if score > 0:
                scored_products.append((score, product))

        # Sort by score and return top results
        scored_products.sort(key=lambda x: x[0], reverse=True)
        return [product for _, product in scored_products[:max_results]]


class LocalLLMClient:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        - production: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
        - development: meta-llama/Llama-3.1-8B-Instruct

        :param model_name:
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None

        self._load_model()

    def _load_model(self):
        """Load local DeepSeek-R1-Distill-Llama-70B model from Hugging Face"""

        print(f"Loading DeepSeek-R1-Distill-Llama-70B model: {self.model_name}")
        print("Warning: This model is large. Approximately 70B parameters.")

        # Check available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"Available GPU memory: {gpu_memory:.1f}GB")
            self.device = "cuda"
        else:
            print("GPU not available. Running in CPU mode (very slow).")
            self.device = "cpu"

        # Configure quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            max_workers=2,  # Limit concurrent downloads
            resume_download=True  # Resume interrupted downloads
        )

        # Load model with quantization
        print("Loading model... (this may take a while)")

        # Create custom device map for CPU offloading
        device_map = {
            "model.embed_tokens": 0,
            "model.norm": 0,
            "lm_head": 0,
        }

        # Distribute transformer layers across available devices
        if torch.cuda.is_available():
            num_layers = 80  # DeepSeek-R1-Distill-Llama-70B has 80 layers
            layers_per_gpu = num_layers // torch.cuda.device_count()

            for i in range(num_layers):
                gpu_id = min(i // layers_per_gpu, torch.cuda.device_count() - 1)
                device_map[f"model.layers.{i}"] = gpu_id
        else:
            device_map = "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_workers=2,  # Limit concurrent downloads
            resume_download=True  # Resume interrupted downloads
        )

        print("✅ DeepSeek-R1-Distill-Llama-70B model loaded successfully!")

    def generate_response(self, prompt: str) -> str:
        """Generate response using local LLM"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Cannot generate response.")
        return self._generate_local(prompt)

    def _generate_local(self, prompt: str) -> str:
        """Generate response using local DeepSeek-R1-Distill-Llama-70B model"""
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def compare_products(self, products: List[Product], user_query: str) -> str:
        """Use local DeepSeek LLM to compare products and provide recommendations"""

        # Prepare product information for LLM
        products_info = []
        for product in products:
            info = f"""
상품명: {product.name}
카테고리: {product.category}
가격: {product.price:,}원
설명: {product.description}
주요 기능: {product.features}
장점: {product.pros}
단점: {product.cons}
"""
            products_info.append(info)

        prompt = f"""
사용자 검색어: "{user_query}"

다음 상품들을 비교분석하여 고객에게 최적의 추천을 제공해주세요:

{chr(10).join(products_info)}

다음 형식으로 응답해주세요:
1. 고객 의도 분석
2. 상품별 장단점 상세 비교
3. 추천 순위 (1-3위)
4. 구매 결정에 도움이 되는 조언

한국어로 친근하고 전문적으로 답변해주세요.
"""

        return self.generate_response(prompt)


class CommercePOC:
    def __init__(self, model_name: str = None):
        self.searcher = ProductSearcher()
        self.llm_client = LocalLLMClient(model_name=model_name)

    def run(self):
        """Run the terminal application"""
        print("🛒 Commerce Product Comparison POC (DeepSeek LLM)")
        print("=" * 50)
        print(f"Total {len(self.searcher.products)} products loaded.")
        print("Enter search term (quit: 'quit' or 'exit')")
        print()

        while True:
            try:
                query = input("🔍 Search: ").strip()

                if query.lower() in ['quit', 'exit', '종료']:
                    print("Exiting program.")
                    break

                if not query:
                    print("Please enter a search term.")
                    continue

                # Search products
                print(f"\nSearching for '{query}'...")
                found_products = self.searcher.search_products(query)

                if not found_products:
                    print("No search results found. Try a different search term.")
                    continue

                print(f"\nFound {len(found_products)} products.")
                print("\nAnalyzing products with DeepSeek LLM...")
                print("-" * 50)

                # Get LLM comparison
                try:
                    comparison = self.llm_client.compare_products(found_products, query)
                    print(comparison)
                    print("\n" + "=" * 50 + "\n")
                except Exception as e:
                    print(f"❌ Failed to generate comparison: {e}")
                    print("Please ensure dependencies are installed and model is working.")
                    continue

            except KeyboardInterrupt:
                print("\n\nExiting program.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    import sys

    # Allow specifying model name as command line argument
    model_name = None
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f"Model configured: {model_name}")

    poc = CommercePOC(model_name=model_name)
    poc.run()
