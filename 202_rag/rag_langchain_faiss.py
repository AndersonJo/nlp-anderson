#!/usr/bin/env python3
"""
Kepler 442c Sci-Fi RAG Example - LangChain + FAISS
Korean sci-fi world knowledge base with RAG
"""

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import os
import math

# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class LangChainKoreanRAG:
    def __init__(self, model_path: str = "openai/gpt-oss-20b", csv_path: str = "kepler_442c_data.csv",
                 similarity_threshold: float = 0.3, max_docs: int = 3, distance_scale: float = 100.0):
        """Initialize LangChain-based RAG system

        Args:
            similarity_threshold: Minimum probability score to consider a document relevant (0-1, higher = more similar)
            max_docs: Maximum number of documents to retrieve
            distance_scale: Scale factor for exponential decay conversion (lower = more sensitive)
        """

        print("🚀 Kepler 442c Sci-Fi RAG with LangChain + FAISS")
        print("=" * 50)

        # Store parameters
        self.similarity_threshold = similarity_threshold
        self.max_docs = max_docs
        self.distance_scale = distance_scale

        print(f"📊 Probability threshold: {similarity_threshold}")
        print(f"📄 Max documents: {max_docs}")
        print(f"🔧 Distance scale: {distance_scale}")

        # Clear GPU cache
        torch.cuda.empty_cache()

        # Load documents
        print("📄 Loading documents...")
        self.documents = self.load_documents(csv_path)

        # Initialize embeddings
        print("🔢 Setting up embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sroberta-multitask',
            model_kwargs={'device': 'cpu'}
        )

        # Create FAISS vector store
        print("🗃️ Creating FAISS vector store...")
        self.vectorstore = self.create_faiss_store()

        # Initialize LLM
        print("🤖 Setting up LLM...")
        self.llm = self.setup_llm(model_path)

        # Create QA chain
        print("⛓️ Building QA chain...")
        self.qa_chain = self.create_qa_chain()

        print("✅ Ready!")

    def load_documents(self, csv_path: str) -> list[Document]:
        """Load CSV data and convert to LangChain Documents"""
        df = pd.read_csv(csv_path)

        documents = []
        for _, row in df.iterrows():
            content = f"제목: {row['title']}\n\n내용: {row['content']}"

            doc = Document(
                page_content=content,
                metadata={
                    "id": str(row['id']),
                    "title": row['title'],
                    "source": "korean_fermentation_data"
                }
            )
            documents.append(doc)

        print(f"Loaded {len(documents)} documents")
        return documents

    def create_faiss_store(self) -> FAISS:
        """Create FAISS vector store with documents"""

        # Text splitter for chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )

        split_docs = text_splitter.split_documents(self.documents)
        print(f"Split into {len(split_docs)} chunks")

        # Create FAISS vector store
        vectorstore = FAISS.from_documents(
            documents=split_docs,
            embedding=self.embeddings
        )

        print(f"FAISS index created with {vectorstore.index.ntotal} vectors")
        return vectorstore

    def setup_llm(self, model_path: str) -> HuggingFacePipeline:
        """Setup local LLM with LangChain wrapper"""

        try:
            # 8-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir="/home/anderson/.cache/huggingface/hub"
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                cache_dir="/home/anderson/.cache/huggingface/hub",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )

            # Create pipeline
            text_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id
            )

            # LangChain wrapper
            llm = HuggingFacePipeline(pipeline=text_pipeline)
            print("LLM loaded with 8-bit quantization")
            return llm

        except Exception as e:
            print(f"LLM loading failed: {e}")
            print("Using retrieval-only mode")
            return None

    def create_qa_chain(self) -> RetrievalQA:
        """Create LangChain RetrievalQA chain"""

        # Korean prompt template
        template = """다음 문서들을 참고하여 질문에 정확하게 답변해주세요.

참고 문서:
{context}

질문: {question}

답변:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Create FAISS retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        if self.llm is None:
            return retriever

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        return qa_chain

    def distance_to_probability(self, distance: float) -> float:
        """Convert FAISS distance to probability using exponential decay

        Args:
            distance: FAISS distance score (lower = more similar)

        Returns:
            probability: Score from 0 to 1 (higher = more similar)
        """
        return math.exp(-distance / self.distance_scale)

    def query(self, question: str) -> dict:
        """Query the RAG system with probability-based filtering"""

        # Get more documents than needed, then filter by threshold
        all_docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=10)

        # Convert distances to probabilities
        docs_with_probs = []
        for doc, distance in all_docs_with_scores:
            probability = self.distance_to_probability(distance)
            docs_with_probs.append((doc, distance, probability))

        # Filter by probability threshold
        filtered_docs = [(doc, distance, prob) for doc, distance, prob in docs_with_probs
                        if prob >= self.similarity_threshold]

        # Limit to max_docs
        docs_with_scores = filtered_docs[:self.max_docs]

        print(f"📊 Found {len(all_docs_with_scores)} total docs, {len(filtered_docs)} above probability threshold ({self.similarity_threshold:.2f}), using top {len(docs_with_scores)}")

        if self.llm is None:
            # Retrieval only
            if docs_with_scores:
                best_doc, best_distance, best_prob = docs_with_scores[0]
                title = best_doc.metadata.get('title', '제목 없음')
                content = best_doc.page_content.split('내용: ')[1] if '내용: ' in best_doc.page_content else best_doc.page_content
                answer = f"[{title}] {content}"
            else:
                answer = "관련 정보를 찾을 수 없습니다. (확률 임계값보다 낮은 문서들만 있음)"

            return {
                "question": question,
                "answer": answer,
                "source_documents": [doc for doc, distance, prob in docs_with_scores],
                "distance_scores": [float(distance) for doc, distance, prob in docs_with_scores],
                "probability_scores": [float(prob) for doc, distance, prob in docs_with_scores],
                "threshold_used": self.similarity_threshold,
                "total_candidates": len(all_docs_with_scores),
                "filtered_count": len(filtered_docs)
            }

        # Full RAG with LLM
        try:
            # Use filtered documents for context
            if docs_with_scores:
                result = self.qa_chain.invoke({"query": question})
                answer = result["result"]
            else:
                answer = "확률 임계값을 만족하는 관련 문서를 찾을 수 없습니다."

            return {
                "question": question,
                "answer": answer,
                "source_documents": [doc for doc, distance, prob in docs_with_scores],
                "distance_scores": [float(distance) for doc, distance, prob in docs_with_scores],
                "probability_scores": [float(prob) for doc, distance, prob in docs_with_scores],
                "threshold_used": self.similarity_threshold,
                "total_candidates": len(all_docs_with_scores),
                "filtered_count": len(filtered_docs)
            }

        except Exception as e:
            print(f"Generation error: {e}")
            # Fallback to retrieval with scores
            if docs_with_scores:
                answer = f"검색 결과: {docs_with_scores[0][0].page_content}"
            else:
                answer = "확률 임계값을 만족하는 관련 정보가 없습니다."

            return {
                "question": question,
                "answer": answer,
                "source_documents": [doc for doc, distance, prob in docs_with_scores],
                "distance_scores": [float(distance) for doc, distance, prob in docs_with_scores],
                "probability_scores": [float(prob) for doc, distance, prob in docs_with_scores],
                "threshold_used": self.similarity_threshold,
                "total_candidates": len(all_docs_with_scores),
                "filtered_count": len(filtered_docs)
            }

def main():
    """Demo LangChain + FAISS RAG with probability-based filtering"""

    # Initialize RAG with probability threshold
    rag = LangChainKoreanRAG(similarity_threshold=0.3, distance_scale=100.0)  # 30% confidence minimum

    # Test questions - mix of relevant and irrelevant
    questions = [
        "미냥쿵 442c의 중력은 지구와 어떻게 다른가요?",  # Should find good match
        "젤라티안은 어떤 생명체인가요?",                  # Should find good match
        "지구의 날씨는 어떤가요?",                     # Should find poor/no matches
        "크로노스 결정의 특성은 무엇인가요?",            # Should find good match
        "피자 만드는 방법은?",                        # Should find no relevant matches
    ]

    print("\n🚀 Probability-Based RAG Demo")
    print("=" * 42)

    for question in questions:
        print(f"\n질문: {question}")
        result = rag.query(question)
        print(f"답변: {result['answer']}")

        print(f"📊 필터링 결과: {result['filtered_count']}/{result['total_candidates']} 문서가 확률 임계값({result['threshold_used']:.1%}) 이상")

        if result['source_documents']:
            print("참고 문서 (거리 → 확률):")
            for i, (doc, distance, prob) in enumerate(zip(result['source_documents'],
                                                          result['distance_scores'],
                                                          result['probability_scores']), 1):
                title = doc.metadata.get('title', '제목 없음')
                confidence = "🔥 HIGH" if prob > 0.5 else "✅ GOOD" if prob > 0.3 else "⚠️ OK" if prob > 0.1 else "❌ LOW"
                print(f"  {i}. {title}")
                print(f"     거리: {distance:.1f} → 확률: {prob:.1%} {confidence}")
        else:
            print("❌ 확률 임계값을 만족하는 문서가 없습니다.")
        print()

def test_different_thresholds():
    """Test with different similarity thresholds"""
    print("\n🔬 Testing Different Thresholds")
    print("=" * 35)

    question = "지구의 날씨는 어떤가요?"  # Irrelevant question
    thresholds = [200.0, 150.0, 120.0, 100.0, 80.0]

    for threshold in thresholds:
        print(f"\n임계값: {threshold}")
        rag = LangChainKoreanRAG(similarity_threshold=threshold)
        result = rag.query(question)
        print(f"검색된 문서 수: {len(result['source_documents'])}")
        if result['source_documents']:
            best_score = result['similarity_scores'][0]
            print(f"최고 점수: {best_score:.2f}")

if __name__ == "__main__":
    main()
    # Uncomment to test different thresholds
    # test_different_thresholds()