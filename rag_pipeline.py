
import os
import torch
import argparse

# --- 모델 로딩 라이브러리 ---
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from unsloth import FastLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI
from peft import PeftModel # LoRA 모델 로딩을 위해 추가

# --- LangChain 및 RAG 관련 라이브러리 ---
from langchain_huggingface import HuggingFacePipeline
from datasets import load_dataset
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 토크나이저 병렬 처리 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_llm(model_type: str, lora_path: str = None):
    """선택된 타입에 따라 언어 모델을 로드하는 함수"""
    
    # 1. API 모델 로드
    if model_type == "api":
        print("Loading model via API (gemini-1.5-flash-latest)...")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", 
            google_api_key=api_key, 
            temperature=0.1,
            convert_system_message_to_human=True
        )
        return llm

    # --- 아래는 로컬 모델 공통 설정 ---
    model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

    # 2. 비양자화 로컬 모델 로드 (순수 Transformers)
    if model_type == "local":
        print(f"Loading NON-QUANTIZED local model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 3. 양자화 로컬 모델 로드 (Unsloth)
    elif model_type == "local-quantized":
        print(f"Loading QUANTIZED local model with Unsloth: {model_id}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
    
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    # 로컬 모델 로드 후 LoRA 어댑터 병합 (선택 사항)
    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload() # 추론을 위해 LoRA 가중치 병합
        print("LoRA adapter loaded and merged.")

    # 로컬 모델들을 위한 파이프라인 생성
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        model_kwargs = {"temperature": 0.1}
    )
    return HuggingFacePipeline(pipeline=pipe)

def setup_retriever():
    """KorQuAD 데이터셋을 로드하고 리트리버를 설정하는 함수"""
    print("Loading KorQuAD dataset for RAG knowledge base...")
    dataset = load_dataset("squad_kor_v1")["train"]
    contexts = list(set(item['context'] for item in dataset))
    docs = [Document(page_content=context) for context in contexts]
    print(f"Dataset loaded. Using {len(docs)} unique contexts.")

    print("Splitting documents and creating vector store...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    print("Vector store created successfully.")
    
    return vectorstore.as_retriever(search_kwargs={'k': 3})

def create_rag_chain(llm, retriever):
    """RAG 체인을 생성합니다."""
    prompt_template = ChatPromptTemplate.from_template(
        """
        다음 컨텍스트 정보를 사용하여 질문에 답변해 주세요. 
        만약 컨텍스트에 답변이 없다면, 모른다고 답해 주세요.

        컨텍스트: 
        {context}

        질문: 
        {question}

        답변:
        """
    )
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Unified RAG Inference Script.")
    parser.add_argument(
        "model_type", 
        type=str, 
        choices=["api", "local", "local-quantized"],
        help="Type of model to use for inference."
    )
    parser.add_argument("question", type=str, help="The question to ask the RAG system.")
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to the LoRA adapter (optional). Only applicable for local models."
    )
    args = parser.parse_args()

    try:
        # 1. 모델 로드
        llm = load_llm(args.model_type, args.lora_path)
        print("Model loaded successfully.")

        # 2. 리트리버 설정
        retriever = setup_retriever()
        print("Retriever setup successfully.")

        # 3. RAG 체인 생성
        rag_chain = create_rag_chain(llm, retriever)
        print("RAG chain created successfully.")

        # 4. RAG 체인 실행
        print("\n--- RAG Inference Start ---")
        print(f"Model Type: {args.model_type}")
        print(f"Question: {args.question}")
        
        answer = rag_chain.invoke(args.question)
        
        print("\nAnswer:")
        print(answer)
        print("--- RAG Inference End ---")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
