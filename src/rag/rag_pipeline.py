
import os
import torch
import argparse
import config

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

def load_llm(model_type: str, model_id: str, lora_path: str = None):
    """선택된 타입에 따라 언어 모델을 로드하는 함수"""
    
    # 1. API 모델 로드
    if model_type == "api":
        print(f"Loading model via API ({config.GEMINI_API_MODEL})...")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_API_MODEL, 
            google_api_key=api_key, 
            temperature=config.RAG_TEMPERATURE,
            convert_system_message_to_human=True
        )
        return llm

    # --- 아래는 로컬 모델 공통 설정 ---
    

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
            max_seq_length=config.UNSLOTH_MAX_SEQ_LENGTH,
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
        max_new_tokens=config.RAG_MAX_NEW_TOKENS,
        model_kwargs = {"temperature": config.RAG_TEMPERATURE}
    )
    return HuggingFacePipeline(pipeline=pipe)

def setup_retriever(knowledge_base_path: str):
    """데이터셋을 로드하고 리트리버를 설정하는 함수"""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_ID)

    if knowledge_base_path == "default":
        print(f"Loading default KorQuAD dataset ({config.KNOWLEDGE_BASE_DATASET}) for RAG knowledge base...")
        dataset = load_dataset(config.KNOWLEDGE_BASE_DATASET)["train"]
        contexts = list(set(item['context'] for item in dataset))
        docs = [Document(page_content=context) for context in contexts]
        print(f"Dataset loaded. Using {len(docs)} unique contexts.")

        print("Splitting documents and creating vector store...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.TEXT_SPLITTER_CHUNK_SIZE, chunk_overlap=config.TEXT_SPLITTER_CHUNK_OVERLAP)
        splits = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        print("In-memory vector store created successfully.")
    
    else:
        if not os.path.exists(knowledge_base_path):
            raise FileNotFoundError(f"Custom vector store not found at: {knowledge_base_path}")
        print(f"Loading custom vector store from: {knowledge_base_path}")
        vectorstore = FAISS.load_local(knowledge_base_path, embeddings, allow_dangerous_deserialization=True)
        print("Custom vector store loaded successfully.")

    return vectorstore.as_retriever(search_kwargs={'k': config.RETRIEVER_SEARCH_K})

def create_rag_chain(llm, retriever):
    """RAG 체인을 생성합니다."""
    prompt_template = ChatPromptTemplate.from_template(config.RAG_PROMPT_TEMPLATE)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain

def main(model_type: str, question: str, lora_path: str = None, knowledge_base: str = "default", model_id: str = config.LOCAL_MODEL_ID):
    """메인 실행 함수"""
    try:
        # 1. 모델 로드
        llm = load_llm(model_type, model_id, lora_path)
        print("Model loaded successfully.")

        # 2. 리트리버 설정
        retriever = setup_retriever(knowledge_base)
        print("Retriever setup successfully.")

        # 3. RAG 체인 생성
        rag_chain = create_rag_chain(llm, retriever)
        print("RAG chain created successfully.")

        # 4. RAG 체인 실행
        print("\n--- RAG Inference Start ---")
        print(f"Model Type: {model_type}")
        print(f"Question: {question}")
        
        answer = rag_chain.invoke(question)
        
        print("\nAnswer:")
        print(answer)
        print("--- RAG Inference End ---")

    except Exception as e:
        print(f"An error occurred: {e}")


