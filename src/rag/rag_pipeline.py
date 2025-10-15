import os
import torch
import typing as tp

# --- Unsloth should be imported first ---
from unsloth import FastLanguageModel

# --- 모델 로딩 라이브러리 ---
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from peft import PeftModel  # LoRA 모델 로딩을 위해 추가

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


def load_llm(
    model_type: str,
    model_id: str,
    unsupervised_lora_path: tp.Optional[str],
    sft_lora_path: tp.Optional[str],
    max_new_tokens: int,
    temperature: float,
    model_max_seq_length: int,
):
    """선택된 타입에 따라 언어 모델을 로드하는 함수"""

    # 1. API 모델 로드
    if model_type == "api":
        print(f"Loading model via API ({model_id})...")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True,
        )
        return llm

    # --- 아래는 로컬 모델 공통 설정 ---

    # 2. 비양자화 로컬 모델 로드 (순수 Transformers)
    if model_type == "local":
        print(f"Loading NON-QUANTIZED local model: {model_id}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            print(f"Error loading local model {model_id}: {e}")
            print("This might be due to insufficient VRAM. Try a smaller model.")
            return None, None  # Return None for model and tokenizer

    # 3. 양자화 로컬 모델 로드 (Unsloth)
    elif model_type == "local-quantized":
        print(f"Loading QUANTIZED local model with Unsloth: {model_id}")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=model_max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(
                f"CUDA Out of Memory Error while loading Unsloth model {model_id}: {e}"
            )
            print(
                "This model is too large for your GPU. Try a smaller model or free up VRAM."
            )
            return None, None  # Return None for model and tokenizer
        except Exception as e:
            print(f"Error loading quantized model {model_id} with Unsloth: {e}")
            return None, None  # Return None for model and tokenizer

    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    # 로컬 모델 로드 후 LoRA 어댑터 순차적 병합 (선택 사항)
    if unsupervised_lora_path and model is not None:
        print(
            f"Loading and merging unsupervised LoRA adapter from {unsupervised_lora_path}..."
        )
        try:
            model = PeftModel.from_pretrained(model, unsupervised_lora_path)
            model = model.merge_and_unload()
            print("Unsupervised LoRA adapter loaded and merged.")
        except Exception as e:
            print(
                f"Error loading unsupervised LoRA adapter from {unsupervised_lora_path}: {e}"
            )
            return None

    if sft_lora_path and model is not None:
        print(f"Loading and merging SFT LoRA adapter from {sft_lora_path}...")
        try:
            model = PeftModel.from_pretrained(model, sft_lora_path)
            model = model.merge_and_unload()
            print("SFT LoRA adapter loaded and merged.")
        except Exception as e:
            print(f"Error loading SFT LoRA adapter from {sft_lora_path}: {e}")
            return None

    # 로컬 모델들을 위한 파이프라인 생성
    if model is None or tokenizer is None:
        return None  # Model or tokenizer failed to load

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        model_kwargs={"temperature": temperature},
    )
    return HuggingFacePipeline(pipeline=pipe)


def setup_retriever(
    knowledge_base_path: str,
    embedding_model_id: str,
    text_splitter_chunk_size: int,
    text_splitter_chunk_overlap: int,
    retriever_search_k: int,
    default_knowledge_base_dataset: str,
):
    """데이터셋을 로드하고 리트리버를 설정하는 함수"""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)

    if knowledge_base_path == "default":
        print(
            f"Loading default KorQuAD dataset ({default_knowledge_base_dataset}) for RAG knowledge base..."
        )
        dataset = load_dataset(default_knowledge_base_dataset)["train"]
        contexts = list(set(item["context"] for item in dataset))
        docs = [Document(page_content=context) for context in contexts]
        print(f"Dataset loaded. Using {len(docs)} unique contexts.")

        print("Splitting documents and creating vector store...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_splitter_chunk_size,
            chunk_overlap=text_splitter_chunk_overlap,
        )
        splits = text_splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        print("In-memory vector store created successfully.")

    else:
        if not os.path.exists(knowledge_base_path):
            raise FileNotFoundError(
                f"Custom vector store not found at: {knowledge_base_path}"
            )
        print(f"Loading custom vector store from: {knowledge_base_path}")
        vectorstore = FAISS.load_local(
            knowledge_base_path, embeddings, allow_dangerous_deserialization=True
        )
        print("Custom vector store loaded successfully.")

    return vectorstore.as_retriever(search_kwargs={"k": retriever_search_k})


def create_rag_chain(llm, retriever, rag_prompt_template: str):
    """RAG 체인을 생성합니다."""
    prompt_template = ChatPromptTemplate.from_template(rag_prompt_template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain


from src.utils.config_loader import InferenceConfig

def main(cfg: InferenceConfig, question: str):
    """메인 실행 함수"""
    try:
        # 1. 모델 로드
        llm_pipeline = load_llm(
            model_type=cfg.model.model_type,
            model_id=cfg.model.model_id,
            unsupervised_lora_path=cfg.model.unsupervised_lora_path,
            sft_lora_path=cfg.model.sft_lora_path,
            max_new_tokens=cfg.generation.max_new_tokens,
            temperature=cfg.generation.temperature,
            model_max_seq_length=cfg.generation.model_max_seq_length,
        )
        if llm_pipeline is None:
            print("Model loading failed. Aborting RAG inference.")
            return

        # 2. 리트리버 설정
        retriever = setup_retriever(
            knowledge_base_path=cfg.knowledge_base_settings.knowledge_base,
            embedding_model_id=cfg.model.embedding_model_id,
            text_splitter_chunk_size=cfg.knowledge_base_settings.text_splitter_chunk_size,
            text_splitter_chunk_overlap=cfg.knowledge_base_settings.text_splitter_chunk_overlap,
            retriever_search_k=cfg.knowledge_base_settings.retriever_search_k,
            default_knowledge_base_dataset=cfg.knowledge_base_settings.default_knowledge_base_dataset,
        )
        print("Retriever setup successfully.")

        # 3. RAG 체인 생성
        rag_chain = create_rag_chain(llm_pipeline, retriever, cfg.generation.rag_prompt_template)
        print("RAG chain created successfully.")

        # 4. RAG 체인 실행
        print("\n--- RAG Inference Start ---")
        print(f"Model Type: {cfg.model.model_type}")
        print(f"Question: {question}")

        answer = rag_chain.invoke(question)

        print("\nAnswer:")
        print(answer)
        print("--- RAG Inference End ---")

    except Exception as e:
        print(f"An error occurred: {e}")
