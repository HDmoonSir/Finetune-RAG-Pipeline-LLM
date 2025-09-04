
import os

# --- GENERAL ---
PROJECT_NAME = "LLM Finetune & RAG Pipeline"
RANDOM_SEED = 3407

# --- PATHS ---
DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "data_result"
VECTOR_STORE_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "vector_store")
QA_DATASET_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "gemini_generated_qa_dataset.jsonl")
UNSUPERVISED_DATASET_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "gemini_generated_unsupervised_dataset.jsonl")

# --- MODELS ---
# Main model for local fine-tuning and inference
LOCAL_MODEL_ID = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
# Embedding model for RAG
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# API models
GEMINI_PREPROCESS_MODEL = "gemini-1.5-flash"
GEMINI_API_MODEL = "gemini-1.5-flash-latest"

# --- DATA PREPROCESSING (preprocess_with_gemini.py) ---
TEXT_PAGE_BATCH_SIZE = 15
QA_PROMPT_TEMPLATE = """아래 [문서 내용]은 '---'로 구분된 여러 텍스트 섹션을 포함하고 있습니다. 이 내용을 바탕으로, 다양하고 핵심적인 Q&A 쌍 25개를 "qa_pairs"를 키로 하는 JSON 배열 형식으로 생성해 주세요. 질문과 답변은 모두 한국어로 작성해야 합니다. 출력은 오직 JSON 배열이어야 합니다.

[문서 내용]
{text}"""
UNSUPERVISED_PROMPT_TEMPLATE = """아래 [문서 내용]은 '---'로 구분된 여러 텍스트 섹션을 포함하고 있습니다. 이 내용을 바탕으로, 문서의 핵심 주제와 정보를 포괄하는 상세하고 구조화된 요약 텍스트를 생성해 주세요. 원본의 전문적인 스타일과 톤을 유지하며, 여러 단락으로 구성된 가독성 높은 글을 작성해야 합니다. 출력은 오직 생성된 텍스트여야 합니다.

[문서 내용]
{text}"""

# --- RAG PIPELINE (rag_pipeline.py, rag_interactive_cli.py) ---
KNOWLEDGE_BASE_DATASET = "squad_kor_v1" # For default retriever
TEXT_SPLITTER_CHUNK_SIZE = 1000
TEXT_SPLITTER_CHUNK_OVERLAP = 100
RETRIEVER_SEARCH_K = 3
RAG_PROMPT_TEMPLATE = """
        다음 컨텍스트 정보를 사용하여 질문에 답변해 주세요. 
        만약 컨텍스트에 답변이 없다면, 모른다고 답해 주세요.

        컨텍스트: 
        {context}

        질문: 
        {question} 

        답변:
        """
RAG_MAX_NEW_TOKENS = 512
RAG_TEMPERATURE = 0.1

# --- TRAINING (train_llm.py) ---
TRAIN_MAX_SEQ_LENGTH = 2048
TRAIN_LORA_R = 8
TRAIN_LORA_ALPHA = 16
TRAIN_LORA_DROPOUT = 0.05
TRAIN_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRAIN_NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 1
TRAIN_GRAD_ACCUM_STEPS = 4
TRAIN_OPTIMIZER = "adamw_torch"
TRAIN_LEARNING_RATE = 2e-4
TRAIN_LR_SCHEDULER_TYPE = "cosine"
TRAIN_WARMUP_RATIO = 0.03
TRAIN_LOGGING_STEPS = 25
TRAIN_SAVE_STEPS = 250

# --- UNSLOTH TRAINING (train_unsloth.py) ---
UNSLOTH_MAX_SEQ_LENGTH = 2048
UNSLOTH_LORA_R = 16
UNSLOTH_LORA_ALPHA = 32
UNSLOTH_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
UNSLOTH_TRAIN_NUM_EPOCHS = 3
UNSLOTH_TRAIN_BATCH_SIZE = 2
UNSLOTH_TRAIN_GRAD_ACCUM_STEPS = 4
UNSLOTH_OPTIMIZER = "adamw_torch"
UNSLOTH_LEARNING_RATE = 2e-4
UNSLOTH_LR_SCHEDULER_TYPE = "linear"
UNSLOTH_WARMUP_STEPS = 5
UNSLOTH_WEIGHT_DECAY = 0.01
