import os
import torch
import typing as tp

# --- Import RAG components from rag_pipeline ---
from src.rag.rag_pipeline import load_llm, setup_retriever, create_rag_chain

# 토크나이저 병렬 처리 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from src.utils.config_loader import InferenceConfig

def main(cfg: InferenceConfig) -> None:
    """메인 실행 함수"""
    try:
        # 1. 모델 로드 (한 번만 실행)
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
            print("Model loading failed. Aborting RAG interactive CLI.")
            return

        # 2. 리트리버 설정 (한 번만 실행)
        retriever = setup_retriever(
            knowledge_base_path=cfg.knowledge_base_settings.knowledge_base,
            embedding_model_id=cfg.model.embedding_model_id,
            text_splitter_chunk_size=cfg.knowledge_base_settings.text_splitter_chunk_size,
            text_splitter_chunk_overlap=cfg.knowledge_base_settings.text_splitter_chunk_overlap,
            retriever_search_k=cfg.knowledge_base_settings.retriever_search_k,
            default_knowledge_base_dataset=cfg.knowledge_base_settings.default_knowledge_base_dataset,
        )
        print("Retriever setup successfully.")

        # 3. RAG 체인 생성 (한 번만 실행)
        rag_chain = create_rag_chain(llm_pipeline, retriever, cfg.generation.rag_prompt_template)
        print("RAG chain created successfully.")

        print("\n--- RAG Interactive CLI Start ---")
        print(f"Model Type: {cfg.model.model_type}")
        print("모델 및 지식 베이스 로딩 완료. 이제 질문을 입력하세요.")
        print("종료하려면 'exit' 또는 'quit'를 입력하세요.")

        while True:
            user_question = input("\n질문: ")
            if user_question.lower() in ["exit", "quit"]:
                print("RAG 추론을 종료합니다.")
                break

            print("답변 생성 중...")
            answer = rag_chain.invoke(user_question)

            print("\n답변:")
            print(answer)
            print("------------------------------------")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
