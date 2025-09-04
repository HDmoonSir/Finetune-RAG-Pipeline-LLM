
import argparse
from src.rag import build_vector_store, rag_pipeline, rag_interactive_cli
import config

def main():
    parser = argparse.ArgumentParser(description="RAG pipeline CLI for the project.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Sub-parser for build_vector_store ---
    parser_build_db = subparsers.add_parser("build-db", help="Build a FAISS vector store.")
    parser_build_db.add_argument(
        '--input_dir', type=str, default=config.DATA_DIR,
        help=f"Directory containing the .jsonl files to process. Defaults to '{config.DATA_DIR}'."
    )
    parser_build_db.add_argument(
        '--vector_store_path', type=str, default=config.VECTOR_STORE_DIR,
        help=f"Path to save the FAISS vector store. Defaults to '{config.VECTOR_STORE_DIR}'."
    )

    # --- Base arguments for RAG inference ---
    rag_parent_parser = argparse.ArgumentParser(add_help=False)
    rag_parent_parser.add_argument(
        "--model_type", type=str, default="api", choices=["api", "local", "local-quantized"],
        help="Type of model to use for inference."
    )
    rag_parent_parser.add_argument(
        "--lora_path", type=str, default=None,
        help="Path to the LoRA adapter (optional). Only applicable for local models."
    )
    rag_parent_parser.add_argument(
        "--knowledge_base", type=str, default="default",
        help=f"Knowledge base to use. 'default' for KorQuAD, or a path to a custom FAISS vector store (e.g., {config.VECTOR_STORE_DIR})."
    )
    rag_parent_parser.add_argument(
        "--model_id", type=str, default=config.LOCAL_MODEL_ID,
        help=f"Path/ID to the base model. Defaults to {config.LOCAL_MODEL_ID}."
    )

    # --- Sub-parser for rag_pipeline ---
    parser_pipeline = subparsers.add_parser("pipeline", parents=[rag_parent_parser], help="Run a single RAG query.")
    parser_pipeline.add_argument("question", type=str, help="The question to ask the RAG system.")

    # --- Sub-parser for rag_interactive_cli ---
    parser_cli = subparsers.add_parser("cli", parents=[rag_parent_parser], help="Run an interactive RAG CLI.")

    args = parser.parse_args()

    if args.command == "build-db":
        print("Building vector store...")
        build_vector_store.main(args.input_dir, args.vector_store_path)
    elif args.command == "pipeline":
        print("Running RAG pipeline...")
        rag_pipeline.main(
            model_type=args.model_type,
            question=args.question,
            lora_path=args.lora_path,
            knowledge_base=args.knowledge_base,
            model_id=args.model_id
        )
    elif args.command == "cli":
        print("Starting RAG interactive CLI...")
        rag_interactive_cli.main(
            model_type=args.model_type,
            lora_path=args.lora_path,
            knowledge_base=args.knowledge_base,
            model_id=args.model_id
        )

if __name__ == "__main__":
    main()
