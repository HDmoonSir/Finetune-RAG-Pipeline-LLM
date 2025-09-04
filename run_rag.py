import os
import argparse
from omegaconf import OmegaConf

# Set CUDA_VISIBLE_DEVICES to restrict GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

from src.utils.config_loader import load_inference_config, load_train_config, InferenceConfig, TrainConfig
from src.rag import rag_pipeline, rag_interactive_cli

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline CLI for the project.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Base arguments for RAG inference (pipeline and cli) ---
    rag_parent_parser = argparse.ArgumentParser(add_help=False)
    rag_parent_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the inference configuration YAML file (e.g., configs/inference/local_rag.yaml)"
    )
    rag_parent_parser.add_argument(
        "--train_experiment_dir",
        type=str,
        default=None,
        help="Optional: Path to a completed training experiment directory. If provided, model_id and lora_path will be loaded from its config.yaml."
    )

    # --- Sub-parser for rag_pipeline (single query) ---
    parser_pipeline = subparsers.add_parser("pipeline", parents=[rag_parent_parser], help="Run a single RAG query.")
    parser_pipeline.add_argument("--question", type=str, required=True, help="The question to ask the RAG system.")

    # --- Sub-parser for rag_interactive_cli ---
    parser_cli = subparsers.add_parser("cli", parents=[rag_parent_parser], help="Run an interactive RAG CLI.")

    args = parser.parse_args()

    # Load the inference configuration
    inference_config: InferenceConfig = load_inference_config(args.config)

    # If a training experiment directory is provided, override model settings
    if args.train_experiment_dir:
        print(f"Loading model configuration from training experiment: {args.train_experiment_dir}")
        train_config_path = os.path.join(args.train_experiment_dir, "config.yaml")
        if not os.path.exists(train_config_path):
            raise FileNotFoundError(f"Training config not found in experiment directory: {train_config_path}")
        
        train_config: TrainConfig = load_train_config(train_config_path)
        
        inference_config.model.model_id = train_config.model.base_model_id
        print(f"Overridden model_id: {inference_config.model.model_id}")

        # Check for unsupervised adapter
        unsupervised_adapter_path = os.path.join(args.train_experiment_dir, "unsupervised_lora_adapter")
        if os.path.exists(unsupervised_adapter_path):
            inference_config.model.unsupervised_lora_path = unsupervised_adapter_path
            print(f"Found and set unsupervised_lora_path: {unsupervised_adapter_path}")

        # Check for sft adapter
        sft_adapter_path = os.path.join(args.train_experiment_dir, "sft_lora_adapter")
        if os.path.exists(sft_adapter_path):
            inference_config.model.sft_lora_path = sft_adapter_path
            print(f"Found and set sft_lora_path: {sft_adapter_path}")

    if args.command == "pipeline":
        print("Running RAG pipeline...")
        rag_pipeline.main(
            model_type=inference_config.model.model_type,
            model_id=inference_config.model.model_id,
            unsupervised_lora_path=inference_config.model.unsupervised_lora_path,
            sft_lora_path=inference_config.model.sft_lora_path,
            embedding_model_id=inference_config.model.embedding_model_id,
            knowledge_base=inference_config.knowledge_base_settings.knowledge_base,
            text_splitter_chunk_size=inference_config.knowledge_base_settings.text_splitter_chunk_size,
            text_splitter_chunk_overlap=inference_config.knowledge_base_settings.text_splitter_chunk_overlap,
            retriever_search_k=inference_config.knowledge_base_settings.retriever_search_k,
            rag_prompt_template=inference_config.generation.rag_prompt_template,
            max_new_tokens=inference_config.generation.max_new_tokens,
            temperature=inference_config.generation.temperature,
            model_max_seq_length=inference_config.generation.model_max_seq_length,
            question=args.question,
            default_knowledge_base_dataset=inference_config.knowledge_base_settings.default_knowledge_base_dataset
        )
    elif args.command == "cli":
        print("Starting RAG interactive CLI...")
        rag_interactive_cli.main(
            model_type=inference_config.model.model_type,
            model_id=inference_config.model.model_id,
            unsupervised_lora_path=inference_config.model.unsupervised_lora_path,
            sft_lora_path=inference_config.model.sft_lora_path,
            embedding_model_id=inference_config.model.embedding_model_id,
            knowledge_base=inference_config.knowledge_base_settings.knowledge_base,
            text_splitter_chunk_size=inference_config.knowledge_base_settings.text_splitter_chunk_size,
            text_splitter_chunk_overlap=inference_config.knowledge_base_settings.text_splitter_chunk_overlap,
            retriever_search_k=inference_config.knowledge_base_settings.retriever_search_k,
            rag_prompt_template=inference_config.generation.rag_prompt_template,
            max_new_tokens=inference_config.generation.max_new_tokens,
            temperature=inference_config.generation.temperature,
            model_max_seq_length=inference_config.generation.model_max_seq_length,
            default_knowledge_base_dataset=inference_config.knowledge_base_settings.default_knowledge_base_dataset
        )

if __name__ == "__main__":
    main()
