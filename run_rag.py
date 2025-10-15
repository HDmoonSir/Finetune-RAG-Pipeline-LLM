import os
import argparse
from omegaconf import OmegaConf

from src.utils.config_loader import (
    load_inference_config,
    InferenceConfig,
)
from src.rag import rag_pipeline, rag_interactive_cli
from src.inference.experiment_loader import update_config_from_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline CLI for the project.")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # --- Base arguments for RAG inference (pipeline and cli) ---
    rag_parent_parser = argparse.ArgumentParser(add_help=False)
    rag_parent_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the inference configuration YAML file (e.g., configs/inference/local_rag.yaml)",
    )
    rag_parent_parser.add_argument(
        "--train_experiment_dir",
        type=str,
        default=None,
        help="Optional: Path to a completed training experiment directory. If provided, model_id and lora_path will be loaded from its config.yaml.",
    )

    # --- Sub-parser for rag_pipeline (single query) ---
    parser_pipeline = subparsers.add_parser(
        "pipeline", parents=[rag_parent_parser], help="Run a single RAG query."
    )
    parser_pipeline.add_argument(
        "--question",
        type=str,
        required=True,
        help="The question to ask the RAG system.",
    )

    # --- Sub-parser for rag_interactive_cli ---
    parser_cli = subparsers.add_parser(
        "cli", parents=[rag_parent_parser], help="Run an interactive RAG CLI."
    )

    args = parser.parse_args()

    # Load the inference configuration
    inference_config: InferenceConfig = load_inference_config(args.config)

    # If a training experiment directory is provided, override model settings
    if args.train_experiment_dir:
        inference_config = update_config_from_experiment(
            config=inference_config, train_experiment_dir=args.train_experiment_dir
        )

    if args.command == "pipeline":
        print("Running RAG pipeline...")
        rag_pipeline.main(cfg=inference_config, question=args.question)
    elif args.command == "cli":
        print("Starting RAG interactive CLI...")
        rag_interactive_cli.main(cfg=inference_config)


if __name__ == "__main__":
    main()
