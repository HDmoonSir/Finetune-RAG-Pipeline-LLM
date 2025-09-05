import argparse
import os
from datetime import datetime
from omegaconf import OmegaConf

from src.utils.config_loader import (
    load_eval_config,
    load_train_config,
    EvalConfig,
    TrainConfig,
)
from src.evaluation import evaluate_rag


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation CLI for the project.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the evaluation configuration YAML file (e.g., configs/eval/default_eval.yaml)",
    )
    parser.add_argument(
        "--train_experiment_dir",
        type=str,
        default=None,
        help="Optional: Path to a completed training experiment directory. If provided, model_id and lora_path will be loaded from its config.yaml.",
    )
    args = parser.parse_args()

    # Load the evaluation configuration
    eval_config: EvalConfig = load_eval_config(args.config)

    # If a training experiment directory is provided, override model settings
    if args.train_experiment_dir:
        print(
            f"Loading model configuration from training experiment: {args.train_experiment_dir}"
        )
        train_config_path = os.path.join(args.train_experiment_dir, "config.yaml")
        if not os.path.exists(train_config_path):
            raise FileNotFoundError(
                f"Training config not found in experiment directory: {train_config_path}"
            )

        train_config: TrainConfig = load_train_config(train_config_path)

        eval_config.model.model_id = train_config.model.base_model_id
        print(f"Overridden model_id: {eval_config.model.model_id}")

        # Check for unsupervised adapter
        unsupervised_adapter_path = os.path.join(
            args.train_experiment_dir, "unsupervised_lora_adapter"
        )
        if os.path.exists(unsupervised_adapter_path):
            eval_config.model.unsupervised_lora_path = unsupervised_adapter_path
            print(f"Found and set unsupervised_lora_path: {unsupervised_adapter_path}")

        # Check for sft adapter
        sft_adapter_path = os.path.join(args.train_experiment_dir, "sft_lora_adapter")
        if os.path.exists(sft_adapter_path):
            eval_config.model.sft_lora_path = sft_adapter_path
            print(f"Found and set sft_lora_path: {sft_adapter_path}")

    # Create a unique output directory for evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Fix: Assign model_id_safe to a variable before using in f-string
    model_id_safe = eval_config.model.model_id.replace("/", "_")
    eval_output_dir = os.path.join(
        "exp_results", "evaluation_results", f"eval_{model_id_safe}_{timestamp}"
    )
    os.makedirs(eval_output_dir, exist_ok=True)
    print(f"Evaluation results will be saved to: {eval_output_dir}")

    # Save the resolved evaluation configuration to the output directory for reproducibility
    OmegaConf.save(eval_config, os.path.join(eval_output_dir, "eval_config.yaml"))

    print("Running RAG evaluation...")
    evaluate_rag.main(
        model_type=eval_config.model.model_type,
        model_id=eval_config.model.model_id,
        unsupervised_lora_path=eval_config.model.unsupervised_lora_path,
        sft_lora_path=eval_config.model.sft_lora_path,
        embedding_model_id=eval_config.model.embedding_model_id,
        knowledge_base=eval_config.knowledge_base_settings.knowledge_base,
        text_splitter_chunk_size=eval_config.knowledge_base_settings.text_splitter_chunk_size,
        text_splitter_chunk_overlap=eval_config.knowledge_base_settings.text_splitter_chunk_overlap,
        retriever_search_k=eval_config.knowledge_base_settings.retriever_search_k,
        default_knowledge_base_dataset=eval_config.knowledge_base_settings.default_knowledge_base_dataset,
        rag_prompt_template=eval_config.generation.rag_prompt_template,
        max_new_tokens=eval_config.generation.max_new_tokens,
        temperature=eval_config.generation.temperature,
        model_max_seq_length=eval_config.generation.model_max_seq_length,
        eval_dataset_path=eval_config.eval_dataset_path,
        num_samples=eval_config.num_samples,
        output_base_dir=eval_output_dir,
    )


if __name__ == "__main__":
    main()
