import argparse
import os
from datetime import datetime
from omegaconf import OmegaConf

from src.utils.config_loader import (
    load_eval_config,
    EvalConfig,
)
from src.evaluation import evaluate_rag
from src.inference.experiment_loader import update_config_from_experiment


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
        eval_config = update_config_from_experiment(
            config=eval_config, train_experiment_dir=args.train_experiment_dir
        )

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

    # Overwrite the output_dir in cfg with the dynamic experiment_output_dir
    eval_config.output_dir = eval_output_dir

    print("Running RAG evaluation...")
    evaluate_rag.main(cfg=eval_config)


if __name__ == "__main__":
    main()
