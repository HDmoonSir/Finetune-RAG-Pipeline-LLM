import argparse
import os
from datetime import datetime
from omegaconf import OmegaConf

from src.utils.config_loader import load_train_config, TrainConfig
from src.training import train_llm, train_unsloth


def main() -> None:
    parser = argparse.ArgumentParser(description="Model training CLI.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration YAML file (e.g., configs/train/llama-8b-sft.yaml)",
    )
    args = parser.parse_args()

    # Load the training configuration
    config: TrainConfig = load_train_config(args.config)

    # Create a unique output directory for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_output_dir = os.path.join(
        config.output_dir, f"{config.experiment_name}_{timestamp}"
    )
    os.makedirs(experiment_output_dir, exist_ok=True)

    # Save the resolved configuration to the experiment output directory for reproducibility
    OmegaConf.save(config, os.path.join(experiment_output_dir, "config.yaml"))
    print(f"Experiment results will be saved to: {experiment_output_dir}")

    # Dispatch based on whether Unsloth is used
    if config.training.use_unsloth:
        print("Running training with Unsloth...")
        train_unsloth.main(
            experiment_name=config.experiment_name,
            output_dir=experiment_output_dir,  # Pass the specific output directory
            seed=config.seed,
            base_model_id=config.model.base_model_id,
            unsupervised_lora_path=config.model.unsupervised_lora_path,
            lora_r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,  # Pass lora_dropout
            lora_target_modules=config.model.lora_target_modules,
            mode=config.training.mode,  # Pass the training mode (sft/unsupervised)
            dataset_path=config.training.dataset_path,
            max_seq_length=config.training.max_seq_length,
            num_epochs=config.training.num_epochs,
            batch_size=config.training.batch_size,
            grad_accum_steps=config.training.grad_accum_steps,
            optimizer=config.training.optimizer,
            learning_rate=config.training.learning_rate,
            lr_scheduler_type=config.training.lr_scheduler_type,
            warmup_steps=config.training.warmup_steps,
            weight_decay=config.training.weight_decay,
            logging_steps=config.training.logging_steps,
            save_steps=config.training.save_steps,
        )
    else:
        print("Running standard training (without Unsloth)...")
        train_llm.main(
            experiment_name=config.experiment_name,
            output_dir=experiment_output_dir,  # Pass the specific output directory
            seed=config.seed,
            base_model_id=config.model.base_model_id,
            unsupervised_lora_path=config.model.unsupervised_lora_path,
            lora_r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            lora_target_modules=config.model.lora_target_modules,
            mode=config.training.mode,  # Pass the training mode (sft/unsupervised)
            dataset_path=config.training.dataset_path,
            max_seq_length=config.training.max_seq_length,
            num_epochs=config.training.num_epochs,
            batch_size=config.training.batch_size,
            grad_accum_steps=config.training.grad_accum_steps,
            optimizer=config.training.optimizer,
            learning_rate=config.training.learning_rate,
            lr_scheduler_type=config.training.lr_scheduler_type,
            warmup_ratio=config.training.warmup_ratio,
            warmup_steps=config.training.warmup_steps,
            weight_decay=config.training.weight_decay,
            logging_steps=config.training.logging_steps,
            save_steps=config.training.save_steps,
        )


if __name__ == "__main__":
    main()
