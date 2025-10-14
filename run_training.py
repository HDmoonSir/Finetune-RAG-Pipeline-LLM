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
    cfg: TrainConfig = load_train_config(args.config)

    # Create a unique output directory for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_output_dir = os.path.join(
        cfg.output_dir, f"{cfg.experiment_name}_{timestamp}"
    )
    os.makedirs(experiment_output_dir, exist_ok=True)

    # Save the resolved configuration to the experiment output directory for reproducibility
    OmegaConf.save(cfg, os.path.join(experiment_output_dir, "config.yaml"))
    print(f"Experiment results will be saved to: {experiment_output_dir}")

    # Overwrite the output_dir in cfg with the dynamic experiment_output_dir
    cfg.output_dir = experiment_output_dir

    # Dispatch based on whether Unsloth is used
    if cfg.training.use_unsloth:
        print("Running training with Unsloth...")
        train_unsloth.main(cfg=cfg)
    else:
        print("Running standard training (without Unsloth)...")
        train_llm.main(cfg=cfg)


if __name__ == "__main__":
    main()
