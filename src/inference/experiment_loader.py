import os
import typing as tp

from src.utils.config_loader import (
    load_train_config,
    EvalConfig,
    InferenceConfig,
    TrainConfig,
)


def update_config_from_experiment(
    config: tp.Union[InferenceConfig, EvalConfig], train_experiment_dir: str
) -> tp.Union[InferenceConfig, EvalConfig]:
    """
    Loads model configuration from a training experiment directory and updates the provided config object.

    Args:
        config: The inference or evaluation config object to update.
        train_experiment_dir: Path to the completed training experiment directory.

    Returns:
        The updated config object.

    Raises:
        FileNotFoundError: If the training config.yaml is not found in the experiment directory.
    """
    print(f"Loading model configuration from training experiment: {train_experiment_dir}")
    train_config_path = os.path.join(train_experiment_dir, "config.yaml")
    if not os.path.exists(train_config_path):
        raise FileNotFoundError(
            f"Training config not found in experiment directory: {train_config_path}"
        )

    train_config: TrainConfig = load_train_config(train_config_path)

    config.model.model_id = train_config.model.base_model_id
    print(f"Overridden model_id: {config.model.model_id}")

    # Check for unsupervised adapter
    unsupervised_adapter_path = os.path.join(
        train_experiment_dir, "unsupervised_lora_adapter"
    )
    if os.path.exists(unsupervised_adapter_path):
        config.model.unsupervised_lora_path = unsupervised_adapter_path
        print(f"Found and set unsupervised_lora_path: {unsupervised_adapter_path}")

    # Check for sft adapter
    sft_adapter_path = os.path.join(train_experiment_dir, "sft_lora_adapter")
    if os.path.exists(sft_adapter_path):
        config.model.sft_lora_path = sft_adapter_path
        print(f"Found and set sft_lora_path: {sft_adapter_path}")

    return config
