import dataclasses
import typing as tp

from omegaconf import DictConfig, OmegaConf


# --- Base Settings --- #
@dataclasses.dataclass
class BaseModelSettings:
    model_type: str
    model_id: str
    embedding_model_id: str
    unsupervised_lora_path: tp.Optional[str] = None
    sft_lora_path: tp.Optional[str] = None


@dataclasses.dataclass
class TrainModelSettings:
    base_model_id: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tp.List[str]
    unsupervised_lora_path: tp.Optional[str] = None  # SFT 시에만 사용


@dataclasses.dataclass
class TrainingSettings:
    mode: str  # "sft" or "unsupervised"
    use_unsloth: bool
    dataset_path: str
    max_seq_length: int
    num_epochs: int
    batch_size: int
    grad_accum_steps: int
    optimizer: str
    learning_rate: float
    lr_scheduler_type: str
    logging_steps: int
    save_steps: int
    warmup_ratio: tp.Optional[float] = None
    warmup_steps: tp.Optional[int] = None
    weight_decay: tp.Optional[float] = None


@dataclasses.dataclass
class GenerationSettings:
    max_new_tokens: int
    temperature: float
    model_max_seq_length: int
    rag_prompt_template: tp.Optional[str] = None


@dataclasses.dataclass
class KnowledgeBaseSettings:
    knowledge_base: str
    text_splitter_chunk_size: int
    text_splitter_chunk_overlap: int
    retriever_search_k: int
    default_knowledge_base_dataset: str


# --- Main Configs --- #
@dataclasses.dataclass
class TrainConfig:
    experiment_name: str
    output_dir: str
    seed: int
    model: TrainModelSettings
    training: TrainingSettings


@dataclasses.dataclass
class EvalConfig:
    eval_dataset_path: str
    num_samples: int
    model: BaseModelSettings
    generation: GenerationSettings
    knowledge_base_settings: KnowledgeBaseSettings


@dataclasses.dataclass
class InferenceConfig:
    model: BaseModelSettings
    generation: GenerationSettings
    knowledge_base_settings: KnowledgeBaseSettings


@dataclasses.dataclass
class DataPreprocessingConfig:
    mode: str
    gemini_preprocess_model: str
    data_dir: str
    text_page_batch_size: int
    qa_prompt_template: str
    unsupervised_prompt_template: str
    qa_dataset_path: str
    unsupervised_dataset_path: str
    default_output_dir: str


@dataclasses.dataclass
class VectorStoreBuildConfig:
    input_dir: str
    vector_store_path: str
    embedding_model_id: str
    text_splitter_chunk_size: int
    text_splitter_chunk_overlap: int


def load_config(
    config_path: str,
    config_type: tp.Type[
        tp.Union[
            TrainConfig,
            EvalConfig,
            InferenceConfig,
            DataPreprocessingConfig,
            VectorStoreBuildConfig,
        ]
    ],
) -> tp.Union[
    TrainConfig,
    EvalConfig,
    InferenceConfig,
    DataPreprocessingConfig,
    VectorStoreBuildConfig,
]:
    """
    Loads a YAML configuration file and converts it into a structured dataclass.

    Args:
        config_path: The path to the YAML configuration file.
        config_type: The dataclass type to load the configuration into (e.g., TrainConfig, EvalConfig).

    Returns:
        An instance of the specified config_type dataclass.

    Raises:
        FileNotFoundError: If the specified config_path does not exist.
        Exception: For other errors during loading or structuring.
    """
    try:
        # Load the YAML file as a DictConfig
        cfg: DictConfig = OmegaConf.load(config_path)

        # Create a structured config from the dataclass
        structured_cfg = OmegaConf.structured(config_type)

        # Merge the loaded config into the structured config for validation and default values
        merged_cfg = OmegaConf.merge(structured_cfg, cfg)

        # Perform custom validation for TrainConfig
        if config_type is TrainConfig:
            if (
                merged_cfg.training.warmup_ratio is not None
                and merged_cfg.training.warmup_steps is not None
            ):
                raise ValueError(
                    "Cannot specify both 'warmup_ratio' and 'warmup_steps' in training configuration. Please choose one."
                )

        # Convert the merged DictConfig to the dataclass instance
        return OmegaConf.to_object(merged_cfg)  # Call with only one argument
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    except Exception as e:
        raise Exception(f"Error loading or structuring config from {config_path}: {e}")


def load_train_config(config_path: str) -> TrainConfig:
    return load_config(config_path, TrainConfig)  # type: ignore


def load_eval_config(config_path: str) -> EvalConfig:
    return load_config(config_path, EvalConfig)  # type: ignore


def load_inference_config(config_path: str) -> InferenceConfig:
    return load_config(config_path, InferenceConfig)  # type: ignore


def load_data_preprocessing_config(config_path: str) -> DataPreprocessingConfig:
    return load_config(config_path, DataPreprocessingConfig)  # type: ignore


def load_vector_store_build_config(config_path: str) -> VectorStoreBuildConfig:
    return load_config(config_path, VectorStoreBuildConfig)  # type: ignore
