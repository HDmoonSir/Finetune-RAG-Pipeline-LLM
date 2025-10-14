import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from trl import SFTTrainer
import typing as tp

from src.utils.config_loader import TrainConfig


def main(cfg: TrainConfig) -> None:
    """주어진 데이터셋으로 언어 모델을 파인튜닝하는 메인 함수"""

    set_seed(cfg.seed)

    # 1. 모델 및 토크나이저 로드
    model_id: str = cfg.model.base_model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
    )

    # 2. SFT를 위한 사전 병합 (선택 사항)
    if cfg.model.unsupervised_lora_path and cfg.training.mode == "sft":
        print(
            f"Loading and merging unsupervised LoRA from {cfg.model.unsupervised_lora_path} before SFT..."
        )
        model = PeftModel.from_pretrained(model, cfg.model.unsupervised_lora_path)
        model = model.merge_and_unload()
        print("Unsupervised LoRA merged.")

    # 3. LoRA 설정
    lora_config: LoraConfig = LoraConfig(
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        target_modules=cfg.model.lora_target_modules,
        lora_dropout=cfg.model.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 데이터셋 로드
    if not os.path.exists(cfg.training.dataset_path):
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {cfg.training.dataset_path}")

    print(f"데이터셋 로딩 중: {cfg.training.dataset_path}")
    dataset = load_dataset("json", data_files=cfg.training.dataset_path, split="train")
    print(f"데이터셋 로딩 완료: {dataset}")

    # 4. 학습 파라미터 설정
    training_arguments_kwargs = {
        "output_dir": cfg.output_dir,  # Use the experiment-specific output_dir from cfg
        "num_train_epochs": cfg.training.num_epochs,
        "per_device_train_batch_size": cfg.training.batch_size,
        "gradient_accumulation_steps": cfg.training.grad_accum_steps,
        "optim": cfg.training.optimizer,
        "learning_rate": cfg.training.learning_rate,
        "fp16": False,
        "bf16": True,
        "max_grad_norm": 0.3,
        "lr_scheduler_type": cfg.training.lr_scheduler_type,
        "logging_steps": cfg.training.logging_steps,
        "save_steps": cfg.training.save_steps,
        "group_by_length": True,
    }

    if cfg.training.warmup_ratio is not None:
        training_arguments_kwargs["warmup_ratio"] = cfg.training.warmup_ratio
    elif cfg.training.warmup_steps is not None:
        training_arguments_kwargs["warmup_steps"] = cfg.training.warmup_steps

    if cfg.training.weight_decay is not None:
        training_arguments_kwargs["weight_decay"] = cfg.training.weight_decay

    training_arguments: TrainingArguments = TrainingArguments(
        **training_arguments_kwargs
    )

    # 5. SFTTrainer를 이용한 학습
    trainer_kwargs = {}
    if cfg.training.mode == "unsupervised":
        trainer_kwargs["dataset_text_field"] = "text"
    elif cfg.training.mode == "sft":
        pass
    else:
        raise ValueError(f"Unknown training mode: {cfg.training.mode}")

    trainer: SFTTrainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=cfg.training.max_seq_length,
        args=training_arguments,
        **trainer_kwargs,
    )

    trainer.train()

    # 6. LoRA 어댑터 저장
    if cfg.training.mode == "unsupervised":
        adapter_name = "unsupervised_lora_adapter"
    elif cfg.training.mode == "sft":
        adapter_name = "sft_lora_adapter"
    else:
        adapter_name = "adapter"  # Fallback

    final_adapter_path = os.path.join(cfg.output_dir, adapter_name)
    trainer.save_model(final_adapter_path)

    print(
        f"Fine-tuning completed and LoRA adapter saved to {final_adapter_path}! (Experiment: {cfg.experiment_name})"
    )