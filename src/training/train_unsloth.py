import os
import torch
from transformers import AutoTokenizer, TrainingArguments, set_seed
from datasets import load_dataset
from trl import SFTTrainer
import typing as tp
from peft import PeftModel

# Unsloth should be imported at the top for global patching
from unsloth import FastLanguageModel

from src.utils.config_loader import TrainConfig


def main(cfg: TrainConfig) -> None:
    """주어진 데이터셋으로 Unsloth를 사용하여 언어 모델을 파인튜닝하는 메인 함수"""

    set_seed(cfg.seed)

    # 1. Unsloth를 사용하여 모델 및 토크나이저 로드
    model_id: str = cfg.model.base_model_id

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=cfg.training.max_seq_length,
            dtype=None,  # Unsloth handles dtype internally
            load_in_4bit=True,
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA Out of Memory Error while loading Unsloth model {model_id}: {e}")
        print(
            "This model is too large for your GPU. Try a smaller model or free up VRAM."
        )
        return
    except Exception as e:
        print(f"Error loading quantized model {model_id} with Unsloth: {e}")
        return

    # SFT를 위한 사전 병합 (선택 사항)
    if cfg.model.unsupervised_lora_path and cfg.training.mode == "sft":
        print(
            f"Loading and merging unsupervised LoRA from {cfg.model.unsupervised_lora_path} before SFT..."
        )
        model = PeftModel.from_pretrained(model, cfg.model.unsupervised_lora_path)
        model = model.merge_and_unload()
        print("Unsupervised LoRA merged.")

    # 2. LoRA 설정
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        target_modules=cfg.model.lora_target_modules,
        lora_dropout=cfg.model.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=cfg.seed,
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()

    # 3. 데이터셋 로드
    if not os.path.exists(cfg.training.dataset_path):
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {cfg.training.dataset_path}")

    print(f"데이터셋 로딩 중: {cfg.training.dataset_path}")
    dataset = load_dataset("json", data_files=cfg.training.dataset_path, split="train")
    print(f"데이터셋 로딩 완료: {dataset}")

    # 4. 학습 파라미터 설정
    training_arguments: TrainingArguments = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.grad_accum_steps,
        warmup_steps=cfg.training.warmup_steps,
        num_train_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=cfg.training.logging_steps,
        optim=cfg.training.optimizer,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        seed=cfg.seed,
        save_steps=cfg.training.save_steps,
    )

    # 5. SFTTrainer를 이용한 학습
    trainer_kwargs = {}
    if cfg.training.mode == "unsupervised":
        trainer_kwargs["dataset_text_field"] = "text"
    elif cfg.training.mode == "sft":
        # For SFT, the dataset is expected to be formatted with 'text' or 'formatted_text'
        # SFTTrainer will handle this by default if the dataset is properly prepared.
        pass
    else:
        raise ValueError(f"Unknown training mode: {cfg.training.mode}")

    trainer: SFTTrainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        max_seq_length=cfg.training.max_seq_length,
        tokenizer=tokenizer,
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