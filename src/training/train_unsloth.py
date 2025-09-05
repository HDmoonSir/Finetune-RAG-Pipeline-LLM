import os
import torch
from transformers import AutoTokenizer, TrainingArguments, set_seed
from datasets import load_dataset
from trl import SFTTrainer
import typing as tp
from peft import PeftModel

# Unsloth should be imported at the top for global patching
from unsloth import FastLanguageModel


def main(
    experiment_name: str,
    output_dir: str,
    seed: int,
    base_model_id: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,  # Added lora_dropout
    lora_target_modules: tp.List[str],
    mode: str,
    dataset_path: str,
    max_seq_length: int,
    num_epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    optimizer: str,
    learning_rate: float,
    lr_scheduler_type: str,
    unsupervised_lora_path: tp.Optional[str] = None,
    warmup_steps: tp.Optional[int] = None,
    weight_decay: tp.Optional[float] = None,
    logging_steps: int = 25,
    save_steps: int = 250,
) -> None:
    """주어진 데이터셋으로 Unsloth를 사용하여 언어 모델을 파인튜닝하는 메인 함수"""

    set_seed(seed)

    # 1. Unsloth를 사용하여 모델 및 토크나이저 로드
    model_id: str = base_model_id

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
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
    if unsupervised_lora_path and mode == "sft":
        print(
            f"Loading and merging unsupervised LoRA from {unsupervised_lora_path} before SFT..."
        )
        model = PeftModel.from_pretrained(model, unsupervised_lora_path)
        model = model.merge_and_unload()
        print("Unsupervised LoRA merged.")

    # 2. LoRA 설정
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,  # Use the passed lora_dropout
        bias="none",
        use_gradient_checkpointing=True,
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()

    # 3. 데이터셋 로드
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")

    print(f"데이터셋 로딩 중: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"데이터셋 로딩 완료: {dataset}")

    # 4. 학습 파라미터 설정
    training_arguments: TrainingArguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=logging_steps,
        optim=optimizer,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        save_steps=save_steps,
    )

    # 5. SFTTrainer를 이용한 학습
    trainer_kwargs = {}
    if mode == "unsupervised":
        trainer_kwargs["dataset_text_field"] = "text"
    elif mode == "sft":
        # For SFT, the dataset is expected to be formatted with 'text' or 'formatted_text'
        # SFTTrainer will handle this by default if the dataset is properly prepared.
        pass
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    trainer: SFTTrainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        **trainer_kwargs,
    )

    trainer.train()

    # 6. LoRA 어댑터 저장
    if mode == "unsupervised":
        adapter_name = "unsupervised_lora_adapter"
    elif mode == "sft":
        adapter_name = "sft_lora_adapter"
    else:
        adapter_name = "adapter"  # Fallback

    final_adapter_path = os.path.join(output_dir, adapter_name)
    trainer.save_model(final_adapter_path)

    print(
        f"Fine-tuning completed and LoRA adapter saved to {final_adapter_path}! (Experiment: {experiment_name})"
    )
