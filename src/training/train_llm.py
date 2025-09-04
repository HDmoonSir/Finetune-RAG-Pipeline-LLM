import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from trl import SFTTrainer
import typing as tp

def main(
    experiment_name: str,
    output_dir: str,
    seed: int,
    base_model_id: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
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
    warmup_ratio: tp.Optional[float] = None,
    warmup_steps: tp.Optional[int] = None,
    weight_decay: tp.Optional[float] = None,
    logging_steps: int = 25,
    save_steps: int = 250,
) -> None:
    """주어진 데이터셋으로 언어 모델을 파인튜닝하는 메인 함수"""

    set_seed(seed)

    # 1. 모델 및 토크나이저 로드
    model_id: str = base_model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={'': torch.cuda.current_device()},
    )

    # 2. SFT를 위한 사전 병합 (선택 사항)
    if unsupervised_lora_path and mode == 'sft':
        print(f"Loading and merging unsupervised LoRA from {unsupervised_lora_path} before SFT...")
        model = PeftModel.from_pretrained(model, unsupervised_lora_path)
        model = model.merge_and_unload()
        print("Unsupervised LoRA merged.")

    # 3. LoRA 설정
    lora_config: LoraConfig = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 데이터셋 로드
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")

    print(f"데이터셋 로딩 중: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"데이터셋 로딩 완료: {dataset}")

    # 4. 학습 파라미터 설정
    # output_dir is now the specific experiment output directory passed from run_training.py
    training_arguments_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "optim": optimizer,
        "learning_rate": learning_rate,
        "fp16": False,
        "bf16": True,
        "max_grad_norm": 0.3,
        "lr_scheduler_type": lr_scheduler_type,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "group_by_length": True,
    }

    if warmup_ratio is not None:
        training_arguments_kwargs["warmup_ratio"] = warmup_ratio
    elif warmup_steps is not None:
        training_arguments_kwargs["warmup_steps"] = warmup_steps
    
    if weight_decay is not None:
        training_arguments_kwargs["weight_decay"] = weight_decay

    training_arguments: TrainingArguments = TrainingArguments(**training_arguments_kwargs)

    # 5. SFTTrainer를 이용한 학습
    trainer_kwargs = {}
    if mode == 'unsupervised':
        trainer_kwargs['dataset_text_field'] = 'text'
    elif mode == 'sft':
        # For SFT, the dataset is expected to be formatted with 'text' or 'formatted_text'
        # SFTTrainer will handle this by default if the dataset is properly prepared.
        # No need to set dataset_text_field explicitly unless it's a custom field.
        pass
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    trainer: SFTTrainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=max_seq_length,
        args=training_arguments,
        **trainer_kwargs,
    )

    trainer.train()

    # 6. LoRA 어댑터 저장
    if mode == 'unsupervised':
        adapter_name = "unsupervised_lora_adapter"
    elif mode == 'sft':
        adapter_name = "sft_lora_adapter"
    else:
        adapter_name = "adapter" # Fallback

    final_adapter_path = os.path.join(output_dir, adapter_name)
    trainer.save_model(final_adapter_path)

    print(f"Fine-tuning completed and LoRA adapter saved to {final_adapter_path}! (Experiment: {experiment_name})")