import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
import typing as tp
import config

def main(args) -> None:
    """주어진 데이터셋으로 언어 모델을 파인튜닝하는 메인 함수"""

    dataset_path = args.dataset_path
    mode = args.mode
    base_model_path = getattr(args, 'base_model_path', None)

    # 1. 모델 및 토크나이저 로드
    model_id: str = base_model_path if base_model_path else config.LOCAL_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={'': torch.cuda.current_device()},
    )

    # 2. LoRA 설정
    lora_config: LoraConfig = LoraConfig(
        r=config.TRAIN_LORA_R,
        lora_alpha=config.TRAIN_LORA_ALPHA,
        target_modules=config.TRAIN_LORA_TARGET_MODULES,
        lora_dropout=config.TRAIN_LORA_DROPOUT,
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
    output_dir = os.path.join(config.DEFAULT_OUTPUT_DIR, f"{os.path.splitext(os.path.basename(dataset_path))[0]}_{mode}_finetuned")
    training_arguments: TrainingArguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.TRAIN_NUM_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.TRAIN_GRAD_ACCUM_STEPS,
        optim=config.TRAIN_OPTIMIZER,
        learning_rate=config.TRAIN_LEARNING_RATE,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3, # This is not in config, but it is fine.
        warmup_ratio=config.TRAIN_WARMUP_RATIO,
        lr_scheduler_type=config.TRAIN_LR_SCHEDULER_TYPE,
        logging_steps=config.TRAIN_LOGGING_STEPS,
        save_steps=config.TRAIN_SAVE_STEPS,
        group_by_length=True,
    )

    # 5. SFTTrainer를 이용한 학습
    trainer_kwargs = {}
    if mode == 'unsupervised':
        trainer_kwargs['dataset_text_field'] = 'text'

    trainer: SFTTrainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=config.TRAIN_MAX_SEQ_LENGTH,
        args=training_arguments,
        **trainer_kwargs,
    )

    trainer.train()

    # 6. LoRA 어댑터 저장
    final_model_path = os.path.join(config.DEFAULT_OUTPUT_DIR, f"{os.path.splitext(os.path.basename(dataset_path))[0]}_{mode}_adapter")
    trainer.save_model(final_model_path)

    print(f"Fine-tuning completed and LoRA adapter saved to {final_model_path}!")
