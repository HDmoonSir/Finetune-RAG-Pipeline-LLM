import os
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
import typing as tp
import config

def main(dataset_path: str, mode: str, base_model_path: str = None) -> None:
    """주어진 데이터셋으로 Unsloth를 사용하여 언어 모델을 파인튜닝하는 메인 함수"""

    # 1. Unsloth를 사용하여 모델 및 토크나이저 로드
    model_id: str = base_model_path if base_model_path else config.LOCAL_MODEL_ID
    max_seq_length = config.UNSLOTH_MAX_SEQ_LENGTH

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = max_seq_length,
        dtype = torch.bfloat16,
        load_in_4bit = True,
    )

    # 2. LoRA 설정
    model = FastLanguageModel.get_peft_model(
        model,
        r = config.UNSLOTH_LORA_R,
        lora_alpha = config.UNSLOTH_LORA_ALPHA,
        target_modules = config.UNSLOTH_LORA_TARGET_MODULES,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = config.RANDOM_SEED,
        use_rslora = False,
        loftq_config = None,
    )
    model.print_trainable_parameters()

    # 3. 데이터셋 로드
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")

    print(f"데이터셋 로딩 중: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"데이터셋 로딩 완료: {dataset}")

    # 4. 학습 파라미터 설정
    output_dir = os.path.join(config.DEFAULT_OUTPUT_DIR, f"{os.path.splitext(os.path.basename(dataset_path))[0]}_unsloth_{mode}_finetuned")
    training_arguments: TrainingArguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size = config.UNSLOTH_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = config.UNSLOTH_TRAIN_GRAD_ACCUM_STEPS,
        warmup_steps = config.UNSLOTH_WARMUP_STEPS,
        num_train_epochs = config.UNSLOTH_TRAIN_NUM_EPOCHS,
        learning_rate = config.UNSLOTH_LEARNING_RATE,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = config.UNSLOTH_OPTIMIZER,
        weight_decay = config.UNSLOTH_WEIGHT_DECAY,
        lr_scheduler_type = config.UNSLOTH_LR_SCHEDULER_TYPE,
        seed = config.RANDOM_SEED,
    )

    # 5. SFTTrainer를 이용한 학습
    trainer: SFTTrainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    trainer.train()

    # 6. LoRA 어댑터 저장
    final_model_path = os.path.join(config.DEFAULT_OUTPUT_DIR, f"{os.path.splitext(os.path.basename(dataset_path))[0]}_unsloth_{mode}_adapter")
    trainer.save_model(final_model_path)

    print(f"Fine-tuning completed and LoRA adapter saved to {final_model_path}!")
