import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
import typing as tp
import argparse

def main(args: argparse.Namespace) -> None:
    """주어진 데이터셋으로 언어 모델을 파인튜닝하는 메인 함수"""

    # 1. 모델 및 토크나이저 로드
    model_id: str = "MLP-KTLim/llama-3-Korean-Bllossom-8B" # 사용하고자 하는 Full-tuned 모델 ID

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
        r=2,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 양자화 관련 코드 제거됨
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 데이터셋 로드
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {args.dataset_path}")

    print(f"데이터셋 로딩 중: {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    print(f"데이터셋 로딩 완료: {dataset}")

    # 4. 학습 파라미터 설정
    output_dir = f"./{os.path.splitext(os.path.basename(args.dataset_path))[0]}_finetuned"
    training_arguments: TrainingArguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="adamw_torch", # 옵티마이저 변경
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_steps=250,
        group_by_length=True,
    )

    # 5. SFTTrainer를 이용한 학습
    trainer: SFTTrainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_arguments,
    )

    trainer.train()

    # 6. LoRA 어댑터 저장
    final_model_path = f"./{os.path.splitext(os.path.basename(args.dataset_path))[0]}_adapter"
    trainer.save_model(final_model_path)

    print(f"Fine-tuning completed and LoRA adapter saved to {final_model_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA와 SFTTrainer를 사용하여 언어 모델을 파인튜닝합니다.")
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='학습에 사용할 .jsonl 데이터셋 파일의 경로입니다.'
    )
    
    cli_args = parser.parse_args()
    main(cli_args)
