import os
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
import typing as tp
import argparse

def main(args: argparse.Namespace) -> None:
    """주어진 데이터셋으로 Unsloth를 사용하여 언어 모델을 파인튜닝하는 메인 함수"""

    # 1. Unsloth를 사용하여 모델 및 토크나이저 로드
    model_id: str = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    max_seq_length = 2048 # Unsloth에서 권장하는 설정

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = max_seq_length,
        dtype = torch.bfloat16,
        load_in_4bit = True,
    )

    # 2. LoRA 설정
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Unsloth는 r=16 또는 32를 권장
        lora_alpha = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    model.print_trainable_parameters()

    # 3. 데이터셋 로드
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {args.dataset_path}")

    print(f"데이터셋 로딩 중: {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    print(f"데이터셋 로딩 완료: {dataset}")

    # 4. 학습 파라미터 설정
    output_dir = f"./{os.path.splitext(os.path.basename(args.dataset_path))[0]}_unsloth_finetuned"
    training_arguments: TrainingArguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_torch", # 8bit 옵티마이저 사용
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
    )

    # 5. SFTTrainer를 이용한 학습
    trainer: SFTTrainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text", # Unsloth SFTTrainer는 이 인자를 잘 처리함
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    trainer.train()

    # 6. LoRA 어댑터 저장
    final_model_path = f"./{os.path.splitext(os.path.basename(args.dataset_path))[0]}_unsloth_adapter"
    trainer.save_model(final_model_path)

    print(f"Fine-tuning completed and LoRA adapter saved to {final_model_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsloth와 SFTTrainer를 사용하여 언어 모델을 파인튜닝합니다.")
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='학습에 사용할 .jsonl 데이터셋 파일의 경로입니다.'
    )
    
    cli_args = parser.parse_args()
    main(cli_args)
