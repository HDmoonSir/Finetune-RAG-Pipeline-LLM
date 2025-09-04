
import argparse
from src.training import train_llm, train_unsloth, merge_lora
import config

def main():
    parser = argparse.ArgumentParser(description="Model training and merging CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Base arguments for training ---
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--dataset_path", type=str, required=True, help="Path to the .jsonl dataset file.")
    parent_parser.add_argument(
        "--mode", type=str, default='sft', choices=['sft', 'unsupervised'],
        help="Training mode: 'sft' for supervised fine-tuning, 'unsupervised' for continued pre-training."
    )

    # --- Sub-parser for train_llm ---
    parser_train = subparsers.add_parser("train", parents=[parent_parser], help="Fine-tune the model using the standard method.")
    # Add any specific arguments for train_llm here if needed in the future

    # --- Sub-parser for train_unsloth ---
    parser_unsloth = subparsers.add_parser("train-unsloth", parents=[parent_parser], help="Fine-tune the model using Unsloth for optimization.")
    # Add any specific arguments for train_unsloth here if needed in the future

    # --- Sub-parser for merge_lora ---
    parser_merge = subparsers.add_parser("merge", help="Merge a LoRA adapter into the base model.")
    parser_merge.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA adapter directory.")
    parser_merge.add_argument("--output_dir", type=str, required=True, help="Directory to save the new, merged model.")
    parser_merge.add_argument("--base_model_path", type=str, default=config.LOCAL_MODEL_ID, help=f"Base model to merge into. Defaults to {config.LOCAL_MODEL_ID}")


    args = parser.parse_args()

    if args.command == "train":
        print("Running standard training...")
        train_llm.main(args)
    elif args.command == "train-unsloth":
        print("Running Unsloth training...")
        train_unsloth.main(args)
    elif args.command == "merge":
        print("Merging LoRA adapter...")
        merge_lora.main(args)

if __name__ == "__main__":
    main()
