
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import config

def merge_lora_to_base_model(base_model_path: str, lora_path: str, output_dir: str):
    """Merges a LoRA adapter into a base model and saves the merged model."""

    print(f"Loading base model from: {base_model_path}")
    # Load the base model in full precision or bfloat16 for stable merging
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu", # Load on CPU to avoid OOM issues during merge
    )

    print(f"Loading tokenizer from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print(f"Loading LoRA adapter from: {lora_path}")
    # Load the LoRA model
    model_to_merge = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging LoRA weights...")
    # Merge the LoRA weights into the base model. This returns a new model.
    merged_model = model_to_merge.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)

    print(f"Saving tokenizer to: {output_dir}")
    tokenizer.save_pretrained(output_dir)

    print("\n✅ LoRA merge complete.")
    print(f"New model saved at: {output_dir}")

def main(args):
    merge_lora_to_base_model(args.base_model_path, args.lora_path, args.output_dir)
