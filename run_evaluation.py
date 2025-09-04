
import argparse
from src.evaluation import evaluate_rag
import config

def main():
    parser = argparse.ArgumentParser(description="Evaluation CLI for the project.")

    parser.add_argument(
        "--model_type", 
        type=str, 
        default="api",
        choices=["api", "local", "local-quantized"],
        help="Type of model to use for inference."
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to the LoRA adapter (optional). Only applicable for local models."
    )
    parser.add_argument(
        "--knowledge_base",
        type=str,
        default="default",
        help=f"Knowledge base to use. 'default' for KorQuAD, or a path to a custom FAISS vector store (e.g., {config.VECTOR_STORE_DIR})."
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default=config.QA_DATASET_PATH,
        help=f"Path to the QA dataset for evaluation. Defaults to {config.QA_DATASET_PATH}. "
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate from the dataset. Set to 0 to use all samples."
    )
    parser.add_argument(
        "--model_id", type=str, default=config.LOCAL_MODEL_ID,
        help=f"Path/ID to the base model. Defaults to {config.LOCAL_MODEL_ID}."
    )

    args = parser.parse_args()

    print("Running RAG evaluation...")
    evaluate_rag.main(
        model_type=args.model_type,
        lora_path=args.lora_path,
        knowledge_base=args.knowledge_base,
        eval_dataset_path=args.eval_dataset_path,
        num_samples=args.num_samples,
        model_id=args.model_id
    )

if __name__ == "__main__":
    main()
