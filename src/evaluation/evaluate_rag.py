
import os
import json
import argparse
import torch
from tqdm import tqdm
import pandas as pd
import evaluate
import typing as tp

# --- Import RAG components --- 
# Note: To avoid code duplication, these functions could be moved to a shared utils file.
from src.rag.rag_pipeline import load_llm, setup_retriever, create_rag_chain
import config

def parse_qa_from_text(text: str) -> tp.Optional[tp.Dict[str, str]]:
    """Parses the question and answer from the Llama-3 formatted text."""
    try:
        user_start_tag = "<|start_header_id|>user<|end_header_id|>\n\n"
        assistant_start_tag = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        end_tag = "<|eot_id|>"

        user_start_index = text.find(user_start_tag)
        if user_start_index == -1:
            return None
        
        question_start = user_start_index + len(user_start_tag)
        question_end = text.find(end_tag, question_start)
        if question_end == -1:
            return None
        question = text[question_start:question_end].strip()

        assistant_start_index = text.find(assistant_start_tag, question_end)
        if assistant_start_index == -1:
            return None

        answer_start = assistant_start_index + len(assistant_start_tag)
        answer_end = text.find(end_tag, answer_start)
        if answer_end == -1:
            return None
        answer = text[answer_start:answer_end].strip()

        if not question or not answer:
            return None

        return {"question": question, "reference_answer": answer}
    except Exception:
        return None

def run_evaluation(
    model_type: str,
    lora_path: str,
    knowledge_base: str,
    eval_dataset_path: str,
    num_samples: int,
    model_id: str
) -> None:
    """Runs the full RAG evaluation process."""

    # 1. Load evaluation dataset
    print(f"Loading evaluation dataset from: {eval_dataset_path}")
    eval_data = []
    with open(eval_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            parsed_qa = parse_qa_from_text(data['text'])
            if parsed_qa:
                eval_data.append(parsed_qa)
    
    if not eval_data:
        print("Could not find any valid Q&A pairs in the evaluation file.")
        return

    # Limit number of samples if requested
    if num_samples > 0:
        eval_data = eval_data[:num_samples]
    print(f"Loaded {len(eval_data)} samples for evaluation.")

    # 2. Load the RAG chain
    print("\n--- Loading RAG Pipeline ---")
    llm = load_llm(model_type, model_id, lora_path)
    retriever = setup_retriever(knowledge_base)
    rag_chain = create_rag_chain(llm, retriever)
    print("--- RAG Pipeline Loaded ---\n")

    # 3. Run inference and collect results
    predictions = []
    references = []
    
    for item in tqdm(eval_data, desc="Running RAG Evaluation"):
        question = item['question']
        reference_answer = item['reference_answer']
        
        # Get generated answer from RAG chain
        generated_answer = rag_chain.invoke(question)
        
        predictions.append(generated_answer)
        references.append(reference_answer)

    # 4. Calculate metrics
    print("\n--- Calculating Metrics ---")
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')

    rouge_results = rouge.compute(predictions=predictions, references=references)
    bleu_results = bleu.compute(predictions=predictions, references=references)

    print("\n--- Evaluation Results ---")
    print(f"Evaluated on {len(predictions)} samples.")
    
    print("\nROUGE Scores:")
    for key, value in rouge_results.items():
        print(f"  {key}: {value:.4f}")
        
    print("\nBLEU Score:")
    print(f"  BLEU: {bleu_results['bleu']:.4f}")

    # 5. Save results to a file for inspection
    results_df = pd.DataFrame({
        'question': [item['question'] for item in eval_data],
        'reference_answer': references,
        'generated_answer': predictions
    })
    
    results_filename = os.path.join(config.DEFAULT_OUTPUT_DIR, "evaluation_results.csv")
    os.makedirs(config.DEFAULT_OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(results_filename, index=False, encoding='utf-8-sig')
    print(f"\nDetailed results saved to: {results_filename}")

    # 6. Save metrics to a file
    metrics_summary = {
        "model_type": model_type,
        "knowledge_base": knowledge_base,
        "eval_dataset_path": eval_dataset_path,
        "num_samples": len(predictions),
        "rouge": rouge_results,
        "bleu": bleu_results
    }
    metrics_filename = os.path.join(config.DEFAULT_OUTPUT_DIR, "evaluation_metrics.json")
    with open(metrics_filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=4)
    print(f"Metrics summary saved to: {metrics_filename}")

def main(model_type: str, lora_path: str, knowledge_base: str, eval_dataset_path: str, num_samples: int, model_id: str = config.LOCAL_MODEL_ID):
    run_evaluation(
        model_type=model_type,
        lora_path=lora_path,
        knowledge_base=knowledge_base,
        eval_dataset_path=eval_dataset_path,
        num_samples=num_samples,
        model_id=model_id
    )
