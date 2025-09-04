# LLM Finetune & RAG Pipeline

This project provides a set of Python scripts to build a custom dataset from PDF files, fine-tune a Large Language Model (LLM), and run a Retrieval-Augmented Generation (RAG) pipeline.

## Project Structure

The project is organized into the following directories:

- `src/`: Contains all Python source code.
  - `data_processing/`: Scripts for data extraction and preprocessing.
  - `training/`: Scripts for model fine-tuning and LoRA merging.
  - `rag/`: Scripts for the RAG pipeline, vector store building, and inference.
  - `evaluation/`: Scripts for evaluating the RAG pipeline.
- `run_data.py`: Runner script for data processing tasks.
- `run_training.py`: Runner script for model training and merging tasks.
- `run_rag.py`: Runner script for RAG pipeline tasks.
- `run_evaluation.py`: Runner script for evaluation tasks.
- `data/`: Directory for raw data (e.g., PDFs, source JSONL files).
- `data_result/`: Directory for generated outputs (datasets, vector stores, models, evaluation results).
- `config.py`: Main configuration file for the project.

## Workflow and Usage

The process is divided into the following stages. All scripts should be run from the project's root directory.

### 1. Data Preparation

#### a. Extract Text from PDF
- **Script:** `run_data.py extract-pdf`
- **Description:** Extracts text from PDF files and saves it in JSONL format.
- **Usage:**
  ```bash
  python3.11 run_data.py extract-pdf --file_path /path/to/your/document.pdf
  ```
- **Arguments:**
  - `--file_path`: Path to the PDF file.
  - `--start_page` (optional): The first page to extract (default: 1).
  - `--end_page` (optional): The last page to extract (default: end of the document).

#### b. Generate Training Data
- **Script:** `run_data.py preprocess-gemini`
- **Description:** Uses extracted text to generate training datasets with the Gemini API.
- **Prerequisites:** `export GEMINI_API_KEY="YOUR_API_KEY"`
- **Usage:**
  ```bash
  # For Question-Answer (SFT) dataset generation
  python3.11 run_data.py preprocess-gemini --mode qa

  # For unsupervised dataset generation
  python3.11 run_data.py preprocess-gemini --mode unsupervised
  ```

### 2. Model Fine-tuning

#### a. Fine-tune the Model
- **Scripts:** `run_training.py train` (Standard) or `run_training.py train-unsloth` (Optimized)
- **Description:** Fine-tunes the model using LoRA. Supports both supervised (SFT) and unsupervised modes.
- **Usage:**
  ```bash
  # For Supervised Fine-Tuning (SFT)
  python3.11 run_training.py train-unsloth --dataset_path <path_to_qa_dataset> --mode sft

  # For Unsupervised Continued Pre-training
  python3.11 run_training.py train-unsloth --dataset_path <path_to_unsupervised_dataset> --mode unsupervised
  ```

#### b. Merge LoRA Adapter
- **Script:** `run_training.py merge`
- **Description:** Merges a trained LoRA adapter into the base model to create a new, standalone model. This is useful for creating a domain-adapted base model after unsupervised tuning.
- **Usage:**
  ```bash
  python3.11 run_training.py merge --lora_path <path_to_lora_adapter> --output_dir <path_to_save_merged_model>
  ```

### 3. RAG Pipeline

#### a. Build Vector Store
- **Script:** `run_rag.py build-db`
- **Description:** Builds a FAISS vector store from your source documents for the RAG pipeline.
- **Usage:**
  ```bash
  python3.11 run_rag.py build-db --input_dir data/
  ```

#### b. Run RAG Inference
- **Scripts:** `run_rag.py pipeline` (single question) or `run_rag.py cli` (interactive chat)
- **Description:** Runs the RAG pipeline using a specified model and knowledge base.
- **Usage (Interactive CLI):**
  ```bash
  # Using API model and custom vector store
  export GEMINI_API_KEY="YOUR_API_KEY"
  python3.11 run_rag.py cli --model_type api --knowledge_base data_result/vector_store

  # Using local model and custom vector store
  python3.11 run_rag.py cli --model_type local --knowledge_base data_result/vector_store
  ```
- **Arguments:**
  - `--model_type`: `api`, `local`, `local-quantized`.
  - `--lora-path`: Path to the LoRA adapter (optional).
  - `--knowledge_base`: Path to a custom FAISS vector store, or `default` to use KorQuAD.

### 4. Evaluation

- **Script:** `run_evaluation.py`
- **Description:** Evaluates the RAG pipeline on a QA dataset and calculates ROUGE/BLEU scores.
- **Usage:**
  ```bash
  python3.11 run_evaluation.py --model_type api --knowledge_base data_result/vector_store --num_samples 20
  ```

## Project Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  Follow the workflow steps above to prepare data, train, and run the pipeline.

## License

This project utilizes Llama 3, which is available under the Llama 3 Community License. Please see the [LICENSE](LICENSE) file for the full license text.