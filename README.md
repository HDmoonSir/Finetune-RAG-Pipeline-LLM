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
- `config/`: Directory containing YAML configuration files for data processing, training, inference, and evaluation.

## General Workflow with Configuration Files

This project now utilizes a YAML-based configuration system for all major workflows, enhancing reproducibility and simplifying the management of complex parameters. Instead of passing numerous command-line arguments, you will primarily use the `--config` argument to specify a YAML configuration file.

### How to Use Configuration Files

1.  **Locate Configuration Files:** Configuration files are organized by task within the `config/` directory (e.g., `config/data/`, `config/train/`, `config/eval/`, `config/inference/`).
2.  **Select/Modify a Configuration:** Choose the appropriate YAML file for your task. You can modify its parameters directly to suit your needs.
3.  **Run the Script with `--config`:** Pass the path to your chosen YAML file using the `--config` argument.

**General Usage Example:**

```bash
python3.11 <runner_script.py> --config <path_to_yaml_config>
```

For example, to run a training job, you would use:

```bash
python3.11 run_training.py --config config/train/llama-8b-sft.yaml
```

This approach ensures that all parameters for a specific run are encapsulated within a single, version-controlled file.

## Workflow and Usage

The process is divided into the following stages. All scripts should be run from the project's root directory.

### 1. Data Preparation

Data preparation tasks are configured via YAML files in `config/data/`.

#### a. Extract Text from PDF
- **Script:** `run_data.py`
- **Description:** Extracts text from PDF files and saves it in JSONL format. Configuration for this task (e.g., input/output paths, page ranges) is defined in `config/data/extract_pdf_text.yaml` (example).
- **Usage:**
  ```bash
  python3.11 run_data.py --config config/data/extract_pdf_text.yaml
  ```

#### b. Generate Training Data
- **Script:** `run_data.py`
- **Description:** Uses extracted text to generate training datasets (QA or unsupervised) with the Gemini API. Configuration for this task (e.g., mode, dataset paths, prompt templates) is defined in `config/data/preprocess_gemini.yaml`.
- **Prerequisites:** `export GEMINI_API_KEY="YOUR_API_KEY"`
- **Usage:**
  ```bash
  python3.11 run_data.py --config config/data/preprocess_gemini.yaml
  ```

### 2. Model Fine-tuning

The fine-tuning process is designed as a two-stage workflow to first adapt the model to a specific domain (unsupervised) and then teach it specific tasks (supervised).

#### a. Stage 1: Unsupervised Domain Adaptation
- **Goal:** Adapt the base LLM to your specific domain of knowledge.
- **Script:** `run_training.py`
- **Configuration:** Use a YAML file where `training.mode` is set to `unsupervised`.
- **Description:** This step fine-tunes the model on a large corpus of your domain-specific text. It produces an `unsupervised_lora_adapter` which captures the nuances of your domain.
- **Usage:**
  ```bash
  # This will produce a LoRA adapter in the output directory
  python3.11 run_training.py --config config/train/llama-8b-unsupervised.yaml # (Example config)
  ```

#### b. Stage 2: Supervised Fine-Tuning (SFT)
- **Goal:** Teach the domain-adapted model to follow instructions or perform specific tasks (e.g., Q&A).
- **Script:** `run_training.py`
- **Configuration:** Use a YAML file where `training.mode` is set to `sft`. Crucially, you can specify the path to the adapter from Stage 1 in the `model.unsupervised_lora_path` field.
- **Description:** This step performs a "just-in-time" merge. It loads the base model, merges the unsupervised LoRA from Stage 1 in-memory (if provided), and then trains a new LoRA adapter on top of it using your supervised dataset. This produces an `sft_lora_adapter`.
- **Usage:**
  ```bash
  # Assumes you have a config with `unsupervised_lora_path` pointing to the Stage 1 adapter
  python3.11 run_training.py --config config/train/llama-8b-sft.yaml
  ```

### 3. RAG Pipeline and Evaluation

The RAG and Evaluation pipelines use the same flexible model loading mechanism, allowing you to combine the effects of both unsupervised and supervised LoRA adapters at runtime.

#### a. Build Vector Store
- **Script:** `run_rag.py`
- **Description:** Builds a FAISS vector store from your source documents for the RAG pipeline. Configuration for this task (e.g., input directory, vector store path, embedding model) is defined in `config/data/build_vector_store.yaml`.
- **Usage:**
  ```bash
  python3.11 run_rag.py --config config/data/build_vector_store.yaml
  ```

#### b. Run RAG Inference or Evaluation
- **Scripts:** `run_rag.py`, `run_evaluation.py`
- **Description:** Runs the RAG pipeline or evaluates it. The key is the model configuration in your YAML file (`config/inference/local_rag.yaml` or `config/eval/default_eval.yaml`). The pipeline performs a sequential, in-memory merge of the specified LoRA adapters.
- **Configuration:**
  - `model.model_id`: The original base model (e.g., `MLP-KTLim/llama-3-Korean-Bllossom-8B`).
  - `model.unsupervised_lora_path`: (Optional) Path to the LoRA from unsupervised training.
  - `model.sft_lora_path`: (Optional) Path to the LoRA from supervised fine-tuning.
  You can provide one, both, or neither of the LoRA paths. The pipeline will load the base model and sequentially merge any provided adapters in memory before running inference.
- **Usage (RAG CLI Example):**
  ```bash
  # This config can specify both unsupervised and sft LoRA paths
  python3.11 run_rag.py --config config/inference/local_rag.yaml
  ```
- **Usage (Evaluation Example):**
  ```bash
  # This config can also specify both LoRA paths for evaluation
  python3.11 run_evaluation.py --config config/eval/default_eval.yaml
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