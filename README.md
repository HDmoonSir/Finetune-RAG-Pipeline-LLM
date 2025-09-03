# LLM Finetune & RAG Pipeline

This project provides a set of Python scripts to build a custom dataset from PDF files, fine-tune a Large Language Model (LLM), and run a Retrieval-Augmented Generation (RAG) pipeline.

## Scripts and Workflow

The process is divided into the following stages:

### 1. Data Preparation

#### `extract_pdf_text.py`

Extracts text from PDF files and saves it in JSONL format. Each line in the output file represents a page from the PDF.

**Usage:**
```bash
python extract_pdf_text.py --file_path /path/to/your/document.pdf --start_page 1 --end_page 50
```
- `--file_path`: Path to the PDF file.
- `--start_page` (optional): The first page to extract (default: 1).
- `--end_page` (optional): The last page to extract (default: end of the document).
- `--output_file` (optional): Name for the output `.jsonl` file.

#### `preprocess_with_gemini.py`

Uses the extracted text to generate a training dataset with the Gemini API. It supports two modes:
- `qa`: Generates question-answer pairs formatted for instruction fine-tuning (Llama 3 format).
- `unsupervised`: Generates coherent, summarized text for unsupervised fine-tuning.

**Prerequisites:**
You must set your Gemini API key as an environment variable:
```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

**Usage:**
```bash
# For Question-Answer dataset generation
python preprocess_with_gemini.py --mode qa

# For unsupervised dataset generation
python preprocess_with_gemini.py --mode unsupervised
```
This script will process all `.jsonl` files found in the `data/` directory and create either `gemini_generated_qa_dataset.jsonl` or `gemini_generated_unsupervised_dataset.jsonl`.

### 2. Model Fine-tuning

Two scripts are provided for fine-tuning the `MLP-KTLim/llama-3-Korean-Bllossom-8B` model.

#### `train_llm.py` (Standard)

Fine-tunes the model using LoRA.

**Usage:**
```bash
python train_llm.py --dataset_path /path/to/your_dataset.jsonl
```
- `--dataset_path`: Path to the generated `.jsonl` dataset file.

#### `train_unsloth.py` (Optimized)

A memory-efficient and faster version of the training script using the Unsloth library.

**Usage:**
```bash
python train_unsloth.py --dataset_path /path/to/your_dataset.jsonl
```
- `--dataset_path`: Path to the generated `.jsonl` dataset file.

### 3. RAG Inference

#### `rag_pipeline.py`

An integrated script to run a Retrieval-Augmented Generation (RAG) pipeline. It uses the KorQuAD dataset as its knowledge base and supports multiple model-loading strategies for inference.

**Usage:**
```bash
python rag_pipeline.py {model_type} "your_question"
```

**Positional Arguments:**

- `model_type`: The type of model to use for inference.
  - `api`: Uses the Gemini API (`gemini-1.5-flash-latest`). Requires the `GEMINI_API_KEY` environment variable to be set.
  - `local`: Uses the full-precision local model (`MLP-KTLim/llama-3-Korean-Bllossom-8B`). This is recommended for GPUs like the Titan V (Volta architecture) where Unsloth may have compatibility issues.
  - `local-quantized`: Uses a 4-bit quantized version of the local model via Unsloth for faster inference and lower memory usage. Recommended for compatible NVIDIA GPUs (Ampere, Turing, Ada, etc.).

- `your_question`: The question you want to ask the RAG system. Must be enclosed in double quotes.

- `--lora-path`: Path to the LoRA adapter (optional). Only applicable for local models.

**Execution Examples:**

1.  **API Model Test:**
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY"
    python rag_pipeline.py api "파우스트의 작가는 누구인가?"
    ```

2.  **Local Model Test (Non-Quantized for Volta/Compatibility):**
    ```bash
    python rag_pipeline.py local "임진왜란이 발발한 연도는 언제야?"
    ```

3.  **Local Model Test (Quantized for compatible GPUs):**
    ```bash
    python rag_pipeline.py local-quantized "세종대왕이 한글을 창제한 연도는?"
    ```

#### `rag_interactive_cli.py`

An interactive command-line interface for the RAG pipeline. This script loads the model and knowledge base once at startup, allowing for continuous questioning without reloading.

**Usage:**
```bash
python rag_interactive_cli.py {model_type}
```

**Arguments:**
- `model_type`: The type of model to use. Same choices as `rag_pipeline.py`: `api`, `local`, `local-quantized`.

- `--lora-path`: Path to the LoRA adapter (optional). Only applicable for local models.

**Interaction:**
After starting, the script will prompt you for questions. Type your question and press Enter. To exit, type `exit` or `quit` and press Enter.

**Examples:**

1.  **API Model (Interactive):**
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY"
    python rag_interactive_cli.py api
    ```
    (Then type questions at the prompt)

2.  **Local Model (Non-Quantized, Interactive):**
    ```bash
    python rag_interactive_cli.py local
    ```
    (Then type questions at the prompt)

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
3.  **Prepare data:**
    - Place your PDF files in a directory (e.g., `data/raw_pdfs/`).
    - Run `extract_pdf_text.py` to convert them to `.jsonl`.
4.  **Generate dataset:**
    - Set the `GEMINI_API_KEY`.
    - Run `preprocess_with_gemini.py` to create your training data.
5.  **Train the model:**
    - Run either `train_llm.py` or `train_unsloth.py` with the path to your generated dataset.

## License

This project utilizes Llama 3, which is available under the Llama 3 Community License. Please see the [LICENSE](LICENSE) file for the full license text.
