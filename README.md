# LLM Fine-tuning Pipeline

This project provides a set of Python scripts to build a custom dataset from PDF files and fine-tune a Large Language Model (LLM) using that data. The pipeline consists of three main stages: PDF text extraction, data preprocessing and generation using the Gemini API, and model training.

## Scripts and Workflow

The process is divided into the following scripts, which should be run in order:

### 1. `extract_pdf_text.py`

Extracts text from PDF files and saves it in JSONL format. Each line in the output file represents a page from the PDF.

**Usage:**
```bash
python extract_pdf_text.py --file_path /path/to/your/document.pdf --start_page 1 --end_page 50
```
- `--file_path`: Path to the PDF file.
- `--start_page` (optional): The first page to extract (default: 1).
- `--end_page` (optional): The last page to extract (default: end of the document).
- `--output_file` (optional): Name for the output `.jsonl` file.

### 2. `preprocess_with_gemini.py`

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

### 3. Model Training

Two scripts are provided for fine-tuning the model.

#### `train_llm.py` (Standard)

Fine-tunes the `MLP-KTLim/llama-3-Korean-Bllossom-8B` model using LoRA.

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

