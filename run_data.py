import argparse
from src.data_preprocessing import extract_pdf_text, preprocess_with_gemini, build_vector_store
from src.utils.config_loader import load_data_preprocessing_config, load_vector_store_build_config, DataPreprocessingConfig, VectorStoreBuildConfig

def main():
    parser = argparse.ArgumentParser(description="Data processing CLI for the project.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Sub-parser for PDF Extraction ---
    parser_extract = subparsers.add_parser("extract-pdf", help="Extract text from a PDF file.")
    parser_extract.add_argument(
        '--file_path', type=str, required=True, help="Path to the PDF file to process."
    )
    parser_extract.add_argument(
        '--start_page', type=int, default=1, help="The first page to process (1-based). Defaults to 1."
    )
    parser_extract.add_argument(
        '--end_page', type=int, default=None, help="The last page to process (inclusive). Defaults to the last page."
    )
    parser_extract.add_argument(
        '--output_file', type=str, default=None, help="Optional: Specify a name for the output .jsonl file."
    )

    # --- Sub-parser for Gemini Preprocessing ---
    parser_preprocess = subparsers.add_parser("preprocess-gemini", help="Generate a dataset using the Gemini API.")
    parser_preprocess.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the data preprocessing configuration YAML file (e.g., configs/data/preprocess.yaml)"
    )

    # --- Sub-parser for Build Vector Store ---
    parser_build_db = subparsers.add_parser("build-db", help="Build a FAISS vector store.")
    parser_build_db.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the vector store build configuration YAML file (e.g., configs/data/build_vector_store.yaml)"
    )

    args = parser.parse_args()

    if args.command == "extract-pdf":
        print("Running PDF text extraction...")
        extract_pdf_text.process_pdf_pages(
            pdf_path=args.file_path, 
            start_page=args.start_page, 
            end_page=args.end_page, 
            output_file=args.output_file
        )
    elif args.command == "preprocess-gemini":
        print("Running Gemini preprocessing...")
        config: DataPreprocessingConfig = load_data_preprocessing_config(args.config)
        preprocess_with_gemini.main(
            mode=config.mode,
            gemini_preprocess_model=config.gemini_preprocess_model,
            data_dir=config.data_dir,
            text_page_batch_size=config.text_page_batch_size,
            qa_prompt_template=config.qa_prompt_template,
            unsupervised_prompt_template=config.unsupervised_prompt_template,
            qa_dataset_path=config.qa_dataset_path,
            unsupervised_dataset_path=config.unsupervised_dataset_path,
            default_output_dir=config.default_output_dir,
        )
    elif args.command == "build-db":
        print("Building vector store...")
        config: VectorStoreBuildConfig = load_vector_store_build_config(args.config)
        build_vector_store.main(
            input_dir=config.input_dir,
            vector_store_path=config.vector_store_path,
            embedding_model_id=config.embedding_model_id,
            text_splitter_chunk_size=config.text_splitter_chunk_size,
            text_splitter_chunk_overlap=config.text_splitter_chunk_overlap,
        )

if __name__ == "__main__":
    main()