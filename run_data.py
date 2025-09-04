
import argparse
from src.data_processing import extract_pdf_text, preprocess_with_gemini

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
        '--mode', type=str, default='qa', choices=['qa', 'unsupervised'],
        help='Generation mode: `qa` for question-answer pairs, `unsupervised` for plain text.'
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
        print(f"Running Gemini preprocessing in '{args.mode}' mode...")
        # The preprocess_with_gemini.main function is refactored to accept the mode directly
        preprocess_with_gemini.main(mode=args.mode)

if __name__ == "__main__":
    main()
