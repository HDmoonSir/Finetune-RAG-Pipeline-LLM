import os
import fitz  # PyMuPDF
import json
import argparse
from tqdm import tqdm
import typing as tp


def process_pdf_pages(
    pdf_path: str,
    start_page: int,
    end_page: tp.Optional[int],
    output_file: tp.Optional[str] = None,
) -> None:
    """
    Extracts text from a specified page range of a single PDF file and saves it to a .jsonl file.
    """
    try:
        doc: fitz.Document = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file '{pdf_path}': {e}")
        return

    # Validate page range
    total_pages: int = doc.page_count
    start_page = max(1, start_page)
    if end_page is None or end_page > total_pages:
        end_page = total_pages

    if start_page > end_page:
        print(
            f"Error: Start page ({start_page}) cannot be greater than end page ({end_page})."
        )
        doc.close()
        return

    # Determine output filename
    if output_file is None:
        base_name: str = os.path.basename(pdf_path)
        file_name_without_ext: str = os.path.splitext(base_name)[0]
        output_file = f"{file_name_without_ext}_pages_{start_page}-{end_page}.jsonl"

    print(
        f"Processing pages {start_page} to {end_page} from '{pdf_path}' -> '{output_file}'"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        # Loop through the specified page range (0-indexed for fitz)
        for i in tqdm(range(start_page - 1, end_page), desc="Processing Pages"):
            page: fitz.Page = doc.load_page(i)
            text: str = page.get_text("text")
            if text.strip():  # Only write lines for pages with content
                data: tp.Dict[str, tp.Union[str, int]] = {
                    "source_pdf": pdf_path,
                    "page_number": i + 1,
                    "text": text.strip(),
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    doc.close()
    print(f"\n✅ Successfully created '{output_file}'.")
