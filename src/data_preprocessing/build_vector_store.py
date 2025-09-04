
import glob
import json
import os
import typing as tp

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm


def build_vector_store(
    input_dir: str,
    vector_store_path: str,
    embedding_model_id: str,
    text_splitter_chunk_size: int,
    text_splitter_chunk_overlap: int,
) -> None:
    """Builds a FAISS vector store from JSONL files in the input directory."""

    print(f"🔍 Reading JSONL files from: {input_dir}")
    jsonl_paths = sorted(
        glob.glob(os.path.join(input_dir, "**/*.jsonl"), recursive=True) +
        glob.glob(os.path.join(input_dir, "*.jsonl"))
    )
    
    if not jsonl_paths:
        print("No JSONL files found to process.")
        return

    documents = list()
    for path in tqdm(jsonl_paths, desc="Reading files"):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get('text')
                    if text:
                        metadata = {"source": os.path.basename(path)}
                        documents.append(Document(page_content=text, metadata=metadata))
                except json.JSONDecodeError:
                    print(f"⚠️ Warning: Skipping a line in {path} due to JSON decoding error.")

    if not documents:
        print("No text content found in the JSONL files.")
        return

    print(f"Found {len(documents)} documents.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=text_splitter_chunk_size,
        chunk_overlap=text_splitter_chunk_overlap
    )
    splits = text_splitter.split_documents(documents)
    print(f"Documents split into {len(splits)} chunks.")

    print(f"Initializing embedding model: {embedding_model_id}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)

    print("Building FAISS vector store... (This may take a while)")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
    vectorstore.save_local(vector_store_path)

    print(f"✅ Vector store successfully built and saved to: {vector_store_path}")

def main(
    input_dir: str,
    vector_store_path: str,
    embedding_model_id: str,
    text_splitter_chunk_size: int,
    text_splitter_chunk_overlap: int,
) -> None:
    build_vector_store(
        input_dir=input_dir,
        vector_store_path=vector_store_path,
        embedding_model_id=embedding_model_id,
        text_splitter_chunk_size=text_splitter_chunk_size,
        text_splitter_chunk_overlap=text_splitter_chunk_overlap,
    )
