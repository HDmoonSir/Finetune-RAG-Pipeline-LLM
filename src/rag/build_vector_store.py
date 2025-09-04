
import os
import json
import glob
import argparse
from tqdm import tqdm
import typing as tp

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config

def build_vector_store(input_dir: str, vector_store_path: str) -> None:
    """Builds a FAISS vector store from JSONL files in the input directory."""

    print(f"🔍 Reading JSONL files from: {input_dir}")
    jsonl_paths = sorted(
        glob.glob(os.path.join(input_dir, "**/*.jsonl"), recursive=True) +
        glob.glob(os.path.join(input_dir, "*.jsonl"))
    )
    
    if not jsonl_paths:
        print("No JSONL files found to process.")
        return

    documents = []
    for path in tqdm(jsonl_paths, desc="Reading files"):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get('text')
                    if text:
                        # We can add metadata here if needed, e.g., from the source file
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
        chunk_size=config.TEXT_SPLITTER_CHUNK_SIZE,
        chunk_overlap=config.TEXT_SPLITTER_CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(documents)
    print(f"Documents split into {len(splits)} chunks.")

    print(f"Initializing embedding model: {config.EMBEDDING_MODEL_ID}")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_ID)

    print("Building FAISS vector store... (This may take a while)")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
    vectorstore.save_local(vector_store_path)

    print(f"✅ Vector store successfully built and saved to: {vector_store_path}")

def main(input_dir: str, vector_store_path: str) -> None:
    build_vector_store(input_dir, vector_store_path)
