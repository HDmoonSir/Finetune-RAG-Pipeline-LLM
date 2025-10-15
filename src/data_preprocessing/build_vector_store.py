import glob
import json
import os
import typing as tp

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm


from src.utils.config_loader import VectorStoreBuildConfig

def build_vector_store(cfg: VectorStoreBuildConfig) -> None:
    """Builds a FAISS vector store from JSONL files in the input directory."""

    print(f"🔍 Reading JSONL files from: {cfg.input_dir}")
    jsonl_paths = sorted(
        glob.glob(os.path.join(cfg.input_dir, "**/*.jsonl"), recursive=True)
        + glob.glob(os.path.join(cfg.input_dir, "*.jsonl"))
    )

    if not jsonl_paths:
        print("No JSONL files found to process.")
        return

    documents = list()
    for path in tqdm(jsonl_paths, desc="Reading files"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text")
                    if text:
                        metadata = {"source": os.path.basename(path)}
                        documents.append(Document(page_content=text, metadata=metadata))
                except json.JSONDecodeError:
                    print(
                        f"⚠️ Warning: Skipping a line in {path} due to JSON decoding error."
                    )

    if not documents:
        print("No text content found in the JSONL files.")
        return

    print(f"Found {len(documents)} documents.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.text_splitter_chunk_size, chunk_overlap=cfg.text_splitter_chunk_overlap
    )
    splits = text_splitter.split_documents(documents)
    print(f"Documents split into {len(splits)} chunks.")

    print(f"Initializing embedding model: {cfg.embedding_model_id}")
    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model_id)

    print("Building FAISS vector store... (This may take a while)")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    os.makedirs(os.path.dirname(cfg.vector_store_path), exist_ok=True)
    vectorstore.save_local(cfg.vector_store_path)

    print(f"✅ Vector store successfully built and saved to: {cfg.vector_store_path}")


def main(cfg: VectorStoreBuildConfig) -> None:
    build_vector_store(cfg=cfg)
