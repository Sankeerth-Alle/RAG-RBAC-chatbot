"""Build the Chroma collection from the repository's source data.

Run from the repository root or backend directory:
    uv run python build_vector_db.py --reset
"""

import argparse
import csv
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.config import Config, get_vector_db_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "training" / "data"


def markdown_documents(path: Path, access_level: str) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    )
    documents = splitter.split_text(path.read_text(encoding="utf-8"))
    source = path.relative_to(PROJECT_ROOT).as_posix()
    for document in documents:
        document.metadata.update({
            "access_level": access_level,
            "source_file": path.name,
            "source": source,
        })
    return documents


def csv_documents(path: Path) -> list[Document]:
    source = path.relative_to(PROJECT_ROOT).as_posix()
    documents = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            content = "\n".join(f"{key}: {value}" for key, value in row.items())
            documents.append(Document(
                page_content=content,
                metadata={
                    "access_level": "hr",
                    "source_file": path.name,
                    "source": source,
                },
            ))
    return documents


def load_documents(data_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() == ".md":
            documents.extend(markdown_documents(path, path.parent.name.lower()))
        elif path.suffix.lower() == ".csv":
            documents.extend(csv_documents(path))
    return documents


def build_vector_db(data_dir: Path, db_path: Path, reset: bool) -> int:
    documents = load_documents(data_dir)
    if not documents:
        raise RuntimeError(f"No .md or .csv source documents found in {data_dir}")

    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=Config.VECTOR_DB_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(db_path),
        collection_metadata={"embedding_model": Config.EMBEDDING_MODEL},
    )
    if reset:
        try:
            vector_store._client.delete_collection(Config.VECTOR_DB_COLLECTION_NAME)
        except Exception:
            pass
        vector_store = Chroma(
            collection_name=Config.VECTOR_DB_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(db_path),
            collection_metadata={"embedding_model": Config.EMBEDDING_MODEL},
        )

    vector_store.add_documents(documents)
    print(f"Indexed documents: {len(documents)}")
    print(f"Embedding model: {Config.EMBEDDING_MODEL}")
    print(f"Chroma path: {db_path}")
    return len(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the RBAC Chroma vector database")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--db-path", type=Path, default=get_vector_db_path())
    parser.add_argument("--reset", action="store_true", help="Replace the existing collection")
    args = parser.parse_args()
    build_vector_db(args.data_dir.resolve(), args.db_path.resolve(), args.reset)