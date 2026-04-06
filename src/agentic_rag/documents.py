from __future__ import annotations

import os
from typing import Iterable, List, Sequence

os.environ.setdefault("USER_AGENT", "agentic-rag/0.1.0")

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(urls: Sequence[str]) -> List[Document]:
    """Fetch all source documents from the configured URLs."""
    documents = [WebBaseLoader(url).load() for url in urls]
    return [document for batch in documents for document in batch]


def split_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Split documents into smaller chunks for semantic search."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(documents))


def preprocess_documents(
    urls: Sequence[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Load and split source documents in one step."""
    documents = load_documents(urls)
    return split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
