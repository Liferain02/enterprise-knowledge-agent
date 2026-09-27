"""Shared, model-free baseline for online uploads and corpus synchronization."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

INGESTION_SIGNATURE = "split1200-overlap150-v1"


def build_ingestion_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=150,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )


def split_ingestion_documents(documents):
    chunks = build_ingestion_splitter().split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not chunks:
        raise ValueError("文件未解析出有效文本，请检查文件或先进行 OCR")
    for index, chunk in enumerate(chunks):
        chunk.metadata.update(chunk_index=index, ingestion_signature=INGESTION_SIGNATURE)
    return chunks
