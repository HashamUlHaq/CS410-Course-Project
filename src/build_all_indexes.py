# src/build_all_indexes.py

from pathlib import Path

from .data_utils import load_corpus
from .retrievers.build_bm25_index import build_bm25_index
from .retrievers.dense_retriever import DenseRetriever
from .config import DATA_DIR, FAISS_INDEX_PATH


def main():
    print("=== DocuSherlock Index Builder ===")

    # 1. Load corpus (mtsamples.csv)
    print("[1/3] Loading corpus from mtsamples.csv ...")
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents.")

    # 2. Build / rebuild BM25 Lucene index
    print("[2/3] Building BM25/BM25+RM3 Lucene index ...")
    build_bm25_index()
    print("BM25 index built under:", DATA_DIR / "bm25_index")

    # 3. Build / rebuild Dense (BioBERT + FAISS) index
    print("[3/3] Building dense (BioBERT + FAISS) index ...")
    DenseRetriever.build(docs)
    print("Dense FAISS index written to:", FAISS_INDEX_PATH)

    print("=== All indexes built successfully. ===")


if __name__ == "__main__":
    main()