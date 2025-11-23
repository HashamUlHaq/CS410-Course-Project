# src/retrievers/bm25_retriever.py

from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path

from pyserini.search.lucene import LuceneSearcher
from ..config import DATA_DIR

INDEX_DIR = DATA_DIR / "bm25_index"


@dataclass
class BM25Retriever:
    searcher: LuceneSearcher

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "BM25Retriever":
        # For a local index, pass the directory path
        searcher = LuceneSearcher(str(index_dir))
        # You can tweak BM25 parameters here if needed
        searcher.set_bm25(k1=0.9, b=0.4)
        return cls(searcher)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        hits = self.searcher.search(query, k=top_k)
        return [(h.docid, float(h.score)) for h in hits]


@dataclass
class BM25RM3Retriever:
    searcher: LuceneSearcher

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "BM25RM3Retriever":
        searcher = LuceneSearcher(str(index_dir))
        searcher.set_bm25(k1=0.9, b=0.4)
        # RM3 query expansion
        searcher.set_rm3()   # you can pass params if you want: fb_docs, fb_terms, original_query_weight
        return cls(searcher)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        hits = self.searcher.search(query, k=top_k)
        return [(h.docid, float(h.score)) for h in hits]