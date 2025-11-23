# src/run_experiments.py

import json
from pathlib import Path
from typing import Dict, List, Tuple

from .data_utils import load_corpus
from .retrievers.tfidf_retriever import TfidfRetriever
from .retrievers.bm25_retriever import BM25Retriever, BM25RM3Retriever
from .retrievers.dense_retriever import DenseRetriever
from .retrievers.hybrid import rrf_fusion
from .eval_metrics import evaluate_run
from .config import QRELS_PATH, TOP_K

def load_qrels(path: Path = QRELS_PATH) -> Dict[str, Dict[str, float]]:
    """
    Expects qrels.jsonl like:
    {"query_id": "q1", "doc_id": "doc3", "rel": 1}
    """
    qrels: Dict[str, Dict[str, float]] = {}
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["query_id"]
            did = obj["doc_id"]
            rel = float(obj["rel"])
            qrels.setdefault(qid, {})[did] = rel
    return qrels

def main():
    docs = load_corpus()
    qrels = load_qrels()

    # You’ll also need a file mapping query_id -> query_text, e.g. queries.jsonl
    queries_path = QRELS_PATH.parent / "queries.jsonl"
    queries = {}
    with queries_path.open() as f:
        for line in f:
            obj = json.loads(line)
            queries[obj["query_id"]] = obj["query"]

    # Build / load retrievers
    tfidf = TfidfRetriever.build(docs)
    bm25 = BM25Retriever.load()
    bm25_rm3 = BM25RM3Retriever.load()
    dense = DenseRetriever.load()

    runs: Dict[str, Dict[str, List[Tuple[str, float]]]] = {
        "tfidf": {},
        "bm25": {},
        "bm25_rm3": {},
        "dense": {},
        "hybrid_bm25_dense": {},
    }

    for qid, qtext in queries.items():
        runs["tfidf"][qid] = tfidf.search(qtext, top_k=TOP_K)
        runs["bm25"][qid] = bm25.search(qtext, top_k=TOP_K)
        runs["bm25_rm3"][qid] = bm25_rm3.search(qtext, top_k=TOP_K)
        runs["dense"][qid] = dense.search(qtext, top_k=TOP_K)

        bm25_res = runs["bm25"][qid]
        dense_res = runs["dense"][qid]
        runs["hybrid_bm25_dense"][qid] = rrf_fusion(
            [bm25_res, dense_res],
            top_k=TOP_K,
        )

    for name, run in runs.items():
        metrics = evaluate_run(run, qrels)
        print(name, metrics)

if __name__ == "__main__":
    main()