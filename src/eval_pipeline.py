# src/eval_pipeline.py

import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from src.config import DATA_DIR
from src.data_utils import load_corpus
from src.eval_metrics import evaluate_run
from src.retrievers.tfidf_retriever import TfidfRetriever
from src.retrievers.bm25_retriever import BM25Retriever, BM25RM3Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.hybrid import rrf_fusion

QRELS_PATH = DATA_DIR / "qrels.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"


def load_qrels() -> Dict[str, Dict[str, float]]:
    """
    qrels[qid][doc_id] = rel
    """
    qrels: Dict[str, Dict[str, float]] = defaultdict(dict)
    with QRELS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["query_id"]
            doc_id = rec["doc_id"]
            rel = float(rec["rel"])
            qrels[qid][doc_id] = rel
    return qrels


def load_queries() -> Dict[str, str]:
    """
    {query_id: query_text}
    """
    qid_to_text: Dict[str, str] = {}
    with QUERIES_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["query_id"]
            qtext = rec["query"]
            qid_to_text[qid] = qtext
    return qid_to_text


def get_method_results_for_query(
    query_text: str,
    tfidf: TfidfRetriever,
    bm25: BM25Retriever,
    bm25_rm3: BM25RM3Retriever,
    dense: Optional[DenseRetriever],
    top_k: int,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    method_name -> [(doc_id, score), ...]
    """
    results: Dict[str, List[Tuple[str, float]]] = {}

    tfidf_res = tfidf.search(query_text, top_k=top_k)
    bm25_res = bm25.search(query_text, top_k=top_k)
    bm25_rm3_res = bm25_rm3.search(query_text, top_k=top_k)

    results["TF-IDF"] = tfidf_res
    results["BM25"] = bm25_res
    results["BM25+RM3"] = bm25_rm3_res

    if dense is not None:
        dense_res = dense.search(query_text, top_k=top_k)
        results["Dense (BioBERT)"] = dense_res

        hybrid_res = rrf_fusion([bm25_res, dense_res], top_k=top_k)
        results["Hybrid (BM25 + Dense, RRF)"] = hybrid_res

    return results


def prepare_runs_and_qrels(
    tfidf: Optional[TfidfRetriever] = None,
    bm25: Optional[BM25Retriever] = None,
    bm25_rm3: Optional[BM25RM3Retriever] = None,
    dense: Optional[DenseRetriever] = None,
    retrieval_depth: int = 100,
) -> Tuple[Dict[str, Dict[str, List[Tuple[str, float]]]], Dict[str, Dict[str, float]]]:
    """
    Build:
      runs_by_method[method_name][qid] = [(doc_id, score), ...]
    and return it along with qrels.

    If retrievers are not provided, they are built/loaded internally.
    This makes it usable from both CLI and Streamlit.
    """
    qrels = load_qrels()
    qid_to_text = load_queries()
    judged_qids = set(qrels.keys())

    # If any of the main retrievers is missing, build/load them here.
    if tfidf is None or bm25 is None or bm25_rm3 is None:
        print("[prepare] loading corpus and building retrievers...")
        docs = load_corpus()
        print(f"[prepare] loaded {len(docs)} documents.")
        if tfidf is None:
            tfidf = TfidfRetriever.build(docs)
        if bm25 is None:
            bm25 = BM25Retriever.load()
        if bm25_rm3 is None:
            bm25_rm3 = BM25RM3Retriever.load()
        # dense may or may not exist; if build fails elsewhere, we can still run lexical-only metrics
        if dense is None:
            try:
                dense = DenseRetriever.load()
            except Exception:
                dense = None

    runs_by_method: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    print(f"[prepare] qrels for {len(judged_qids)} queries")
    print(f"[prepare] loaded {len(qid_to_text)} queries from {QUERIES_PATH}")

    for qid in judged_qids:
        if qid not in qid_to_text:
            continue
        qtext = qid_to_text[qid]
        method_results = get_method_results_for_query(
            qtext,
            tfidf=tfidf,
            bm25=bm25,
            bm25_rm3=bm25_rm3,
            dense=dense,
            top_k=retrieval_depth,
        )

        for method_name, res in method_results.items():
            runs_by_method[method_name][qid] = res

    return runs_by_method, qrels


def evaluate_all_methods(
    runs_by_method: Dict[str, Dict[str, List[Tuple[str, float]]]],
    qrels: Dict[str, Dict[str, float]],
    k_prec: int = 10,
    k_ndcg: int = 10,
    k_recall: int = 50,
) -> Dict[str, Dict[str, float]]:
    """
    Return:
      {method_name: {metric_name: value}}
    Using your evaluate_run() under the hood.
    """
    metrics_by_method: Dict[str, Dict[str, float]] = {}

    for method_name, run in sorted(runs_by_method.items()):
        metrics = evaluate_run(
            run=run,
            qrels=qrels,
            k_prec=k_prec,
            k_ndcg=k_ndcg,
            k_recall=k_recall,
        )
        metrics_by_method[method_name] = metrics

    return metrics_by_method