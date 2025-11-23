import json
from pathlib import Path
from typing import Dict, List, Tuple, Set

from src.config import DATA_DIR, TOP_K
from src.data_utils import load_corpus, Document
from src.retrievers.tfidf_retriever import TfidfRetriever
from src.retrievers.bm25_retriever import BM25Retriever, BM25RM3Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.hybrid import rrf_fusion


QUERIES_PATH = DATA_DIR / "queries.jsonl"
CANDIDATES_PATH = DATA_DIR / "candidates.jsonl"

# How many *unique* docs to keep per query
MAX_UNIQUE_DOCS_PER_QUERY = 20

# How many docs to fetch from each method before pooling (bigger => more coverage)
PER_METHOD_K = 50


def build_doc_lookup(docs: List[Document]) -> Dict[str, Document]:
    return {d.doc_id: d for d in docs}


def load_queries() -> List[Dict]:
    queries = []
    with QUERIES_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


def load_already_processed_query_ids() -> Set[str]:
    """
    So you can re-run this script and it won't duplicate work.
    It inspects candidates.jsonl (if present) and collects query_ids already written.
    """
    if not CANDIDATES_PATH.exists():
        return set()

    seen: Set[str] = set()
    with CANDIDATES_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seen.add(rec["query_id"])
    return seen


def get_method_results_for_query(
    query_text: str,
    tfidf: TfidfRetriever,
    bm25: BM25Retriever,
    bm25_rm3: BM25RM3Retriever,
    dense: DenseRetriever,
    top_k: int = PER_METHOD_K,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Returns dict:
      method_name -> [(doc_id, score), ...]
    """
    results: Dict[str, List[Tuple[str, float]]] = {}

    tfidf_res = tfidf.search(query_text, top_k=top_k)
    bm25_res = bm25.search(query_text, top_k=top_k)
    bm25_rm3_res = bm25_rm3.search(query_text, top_k=top_k)
    dense_res = dense.search(query_text, top_k=top_k)
    hybrid_res = rrf_fusion([bm25_res, dense_res], top_k=top_k)

    results["TF-IDF"] = tfidf_res
    results["BM25"] = bm25_res
    results["BM25+RM3"] = bm25_rm3_res
    results["Dense"] = dense_res
    results["Hybrid"] = hybrid_res

    return results


def pool_and_rank_candidates(
    method_results: Dict[str, List[Tuple[str, float]]],
    max_docs: int = MAX_UNIQUE_DOCS_PER_QUERY,
) -> List[Dict]:
    """
    Pool candidates from multiple methods and select up to max_docs unique docs.

    We rank the union by:
      1. best (lowest) rank across methods
      2. sum of ranks across methods
    This is a simple fair-ish heuristic.
    """
    # Map: doc_id -> per-method rank/score info
    doc_info: Dict[str, Dict] = {}

    for method_name, res in method_results.items():
        for rank_idx, (doc_id, score) in enumerate(res, start=1):
            if doc_id not in doc_info:
                doc_info[doc_id] = {
                    "doc_id": doc_id,
                    "methods": {},
                }
            doc_info[doc_id]["methods"][method_name] = {
                "rank": rank_idx,
                "score": float(score),
            }

    # Compute ranking features
    ranked_docs = []
    for doc_id, info in doc_info.items():
        method_ranks = [m["rank"] for m in info["methods"].values()]
        best_rank = min(method_ranks)
        sum_ranks = sum(method_ranks)
        num_methods = len(method_ranks)
        ranked_docs.append(
            {
                "doc_id": doc_id,
                "methods": info["methods"],
                "best_rank": best_rank,
                "sum_ranks": sum_ranks,
                "num_methods": num_methods,
            }
        )

    # Sort: best_rank asc, then sum_ranks asc, then num_methods desc
    ranked_docs.sort(
        key=lambda x: (x["best_rank"], x["sum_ranks"], -x["num_methods"])
    )

    # Take top max_docs
    return ranked_docs[:max_docs]


def main():
    print("=== Building candidate pools for evaluation ===")

    # 1. Load queries
    queries = load_queries()
    print(f"Loaded {len(queries)} queries from {QUERIES_PATH}")

    # 2. Load corpus + retrievers
    print("Loading corpus and retrievers...")
    docs = load_corpus()
    doc_lookup = build_doc_lookup(docs)

    tfidf = TfidfRetriever.build(docs)
    bm25 = BM25Retriever.load()
    bm25_rm3 = BM25RM3Retriever.load()
    dense = DenseRetriever.load()

    # 3. See which queries already processed
    done_query_ids = load_already_processed_query_ids()
    if done_query_ids:
        print(f"Found existing candidates for {len(done_query_ids)} queries, will skip those.")

    # 4. Open candidates file in append mode
    out_mode = "a" if CANDIDATES_PATH.exists() else "w"
    with CANDIDATES_PATH.open(out_mode) as fout:
        for q in queries:
            qid = q["query_id"]
            qtext = q["query"]

            if qid in done_query_ids:
                print(f"Skipping {qid} (already in candidates file).")
                continue

            print(f"\n=== Processing {qid}: {qtext} ===")

            method_results = get_method_results_for_query(
                qtext, tfidf, bm25, bm25_rm3, dense, top_k=PER_METHOD_K
            )
            pooled = pool_and_rank_candidates(method_results, max_docs=MAX_UNIQUE_DOCS_PER_QUERY)

            print(f"Selected {len(pooled)} unique docs for {qid}.")

            # Write each candidate as one JSONL line with full text (for you to inspect)
            for rank_idx, info in enumerate(pooled, start=1):
                doc_id = info["doc_id"]
                text = doc_lookup[doc_id].text

                record = {
                    "query_id": qid,
                    "query": qtext,
                    "rank": rank_idx,
                    "doc_id": doc_id,
                    "methods": info["methods"],  # per-method rank & score
                    "text": text,
                }
                fout.write(json.dumps(record) + "\n")

    print(f"\nDone. Candidate pools written/appended to {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()