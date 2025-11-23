# src/eval_metrics.py

from typing import Dict, List, Set, Tuple
import math

# qrels: {query_id: {doc_id: relevance_label (0/1 or graded)}}
# results: {query_id: [(doc_id, score), ...]}

def precision_at_k(results: List[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved = results[:k]
    hits = sum(1 for d in retrieved if d in relevant)
    return hits / k

def recall_at_k(results: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved = results[:k]
    hits = sum(1 for d in retrieved if d in relevant)
    return hits / len(relevant)

def dcg_at_k(rels: List[float], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(rels[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg

def ndcg_at_k(results: List[str], rel_dict: Dict[str, float], k: int) -> float:
    rels = [rel_dict.get(d, 0.0) for d in results]
    dcg = dcg_at_k(rels, k)
    ideal_rels = sorted(rel_dict.values(), reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def evaluate_run(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, float]],
    k_prec: int = 10,
    k_ndcg: int = 10,
    k_recall: int = 50,
) -> Dict[str, float]:
    """
    run: {qid: [(doc_id, score), ...]}
    qrels: {qid: {doc_id: rel}}
    """
    p_at_k = []
    ndcgs = []
    recalls = []

    for qid, rels in qrels.items():
        if qid not in run:
            continue
        ranked_doc_ids = [d for d, _ in run[qid]]
        rel_set = {d for d, r in rels.items() if r > 0}
        p_at_k.append(precision_at_k(ranked_doc_ids, rel_set, k_prec))
        ndcgs.append(ndcg_at_k(ranked_doc_ids, rels, k_ndcg))
        recalls.append(recall_at_k(ranked_doc_ids, rel_set, k_recall))

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        f"P@{k_prec}": avg(p_at_k),
        f"nDCG@{k_ndcg}": avg(ndcgs),
        f"Recall@{k_recall}": avg(recalls),
    }