# src/retrievers/hybrid.py

from typing import List, Tuple, Dict

def rrf_fusion(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    rankings: list of ranked lists, each is [(doc_id, score), ...]
    k: RRF constant, typical 60
    """
    scores: Dict[str, float] = {}

    for rank_list in rankings:
        for rank, (doc_id, _) in enumerate(rank_list):
            rrf_score = 1.0 / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score

    # sort by fused score
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return fused