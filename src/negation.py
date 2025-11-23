# src/negation.py

import re
from typing import List

NEGATION_CUES = [
    r"\bno\b",
    r"\bdenies\b",
    r"\bwithout\b",
    r"\bnot\b",
    r"\bnegative for\b",
]

def has_negation(text: str, term: str) -> bool:
    """
    Extremely simplified heuristic:
    - Look for 'no/denies/without/not' within a window of the term.
    """
    text_lower = text.lower()
    term_lower = term.lower()

    # simple window: +/- 5 tokens around term
    pattern = r"(\b(?:%s)\b.*?\b%s\b|\b%s\b.*?\b(?:%s)\b)" % (
        "|".join([c.strip(r"\b") for c in NEGATION_CUES]),
        re.escape(term_lower),
        re.escape(term_lower),
        "|".join([c.strip(r"\b") for c in NEGATION_CUES]),
    )
    return re.search(pattern, text_lower) is not None


def negation_penalty(text: str, query_term: str, base_score: float) -> float:
    """
    Example: if term appears under negation, discount the score.
    """
    if has_negation(text, query_term):
        return base_score * 0.3   # heavy penalty
    return base_score