from typing import List, Tuple
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from ..data_utils import Document, corpus_texts, corpus_ids

@dataclass
class TfidfRetriever:
    vectorizer: TfidfVectorizer
    doc_matrix: any
    doc_ids: List[str]

    @classmethod
    def build(cls, docs: List[Document]) -> "TfidfRetriever":
        texts = corpus_texts(docs)
        doc_ids = corpus_ids(docs)
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )
        doc_matrix = vectorizer.fit_transform(texts)
        return cls(vectorizer, doc_matrix, doc_ids)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_vec = self.vectorizer.transform([query])
        scores = linear_kernel(q_vec, self.doc_matrix).flatten()
        ranked_idx = scores.argsort()[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in ranked_idx]