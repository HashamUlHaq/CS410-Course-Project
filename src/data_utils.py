# src/data_utils.py

import pandas as pd
from typing import List
from dataclasses import dataclass
from pathlib import Path
from .config import MTSAMPLES_CSV

@dataclass
class Document:
    doc_id: str
    text: str

def load_corpus(path: Path = MTSAMPLES_CSV) -> List[Document]:
    """
    mtsamples.csv has a single column: 'text'.
    We'll auto-generate doc_ids as 'doc_0', 'doc_1', ...
    """
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError("mtsamples.csv must have a 'text' column.")

    docs: List[Document] = []
    for idx, row in df.iterrows():
        docs.append(Document(
            doc_id=f"doc_{idx}",
            text=str(row["text"]),
        ))
    return docs

def corpus_texts(docs: List[Document]) -> List[str]:
    return [d.text for d in docs]

def corpus_ids(docs: List[Document]) -> List[str]:
    return [d.doc_id for d in docs]