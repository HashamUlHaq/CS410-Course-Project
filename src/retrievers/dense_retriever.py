# src/retrievers/dense_retriever.py

from typing import List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import faiss
import torch
torch.set_num_threads(1)
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from ..config import (
    DENSE_MODEL_NAME,
    FAISS_INDEX_PATH,
    DENSE_EMB_CACHE,
    DOCID_MAP_PATH,
    EMBED_BATCH_SIZE,          # <-- import from config
)
from ..data_utils import Document, corpus_texts, corpus_ids


def mean_pool(last_hidden_state, attention_mask):
    # last_hidden_state: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@dataclass
class DenseRetriever:
    tokenizer: AutoTokenizer
    model: AutoModel
    index: faiss.IndexFlatIP
    id2doc: Dict[int, str]

    @classmethod
    def build(cls, docs: List[Document], device: str = None) -> "DenseRetriever":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL_NAME)
        model = AutoModel.from_pretrained(DENSE_MODEL_NAME, 
                                            dtype=torch.float32, 
                                            use_safetensors=False,
                                            low_cpu_mem_usage=False)
        model.to(device)
        model.eval()

        texts = corpus_texts(docs)
        doc_ids = corpus_ids(docs)
        
        embeddings = []
        batch_size = EMBED_BATCH_SIZE

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding docs"):
            batch = texts[i : i + batch_size]

            # skip empty strings just in case
            batch = [t for t in batch if t.strip()]
            if not batch:
                continue

            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,   # 256 is usually enough & faster than 512
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                out = model(**enc)
                emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                embeddings.append(emb.cpu().numpy())
            
        # Keep float32 for FAISS (see our earlier reasoning)
        embeddings = np.vstack(embeddings).astype("float32")

        # Save for reuse (optional but handy)
        np.save(DENSE_EMB_CACHE, embeddings)

        # index -> doc_id map (JSON will stringify keys)
        id2doc = {i: doc_id for i, doc_id in enumerate(doc_ids)}
        with open(DOCID_MAP_PATH, "w") as f:
            json.dump(id2doc, f)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(FAISS_INDEX_PATH))

        return cls(tokenizer, model, index, id2doc)

    @classmethod
    def load(cls, device: str = None) -> "DenseRetriever":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL_NAME)
        model = AutoModel.from_pretrained(DENSE_MODEL_NAME)
        model.to(device)
        model.eval()

        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(DOCID_MAP_PATH) as f:
            id2doc = json.load(f)

        return cls(tokenizer, model, index, id2doc)

    def encode_query(self, query: str, device: str = None) -> np.ndarray:
        if device is None:
            device = next(self.model.parameters()).device

        enc = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = self.model(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype("float32")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_emb = self.encode_query(query)
        scores, idx = self.index.search(q_emb, top_k)
        scores = scores[0]
        idx = idx[0]
        results = []
        for i, s in zip(idx, scores):
            if i == -1:
                continue
            # keys come back as strings after json.load()
            doc_id = self.id2doc[str(i)]
            results.append((doc_id, float(s)))
        return results