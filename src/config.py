# src/config.py

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MTSAMPLES_CSV = DATA_DIR / "mtsamples.csv"   # <— NEW

# Retrieval settings
TOP_K = 50          # for Recall@50 etc.
EMBED_BATCH_SIZE = 32

# BioBERT model name - you can swap this if needed
DENSE_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
DENSE_EMB_CACHE = DATA_DIR / "dense_embeddings.npy"
DOCID_MAP_PATH = DATA_DIR / "docid_map.json"