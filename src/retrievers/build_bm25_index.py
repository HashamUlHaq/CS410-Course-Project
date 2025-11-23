# src/retrievers/build_bm25_index.py

import json
import subprocess
from pathlib import Path

from ..config import DATA_DIR
from ..data_utils import load_corpus

INPUT_DIR = DATA_DIR / "pyserini_json"
INDEX_DIR = DATA_DIR / "bm25_index"


def build_bm25_index():
    # 1. Load docs
    docs = load_corpus()

    # 2. Write Pyserini JsonCollection
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = INPUT_DIR / "docs.json"

    with json_path.open("w") as fout:
        for d in docs:
            fout.write(json.dumps({
                "id": d.doc_id,
                "contents": d.text,
            }) + "\n")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Call Pyserini indexer
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(INPUT_DIR),
        "--index", str(INDEX_DIR),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "4",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Lucene index built at", INDEX_DIR)


if __name__ == "__main__":
    build_bm25_index()