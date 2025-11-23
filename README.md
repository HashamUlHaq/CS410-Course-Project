# Build indexes

First the indexes need to be build. Two indexes are pre-built: `BM25` and `Dense Embeddings`. TF-IDF is built on the fly when the application starts as it doesn't need more processing.

Run the following command to build the indexes:
`python -m src.build_all_indexes`

You'll see two files:
- `faiss_index.bin`
- `bm25_index`

# Start the streamlit app

`streamlit run app.app.py`