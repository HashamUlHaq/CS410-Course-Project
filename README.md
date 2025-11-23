
# Start the streamlit app

`streamlit run app.app.py`

# Build indexes

Indexes are aleardy provided in data - You'll see two files:
- `faiss_index.bin`
- `bm25_index`

If the indexes need to be refreshed / built again, Run the following command to build them:
`python -m src.build_all_indexes`

We only pre-build and store tw indexes: `BM25` and `Dense Embeddings`. TF-IDF is built on the fly when the application starts as it doesn't need heavy processing.
