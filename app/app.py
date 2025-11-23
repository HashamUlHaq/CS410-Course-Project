import os
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add project root to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from src.data_utils import load_corpus, Document
from src.retrievers.tfidf_retriever import TfidfRetriever
from src.retrievers.bm25_retriever import BM25Retriever, BM25RM3Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.hybrid import rrf_fusion
from src.config import TOP_K, FAISS_INDEX_PATH, DATA_DIR

# NEW: evaluation pipeline helpers
from src.eval_pipeline import (
    prepare_runs_and_qrels,
    evaluate_all_methods,
)


# ---------- Helpers ----------

def build_doc_lookup(docs: List[Document]) -> Dict[str, Document]:
    return {d.doc_id: d for d in docs}


def make_snippet(text: str, max_len: int = 300) -> str:
    text = str(text).strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# ---------- Cached resources ----------

@st.cache_resource(show_spinner=True)
def get_corpus_and_tfidf():
    docs = load_corpus()
    tfidf = TfidfRetriever.build(docs)
    doc_lookup = build_doc_lookup(docs)
    return docs, tfidf, doc_lookup


@st.cache_resource(show_spinner=True)
def get_bm25():
    # Assumes you've already built the Lucene index with build_bm25_index.py
    return BM25Retriever.load()


@st.cache_resource(show_spinner=True)
def get_bm25_rm3():
    return BM25RM3Retriever.load()


@st.cache_resource(show_spinner=True)
def get_dense():
    """
    Loads an existing FAISS index + BioBERT weights.
    """
    if not Path(FAISS_INDEX_PATH).exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run DenseRetriever.build(docs) once before using the app."
        )
    return DenseRetriever.load()


# ---------- Retrieval logic (search tab) ----------

def run_search(
    query: str,
    top_k: int,
    tfidf: TfidfRetriever,
    bm25: BM25Retriever,
    bm25_rm3: BM25RM3Retriever,
    dense: DenseRetriever | None,
    doc_lookup: Dict[str, Document],
    include_rm3: bool = True,
    include_dense: bool = True,
    include_hybrid: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Runs multiple retrieval strategies and returns a dict of
    method_name -> DataFrame with columns [rank, doc_id, score, snippet, text].
    """

    results: Dict[str, pd.DataFrame] = {}

    # --- TF-IDF ---
    tfidf_res = tfidf.search(query, top_k=top_k)
    results["TF-IDF"] = results_to_df(tfidf_res, doc_lookup)

    # --- BM25 ---
    bm25_res = bm25.search(query, top_k=top_k)
    results["BM25"] = results_to_df(bm25_res, doc_lookup)

    # --- BM25 + RM3 ---
    if include_rm3:
        bm25_rm3_res = bm25_rm3.search(query, top_k=top_k)
        results["BM25+RM3"] = results_to_df(bm25_rm3_res, doc_lookup)

    # --- Dense ---
    if include_dense and dense is not None:
        dense_res = dense.search(query, top_k=top_k)
        results["Dense (BioBERT)"] = results_to_df(dense_res, doc_lookup)

    # --- Hybrid (RRF of BM25 + Dense) ---
    if include_dense and include_hybrid and dense is not None:
        hybrid_res = rrf_fusion(
            [bm25_res, dense_res],
            top_k=top_k,
        )
        results["Hybrid (BM25 + Dense, RRF)"] = results_to_df(hybrid_res, doc_lookup)

    return results


def results_to_df(
    results: List[Tuple[str, float]],
    doc_lookup: Dict[str, Document],
) -> pd.DataFrame:
    rows = []
    for rank, (doc_id, score) in enumerate(results, start=1):
        doc = doc_lookup.get(doc_id)
        text = doc.text if doc is not None else ""
        rows.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "score": score,
                "snippet": make_snippet(text),
                "text": text,  # full text here
            }
        )
    return pd.DataFrame(rows)


# ---------- Main app ----------

def main():
    st.set_page_config(
        page_title="DocuSherlock Retrieval Workbench",
        layout="wide",
    )

    st.title("🔎 DocuSherlock: Clinical Retrieval Workbench")
    st.markdown(
        """
        Enter a clinical-style query below (e.g., **"leg pain without swelling"**, 
        **"hyperlipidemia without diabetes"**) and compare what different retrieval
        strategies return on the same corpus, or switch to the *Evaluation* tab
        to see metrics on your annotated benchmark.
        """
    )

    # Sidebar: search settings
    with st.sidebar:
        st.header("Search Settings")

        top_k = st.slider(
            "Top K results per method",
            5,
            50,
            value=min(TOP_K, 10),
            step=1,
        )

        include_rm3 = st.checkbox("Include BM25+RM3", value=True)
        include_dense = st.checkbox("Include Dense (BioBERT) retrieval", value=True)
        include_hybrid = st.checkbox("Include Hybrid (BM25 + Dense RRF)", value=True)

        show_dataframes = st.checkbox("Show dataframes", value=False)

        st.markdown("---")
        st.caption(
            "Note: BM25 index and FAISS dense index must be built ahead of time.\n"
            "See README."
        )

    # Load corpus & retrievers (cached, used by both tabs)
    with st.spinner("Loading corpus and retrievers..."):
        docs, tfidf, doc_lookup = get_corpus_and_tfidf()
        bm25 = get_bm25()
        bm25_rm3 = get_bm25_rm3()

        dense: DenseRetriever | None = None
        try:
            dense = get_dense()
        except FileNotFoundError as e:
            st.sidebar.error(str(e))
            include_dense = False
            include_hybrid = False

    # Tabs: Search | Evaluation
    tab_search, tab_eval = st.tabs(["🔍 Search", "📊 Evaluation"])

    # ----- Search tab -----
    with tab_search:
        query = st.text_input(
            "Enter your query:",
            value="hyperlipidemia without diabetes",
            help="Clinical-style query. No chat history, just a single query.",
        )

        run_button = st.button("Run retrieval", type="primary")

        if run_button and query.strip():
            st.markdown(f"### Results for query: `{query.strip()}`")

            results = run_search(
                query=query.strip(),
                top_k=top_k,
                tfidf=tfidf,
                bm25=bm25,
                bm25_rm3=bm25_rm3,
                dense=dense,
                doc_lookup=doc_lookup,
                include_rm3=include_rm3,
                include_dense=include_dense,
                include_hybrid=include_hybrid,
            )

            method_names = list(results.keys())

            # Show 2–3 columns per row depending on how many methods are active
            n_methods = len(method_names)
            n_cols = min(3, n_methods)

            st.markdown("**Full documents (top 5)**")
            st.caption("Click to expand any result to view the entire note.")
            
            for i in range(0, n_methods, n_cols):
                row_methods = method_names[i : i + n_cols]
                cols = st.columns(len(row_methods))
                for col, method_name in zip(cols, row_methods):
                    with col:
                        st.subheader(method_name)

                        df = results[method_name]

                        if show_dataframes:
                            st.dataframe(
                                df[["rank", "doc_id", "score", "snippet"]],
                                use_container_width=True,
                            )

                        top_n_full = min(5, len(df))
                        for _, row in df.head(top_n_full).iterrows():
                            header = (
                                f"Rank {int(row['rank'])} – {row['doc_id']} "
                                f"(score={row['score']:.3f})"
                            )
                            with st.expander(header):
                                st.write(row["text"])

        elif run_button and not query.strip():
            st.warning("Please enter a query first.")

    # ----- Evaluation tab -----
    with tab_eval:
        st.subheader("Evaluation on Annotated DocuSherlock Benchmark")
        st.markdown(
            """
            This tab evaluates all retrieval strategies on your **qrels.jsonl** +
            **queries.jsonl** benchmark using your existing **evaluate_run** metrics
            (P@k, nDCG@k, Recall@k).
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            k = st.slider(
                "k for P@k and nDCG@k",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
            )

        with col2:
            k_recall = st.slider(
                "k for Recall@k",
                min_value=10,
                max_value=100,
                value=50,
                step=5,
            )

        eval_button = st.button("Run evaluation", key="run_eval", type="secondary")

        if eval_button:
            retrieval_depth = max(k, k_recall)

            with st.spinner("Running evaluation over judged queries..."):
                runs_by_method, qrels = prepare_runs_and_qrels(
                    tfidf=tfidf,
                    bm25=bm25,
                    bm25_rm3=bm25_rm3,
                    dense=dense,
                    retrieval_depth=retrieval_depth,
                )

                metrics_by_method = evaluate_all_methods(
                    runs_by_method,
                    qrels,
                    k_prec=k,
                    k_ndcg=k,
                    k_recall=k_recall,
                )

            # Convert to DataFrame for a nice table
            df = (
                pd.DataFrame(metrics_by_method)
                .T  # methods as rows
                .reset_index()
                .rename(columns={"index": "Method"})
            )

            # Sort by nDCG@k if present
            ndcg_col = f"nDCG@{k}"
            if ndcg_col in df.columns:
                df = df.sort_values(ndcg_col, ascending=False)

            st.markdown("### Metrics per method")
            st.dataframe(df.style.format(precision=4), use_container_width=True)

            st.caption(
                f"Evaluated on {len(qrels)} queries from qrels.jsonl · "
                f"P@{k}, nDCG@{k}, Recall@{k_recall}."
            )


if __name__ == "__main__":
    main()