# src/evaluate_all_methods.py

from src.eval_pipeline import prepare_runs_and_qrels, evaluate_all_methods


def pretty_print_metrics(method_name: str, metrics: dict) -> None:
    print(f"\n=== {method_name} ===")
    for m_name, value in metrics.items():
        print(f"{m_name}: {value:.4f}")


def main():
    print("=== Evaluating all retrieval methods (on the fly) ===")

    # No retrievers passed -> eval_pipeline will build/load them internally
    runs_by_method, qrels = prepare_runs_and_qrels(retrieval_depth=100)

    metrics_by_method = evaluate_all_methods(
        runs_by_method,
        qrels,
        k_prec=10,
        k_ndcg=10,
        k_recall=10,
    )

    for method_name, metrics in metrics_by_method.items():
        pretty_print_metrics(method_name, metrics)


if __name__ == "__main__":
    main()