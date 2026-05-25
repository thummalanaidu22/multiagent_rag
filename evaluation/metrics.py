"""
Metrics reporting: aggregate scores, print tables, save charts.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
from config import RESULTS_DIR


def compute_metrics_report(
    records: List[Dict[str, Any]],
    metrics: Dict[str, float],
    k: int,
    output_prefix: str = "evaluation",
) -> str:
    """
    Save evaluation records to CSV, print a summary table,
    and generate bar charts. Returns path to saved CSV.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Save raw records ---
    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / f"{output_prefix}_records.csv"
    df.to_csv(csv_path, index=False)

    # --- Save metrics summary ---
    metrics_path = RESULTS_DIR / f"{output_prefix}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Print summary table ---
    print("\n" + "=" * 55)
    print("  EVALUATION METRICS SUMMARY")
    print("=" * 55)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric:<25} {value:.4f}")
        else:
            print(f"  {metric:<25} {value}")
    print("=" * 55)

    # --- Bar chart: retrieval metrics ---
    retrieval_keys = [f"precision@{k}", f"recall@{k}", "mrr"]
    retrieval_vals = [metrics.get(m, 0) for m in retrieval_keys]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(retrieval_keys, retrieval_vals, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Retrieval Metrics (K={k})", fontsize=13)
    axes[0].set_ylabel("Score")
    for i, v in enumerate(retrieval_vals):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)

    # --- Bar chart: generation metrics ---
    gen_keys = ["faithfulness", "answer_relevance"]
    gen_vals = [metrics.get(m, 0) for m in gen_keys]

    axes[1].bar(gen_keys, gen_vals, color=["#C44E52", "#8172B2"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Generation Metrics (LLM-as-Judge)", fontsize=13)
    axes[1].set_ylabel("Score")
    for i, v in enumerate(gen_vals):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)

    plt.tight_layout()
    chart_path = RESULTS_DIR / f"{output_prefix}_chart.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n  Chart saved → {chart_path}")

    # --- Per-question score distribution ---
    if "faithfulness" in df.columns and "answer_relevance" in df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        df[["faithfulness", "answer_relevance"]].plot(
            kind="box", ax=ax2, patch_artist=True,
            boxprops=dict(facecolor="#AEC6CF"),
        )
        ax2.set_title("Score Distribution per Question")
        ax2.set_ylabel("Score (0-1)")
        dist_path = RESULTS_DIR / f"{output_prefix}_distribution.png"
        plt.tight_layout()
        plt.savefig(dist_path, dpi=150)
        plt.close()

    print(f"  Records saved → {csv_path}")
    print(f"  Metrics saved → {metrics_path}")
    return str(csv_path)
