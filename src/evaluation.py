"""
Evaluation module for VAE Music Clustering project.

Computes all required clustering metrics:
  - Silhouette Score
  - Calinski-Harabasz Index
  - Davies-Bouldin Index
  - Adjusted Rand Index (ARI) — requires ground truth
  - Normalized Mutual Information (NMI) — requires ground truth
  - Cluster Purity — requires ground truth
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from collections import Counter


def compute_cluster_purity(labels_true: np.ndarray,
                           labels_pred: np.ndarray) -> float:
    """
    Compute Cluster Purity.

    Purity = (1/n) * sum_k max_j |c_k ∩ t_j|

    Args:
        labels_true: ground truth labels
        labels_pred: predicted cluster labels

    Returns:
        purity score in [0, 1]
    """
    contingency = {}
    for true, pred in zip(labels_true, labels_pred):
        if pred not in contingency:
            contingency[pred] = Counter()
        contingency[pred][true] += 1

    total_correct = sum(counter.most_common(1)[0][1]
                        for counter in contingency.values())
    return total_correct / len(labels_true)


def evaluate_clustering(features: np.ndarray,
                        labels_pred: np.ndarray,
                        labels_true: np.ndarray = None,
                        method_name: str = "Method") -> dict:
    """
    Compute all clustering evaluation metrics.

    Args:
        features: (n_samples, n_features) — used for internal metrics
        labels_pred: predicted cluster labels
        labels_true: optional ground truth labels
        method_name: name for display

    Returns:
        dict of metric_name -> value
    """
    results = {"method": method_name}

    # Internal metrics (don't need ground truth)
    try:
        results["silhouette_score"] = silhouette_score(features, labels_pred)
    except Exception:
        results["silhouette_score"] = float("nan")

    try:
        results["calinski_harabasz_index"] = calinski_harabasz_score(features, labels_pred)
    except Exception:
        results["calinski_harabasz_index"] = float("nan")

    try:
        results["davies_bouldin_index"] = davies_bouldin_score(features, labels_pred)
    except Exception:
        results["davies_bouldin_index"] = float("nan")

    # External metrics (need ground truth)
    if labels_true is not None:
        results["adjusted_rand_index"] = adjusted_rand_score(labels_true, labels_pred)
        results["normalized_mutual_info"] = normalized_mutual_info_score(
            labels_true, labels_pred
        )
        results["cluster_purity"] = compute_cluster_purity(labels_true, labels_pred)
    else:
        results["adjusted_rand_index"] = None
        results["normalized_mutual_info"] = None
        results["cluster_purity"] = None

    # Pretty print
    print(f"\n{'='*50}")
    print(f" Clustering Evaluation: {method_name}")
    print(f"{'='*50}")
    print(f"  Silhouette Score:        {results['silhouette_score']:.4f}")
    print(f"  Calinski-Harabasz Index: {results['calinski_harabasz_index']:.2f}")
    print(f"  Davies-Bouldin Index:    {results['davies_bouldin_index']:.4f}")
    if labels_true is not None:
        print(f"  Adjusted Rand Index:     {results['adjusted_rand_index']:.4f}")
        print(f"  Norm. Mutual Info:       {results['normalized_mutual_info']:.4f}")
        print(f"  Cluster Purity:          {results['cluster_purity']:.4f}")
    print(f"{'='*50}\n")

    return results


def compare_methods(results_list: list) -> pd.DataFrame:
    """
    Create a comparison table from multiple evaluation results.

    Args:
        results_list: list of dicts from evaluate_clustering()

    Returns:
        pd.DataFrame comparison table
    """
    df = pd.DataFrame(results_list)
    df = df.set_index("method")

    # Format nicely
    print("\n" + "=" * 70)
    print(" METHOD COMPARISON TABLE")
    print("=" * 70)
    print(df.to_string(float_format=lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"))
    print("=" * 70 + "\n")

    return df
