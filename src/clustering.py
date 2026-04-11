"""
Clustering module for VAE Music Clustering project.

Contains:
  - K-Means clustering
  - Agglomerative (hierarchical) clustering
  - DBSCAN with automatic hyperparameter tuning
  - Elbow method for optimal K
  - PCA + K-Means baseline
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from typing import Optional

from src.config import N_CLUSTERS, RANDOM_STATE


def kmeans_clustering(features: np.ndarray,
                      n_clusters: int = N_CLUSTERS,
                      random_state: int = RANDOM_STATE) -> np.ndarray:
    """
    Apply K-Means clustering.

    Args:
        features: (n_samples, n_features)
        n_clusters: number of clusters
        random_state: random seed

    Returns:
        cluster_labels: (n_samples,)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state,
                    n_init=30, max_iter=500)
    labels = kmeans.fit_predict(features)
    return labels


def pca_kmeans_baseline(features: np.ndarray,
                        n_components: int = 32,
                        n_clusters: int = N_CLUSTERS,
                        random_state: int = RANDOM_STATE) -> tuple:
    """
    Baseline: PCA dimensionality reduction + K-Means clustering.

    Args:
        features: (n_samples, n_features)
        n_components: PCA components (should match latent_dim for fair comparison)
        n_clusters: number of clusters
        random_state: random seed

    Returns:
        labels: cluster labels
        pca_features: PCA-reduced features
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_features = pca.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state,
                    n_init=30, max_iter=500)
    labels = kmeans.fit_predict(pca_features)

    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA explained variance: {explained_var:.1f}% with {n_components} components")

    return labels, pca_features


def find_optimal_k(features: np.ndarray,
                   k_range: range = range(2, 11),
                   random_state: int = RANDOM_STATE) -> dict:
    """
    Find optimal number of clusters using elbow method and silhouette scores.

    Args:
        features: (n_samples, n_features)
        k_range: range of K values to try
        random_state: random seed

    Returns:
        dict with:
            inertias: list of inertia values
            silhouette_scores: list of silhouette scores
            best_k: optimal K based on silhouette
    """
    from sklearn.metrics import silhouette_score

    inertias = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state,
                        n_init=30, max_iter=500)
        labels = kmeans.fit_predict(features)
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(features, labels)
        silhouette_scores.append(sil_score)
        print(f"  K={k}: Inertia={kmeans.inertia_:.1f}, Silhouette={sil_score:.4f}")

    best_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f"\nBest K by silhouette score: {best_k}")

    return {
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "best_k": best_k,
        "k_range": list(k_range),
    }


def gmm_clustering(features: np.ndarray,
                    n_clusters: int = N_CLUSTERS,
                    random_state: int = RANDOM_STATE) -> np.ndarray:
    """
    Apply Gaussian Mixture Model clustering.

    A natural fit for VAE latent spaces since VAEs assume Gaussian
    distributions in the latent space.

    Args:
        features: (n_samples, n_features)
        n_clusters: number of mixture components
        random_state: random seed

    Returns:
        cluster_labels: (n_samples,)
    """
    gmm = GaussianMixture(n_components=n_clusters,
                           covariance_type="full",
                           random_state=random_state,
                           n_init=10, max_iter=500)
    return gmm.fit_predict(features)


# ============================================================
# Medium Task: Agglomerative + DBSCAN
# ============================================================

def agglomerative_clustering(features: np.ndarray,
                              n_clusters: int = N_CLUSTERS,
                              linkage: str = "ward") -> np.ndarray:
    """
    Apply Agglomerative (Hierarchical) Clustering.

    Args:
        features: (n_samples, n_features)
        n_clusters: number of clusters
        linkage: linkage criterion — "ward" (default), "complete",
                 "average", or "single". Note: "ward" requires
                 euclidean affinity.

    Returns:
        cluster_labels: (n_samples,)
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return model.fit_predict(features)


def dbscan_clustering(features: np.ndarray,
                      eps: float = 0.5,
                      min_samples: int = 5) -> np.ndarray:
    """
    Apply DBSCAN clustering.

    Args:
        features: (n_samples, n_features)
        eps: maximum distance between two samples to be considered
             in the same neighbourhood
        min_samples: minimum number of samples in a neighbourhood
                     to form a core point

    Returns:
        cluster_labels: (n_samples,) — label -1 means noise point.
        Use mask = labels != -1 before computing silhouette score.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    return model.fit_predict(features)


def tune_dbscan(features: np.ndarray,
                eps_range: Optional[list] = None,
                min_samples_range: Optional[list] = None,
                random_state: int = RANDOM_STATE) -> dict:
    """
    Grid search for DBSCAN hyperparameters using silhouette score.

    Only configurations that produce at least 2 non-noise clusters
    and fewer than n_samples//2 noise points are considered.

    Args:
        features: (n_samples, n_features)
        eps_range: list of eps values to try (default: 0.3 … 3.0)
        min_samples_range: list of min_samples values to try (default: 3, 5, 10)
        random_state: unused (DBSCAN is deterministic); kept for API consistency

    Returns:
        dict with: best_eps, best_min_samples, best_score, best_labels,
                   all_results (list of dicts)
    """
    from sklearn.metrics import silhouette_score

    if eps_range is None:
        eps_range = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    if min_samples_range is None:
        min_samples_range = [3, 5, 10]

    n_samples = len(features)
    best_score = -2.0
    best_eps = eps_range[0]
    best_min_samples = min_samples_range[0]
    best_labels = None
    all_results = []

    print("Tuning DBSCAN hyperparameters...")
    for eps in eps_range:
        for min_s in min_samples_range:
            labels = dbscan_clustering(features, eps=eps, min_samples=min_s)
            noise_count = np.sum(labels == -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            # Skip degenerate configurations
            if n_clusters < 2 or noise_count > n_samples // 2:
                all_results.append({
                    "eps": eps, "min_samples": min_s,
                    "n_clusters": n_clusters, "noise": noise_count,
                    "silhouette": None,
                })
                continue

            mask = labels != -1
            score = silhouette_score(features[mask], labels[mask])
            all_results.append({
                "eps": eps, "min_samples": min_s,
                "n_clusters": n_clusters, "noise": noise_count,
                "silhouette": score,
            })

            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_s
                best_labels = labels
            print(f"  eps={eps:.1f}, min_s={min_s}: "
                  f"clusters={n_clusters}, noise={noise_count}, sil={score:.4f}")

    # Fallback if no valid config found
    if best_labels is None:
        print("Warning: no valid DBSCAN config found; using eps=1.5, min_samples=3")
        best_eps, best_min_samples = 1.5, 3
        best_labels = dbscan_clustering(features, eps=best_eps,
                                        min_samples=best_min_samples)

    print(f"\nBest DBSCAN: eps={best_eps}, min_samples={best_min_samples}, "
          f"silhouette={best_score:.4f}")
    return {
        "best_eps": best_eps,
        "best_min_samples": best_min_samples,
        "best_score": best_score,
        "best_labels": best_labels,
        "all_results": all_results,
    }
