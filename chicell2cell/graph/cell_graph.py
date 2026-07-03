"""Spatial cell adjacency graph construction."""

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors


def load_celladj(adata, distance):
    """Build binary cell adjacency matrix using a fixed distance threshold."""
    coords   = adata.obsm['spatial']
    dists    = distance_matrix(coords, coords)
    cell_adj = (dists <= distance).astype(int)
    np.fill_diagonal(cell_adj, 0)
    nc = cell_adj.sum(axis=1)
    print(f"Threshold {distance:.1f}: {nc.mean():.1f} avg neighbours, "
          f"{(nc == 0).sum()} isolated")
    return cell_adj


def knn_threshold(adata, k=4, percentile=75):
    """
    Derive a distance threshold from the k-th nearest neighbour distances.
    Uses the specified percentile to balance connectivity and sparsity.
    """
    coords = adata.obsm['spatial']
    nbrs   = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    threshold    = float(np.percentile(distances[:, k], percentile))
    print(f"KNN threshold (k={k}, p{percentile}): {threshold:.2f}")
    return threshold


def build_cell_graph(adata, k=15, percentile=90):
    """
    Convenience wrapper: compute KNN-based threshold then build adjacency.
    Default k=15, percentile=90 matches the GBM analysis configuration.
    """
    threshold = knn_threshold(adata, k=k, percentile=percentile)
    cell_adj  = load_celladj(adata, threshold)
    return cell_adj, threshold
