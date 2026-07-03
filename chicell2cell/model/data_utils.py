"""Data preparation utilities for graph construction and edge splitting."""

import numpy as np
import torch
import scipy.sparse as sp
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


def normalize_data(X):
    """Standardise features to zero mean and unit variance."""
    return StandardScaler().fit_transform(X)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to (coords, values, shape) tuple."""
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    return coords, sparse_mx.data, sparse_mx.shape


def mask_test_edges_general_link_prediction(adj, test_percent=10., val_percent=5.):
    """
    Split graph edges into train/val/test sets for link prediction.
    Samples an equal number of negative edges for each split,
    ensuring no overlap between positive and negative sets.
    Returns adj_train, train_edges, train_false, val_edges, val_false,
    test_edges, test_false.
    """
    if isinstance(adj, np.matrix):
        adj = np.asarray(adj)
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    # Remove self-loops
    adj = adj - sp.dia_matrix((adj.diagonal(), [0]), shape=adj.shape)
    adj.eliminate_zeros()
    edges_positive, _, _ = sparse_to_tuple(adj)

    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val  = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))

    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx  = edges_positive_idx[:num_val]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    val_edges   = edges_positive[val_edge_idx]
    test_edges  = edges_positive[test_edge_idx]
    train_edges = np.delete(edges_positive,
                            np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    positive_idx, _, _ = sparse_to_tuple(adj)
    positive_idx = positive_idx[:, 0] * adj.shape[0] + positive_idx[:, 1]

    def _sample_neg(n_needed, exclude_idx_sets):
        """Sample negative edges that do not appear in any positive set."""
        out  = np.empty((0, 2), dtype='int64')
        seen = np.concatenate(list(exclude_idx_sets)) if exclude_idx_sets \
               else np.array([], dtype='int64')
        while len(out) < n_needed:
            idx  = np.random.choice(adj.shape[0] ** 2,
                                    2 * (n_needed - len(out)), replace=True)
            idx  = idx[~np.in1d(idx, positive_idx, assume_unique=True)]
            idx  = idx[~np.in1d(idx, seen,          assume_unique=True)]
            rows = idx // adj.shape[0]
            cols = idx  % adj.shape[0]
            coords = np.unique(np.vstack((rows, cols)).T, axis=0)
            coords = coords[coords[:, 0] != coords[:, 1]]
            coords = coords[:n_needed - len(out)]
            out  = np.vstack([out, coords]) if len(out) else coords
            seen = np.concatenate([seen, idx])
        return out

    test_edges_false  = _sample_neg(len(test_edges),  [positive_idx])
    val_edges_false   = _sample_neg(len(val_edges),   [positive_idx])
    train_edges_false = _sample_neg(len(train_edges), [positive_idx])

    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix(
        (data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape
    )

    return (adj_train,
            train_edges, train_edges_false,
            val_edges,   val_edges_false,
            test_edges,  test_edges_false)


def adj_to_edge_index(adj, device=None):
    """Convert adjacency matrix to PyG edge_index format (2 x E)."""
    if isinstance(adj, np.matrix):
        adj = np.asarray(adj)
    if sp.issparse(adj):
        adj = adj.toarray()
    edges = np.array(np.nonzero(adj))
    t = torch.tensor(edges, dtype=torch.long)
    return t.to(device) if device else t


def apply_edge_dropout(edge_index, dropout=0.5):
    """Randomly drop edges during training for graph-level regularisation."""
    if dropout <= 0:
        return edge_index
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > dropout
    return edge_index[:, mask]


def cross_correlation(x1, x2):
    """Compute normalised cross-correlation matrix between two feature sets."""
    x1_norm = F.normalize(x1, p=2, dim=1)
    x2_norm = F.normalize(x2, p=2, dim=1)
    return torch.mm(x1_norm, x2_norm)


def correlation_reduction_loss(S):
    """
    Barlow Twins-style correlation reduction loss.
    Penalises off-diagonal entries of the cross-correlation matrix
    to reduce redundancy between cell and gene representations.
    """
    n = S.size(0)
    diag_loss     = torch.mean((torch.diagonal(S) - 1) ** 2)
    mask          = torch.ones_like(S) - torch.eye(n, device=S.device)
    off_diag_loss = torch.mean((S * mask) ** 2)
    return diag_loss + off_diag_loss
