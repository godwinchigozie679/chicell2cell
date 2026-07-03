"""Evaluation metrics and threshold selection for link prediction."""

import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def compute_scores(edges_pos, edges_neg, emb):
    """
    Compute ROC-AUC and Average Precision using inner product scores
    from the latent embeddings.
    """
    if isinstance(emb, np.matrix):
        emb = np.asarray(emb)
    preds     = [_sigmoid(np.dot(emb[s], emb[t])) for s, t in edges_pos]
    preds_neg = [_sigmoid(np.dot(emb[s], emb[t])) for s, t in edges_neg]
    all_preds  = np.hstack([preds, preds_neg])
    all_labels = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    return roc_auc_score(all_labels, all_preds), \
           average_precision_score(all_labels, all_preds)


def get_link_predictions_decoder(model, cell_features, cell_edge_index,
                                  gene_features, gene_edge_index,
                                  edges_pos, edges_neg):
    """
    Extract decoder prediction scores for positive and negative edges.
    Returns concatenated predictions and binary labels.
    """
    import torch
    model.eval()
    with torch.no_grad():
        outputs  = model(cell_features, cell_edge_index,
                         gene_features, gene_edge_index)
        adj_pred = outputs['adj_pred_cell'].cpu().numpy()

    pos_scores  = [adj_pred[s, t] for s, t in edges_pos]
    neg_scores  = [adj_pred[s, t] for s, t in edges_neg]
    predictions = np.concatenate([pos_scores, neg_scores])
    labels      = np.concatenate([np.ones(len(pos_scores)),
                                   np.zeros(len(neg_scores))])
    return predictions, labels


def evaluate_on_edges(model,
                      cell_features, cell_edge_index,
                      gene_features, gene_edge_index,
                      edges_cell, edges_neg_cell,
                      edges_gene, edges_neg_gene):
    """
    Evaluate ROC-AUC and AP on cell and gene edge sets
    using the bilinear decoder predictions.
    """
    import torch
    model.eval()
    with torch.no_grad():
        outputs       = model(cell_features, cell_edge_index,
                              gene_features, gene_edge_index)
        adj_pred_cell = outputs['adj_pred_cell'].cpu().numpy()
        adj_pred_gene = outputs['adj_pred_gene'].cpu().numpy()

    def _scores(adj, pos, neg):
        p  = [adj[s, t] for s, t in pos]
        n  = [adj[s, t] for s, t in neg]
        ap = np.hstack([p, n])
        al = np.hstack([np.ones(len(p)), np.zeros(len(n))])
        return roc_auc_score(al, ap), average_precision_score(al, ap)

    roc_c, ap_c = _scores(adj_pred_cell, edges_cell,  edges_neg_cell)
    roc_g, ap_g = _scores(adj_pred_gene, edges_gene,  edges_neg_gene)
    model.train()
    return {'roc_cell': roc_c, 'ap_cell': ap_c,
            'roc_gene': roc_g, 'ap_gene': ap_g}


class ThresholdSelector:
    """
    Finds the decision threshold that maximises classification accuracy
    on the held-out test edge set.
    """

    def __init__(self, edges_pos, edges_neg):
        self.edges_pos = edges_pos
        self.edges_neg = edges_neg

    def select(self, model, cell_features, cell_edge_index,
               gene_features, gene_edge_index):
        import torch
        model.eval()
        with torch.no_grad():
            outputs  = model(cell_features, cell_edge_index,
                             gene_features, gene_edge_index)
            adj_pred = outputs['adj_pred_cell'].cpu().numpy()
        np.fill_diagonal(adj_pred, 0)

        pos_scores = np.array([adj_pred[s, t] for s, t in self.edges_pos])
        neg_scores = np.array([adj_pred[s, t] for s, t in self.edges_neg])
        all_scores = np.hstack([pos_scores, neg_scores])
        all_labels = np.hstack([np.ones(len(pos_scores)),
                                 np.zeros(len(neg_scores))])

        all_acc_score    = {}
        max_acc          = 0.0
        optimal_threshold = 0.0

        # Grid search over candidate thresholds
        for thr in np.arange(0.01, 1.0, 0.005):
            preds = (all_scores > thr).astype(int)
            acc   = accuracy_score(all_labels, preds)
            all_acc_score[float(thr)] = acc
            if acc > max_acc:
                max_acc           = acc
                optimal_threshold = float(thr)

        adj_binary = (adj_pred > optimal_threshold).astype(int)
        np.fill_diagonal(adj_binary, 0)

        print(f"Optimal threshold : {optimal_threshold:.3f}")
        print(f"Best accuracy     : {max_acc:.4f}")
        return adj_pred, adj_binary, all_acc_score, max_acc, optimal_threshold


class ModelUnpickler(pickle.Unpickler):
    """Resolves class references when loading models saved under a different namespace."""

    def find_class(self, module, name):
        from chicell2cell.model.layers import GATVAELayer, ExpRec, Decoder
        from chicell2cell.model.gatvae import CELL2CELLGATVAE
        from chicell2cell.model.config import FLAGS

        class_map = {
            'GATVAELayer'     : GATVAELayer,
            'ExpRec'          : ExpRec,
            'Decoder'         : Decoder,
            'CELL2CELLGATVAE' : CELL2CELLGATVAE,
            'FLAGS'           : FLAGS,
        }
        if name in class_map:
            return class_map[name]
        return super().find_class(module, name)


def load_model(path):
    """Load a saved chicell2cell model from a pickle file."""
    with open(path, 'rb') as f:
        return ModelUnpickler(f).load()
