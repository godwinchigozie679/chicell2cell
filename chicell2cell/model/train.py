"""Training loop with comprehensive metrics tracking."""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .config import FLAGS
from .data_utils import cross_correlation, correlation_reduction_loss
from .evaluate import evaluate_on_edges


def compute_total_loss(outputs, cell_edge_index, gene_edge_index, exp,
                       pos_weight_cell, norm_cell, pos_weight_gene, norm_gene,
                       num_nodes_cell, num_nodes_gene, epoch, epochs):
    """
    Composite training loss combining:
    - Weighted BCE reconstruction loss for cell and gene graphs
    - KL divergence with linear annealing
    - Expression reconstruction MSE
    - Barlow Twins correlation reduction loss
    - L2 regularisation on latent means
    """
    device = outputs['z_cell'].device

    # Reconstruct adjacency targets from edge indices
    adj_true_cell = torch.zeros_like(outputs['adj_pred_cell'])
    adj_true_cell[cell_edge_index[0], cell_edge_index[1]] = 1
    adj_true_gene = torch.zeros_like(outputs['adj_pred_gene'])
    adj_true_gene[gene_edge_index[0], gene_edge_index[1]] = 1

    eps = 1e-7
    logits_cell = torch.log(
        (outputs['adj_pred_cell'] + eps) / (1 - outputs['adj_pred_cell'] + eps))
    logits_gene = torch.log(
        (outputs['adj_pred_gene'] + eps) / (1 - outputs['adj_pred_gene'] + eps))

    # Weighted BCE to handle class imbalance in sparse graphs
    cross_gene = F.binary_cross_entropy_with_logits(
        logits_gene, adj_true_gene,
        pos_weight=torch.full_like(adj_true_gene, pos_weight_gene),
        reduction='none')
    cross_gene = torch.where(torch.isnan(cross_gene),
                             torch.zeros_like(cross_gene), cross_gene)
    log_lik_gene = norm_gene * torch.mean(cross_gene)

    cross_cell = F.binary_cross_entropy_with_logits(
        logits_cell, adj_true_cell,
        pos_weight=torch.full_like(adj_true_cell, pos_weight_cell),
        reduction='none')
    cross_cell = torch.where(torch.isnan(cross_cell),
                             torch.zeros_like(cross_cell), cross_cell)
    log_lik_cell = norm_cell * torch.mean(cross_cell)

    # KL divergence terms
    kl_gene = (0.5 / num_nodes_gene) * torch.mean(torch.sum(
        1 + 2 * outputs['z_log_std_gene']
        - torch.square(outputs['z_mean_gene'])
        - torch.square(torch.exp(outputs['z_log_std_gene'])), dim=1))
    kl_cell = (0.5 / num_nodes_cell) * torch.mean(torch.sum(
        1 + 2 * outputs['z_log_std_cell']
        - torch.square(outputs['z_mean_cell'])
        - torch.square(torch.exp(outputs['z_log_std_cell'])), dim=1))

    # Linear KL annealing — weight increases from 0 to 1 over first half of training
    kl_weight = min(1.0, epoch / (epochs * 0.5))
    cost_gene  = log_lik_gene - kl_weight * kl_gene
    cost_cell  = log_lik_cell - kl_weight * kl_cell

    # Expression reconstruction loss
    exp_tensor = torch.tensor(
        exp.values if hasattr(exp, 'values') else exp,
        dtype=torch.float32, device=device)
    x_g = torch.mean(torch.square(outputs['exp_rec_gene'] - exp_tensor.T))
    x_c = torch.mean(torch.square(outputs['exp_rec_cell'] - exp_tensor))

    # Barlow Twins correlation reduction between cell and gene reconstructions
    S   = cross_correlation(outputs['exp_rec_gene'], outputs['exp_rec_cell'])
    l_x = correlation_reduction_loss(S)

    # L2 regularisation on latent means
    l2_cell = FLAGS.l2_latent * torch.mean(outputs['z_mean_cell'] ** 2)
    l2_gene = FLAGS.l2_latent * torch.mean(outputs['z_mean_gene'] ** 2)

    total_cost = cost_gene + cost_cell + x_g + x_c + l_x + l2_cell + l2_gene

    return {'total_loss': total_cost,
            'adj_true_cell': adj_true_cell,
            'adj_true_gene': adj_true_gene}


def train_model(model,
                cell_features, cell_edge_index,
                gene_features, gene_edge_index,
                train_edges_cell, train_edges_neg_cell,
                train_edges_gene, train_edges_neg_gene,
                val_edges_cell,   val_edges_neg_cell,
                val_edges_gene,   val_edges_neg_gene,
                pos_weight_cell, norm_cell,
                pos_weight_gene, norm_gene,
                num_nodes_cell, num_nodes_gene,
                exp,
                epochs=None, patience=None):
    """
    Train the GAT-VAE model with early stopping based on
    validation cell-level ROC-AUC. Restores best model weights
    at the end of training.
    """
    epochs   = epochs   or FLAGS.epochs
    patience = patience or FLAGS.patience

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
    # Halve LR when validation ROC-AUC plateaus for 15 intervals
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15)

    history = {
        'total_loss': [],
        'train_roc_cell': [], 'train_ap_cell': [],
        'train_roc_gene': [], 'train_ap_gene': [],
        'val_roc_cell':   [], 'val_ap_cell':   [],
        'val_roc_gene':   [], 'val_ap_gene':   [],
        'gap_roc_cell':   [], 'gap_ap_cell':   [],
        'gap_roc_gene':   [], 'gap_ap_gene':   [],
    }

    best_val_roc     = 0.0
    best_epoch       = 0
    patience_counter = 0
    best_state       = None

    for epoch in tqdm(range(1, epochs + 1), desc='Training'):
        model.train()
        outputs = model(cell_features, cell_edge_index,
                        gene_features, gene_edge_index)
        losses  = compute_total_loss(
            outputs, cell_edge_index, gene_edge_index, exp,
            pos_weight_cell, norm_cell, pos_weight_gene, norm_gene,
            num_nodes_cell, num_nodes_gene, epoch, epochs)

        optimizer.zero_grad()
        losses['total_loss'].backward()
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        history['total_loss'].append(losses['total_loss'].item())

        # Evaluate every log_interval epochs
        if epoch % FLAGS.log_interval == 0 or epoch == 1:
            train_m = evaluate_on_edges(
                model, cell_features, cell_edge_index,
                gene_features, gene_edge_index,
                train_edges_cell, train_edges_neg_cell,
                train_edges_gene, train_edges_neg_gene)
            val_m = evaluate_on_edges(
                model, cell_features, cell_edge_index,
                gene_features, gene_edge_index,
                val_edges_cell, val_edges_neg_cell,
                val_edges_gene, val_edges_neg_gene)

            for key in ('roc_cell', 'ap_cell', 'roc_gene', 'ap_gene'):
                history[f'train_{key}'].append(train_m[key])
                history[f'val_{key}'].append(val_m[key])

            history['gap_roc_cell'].append(train_m['roc_cell'] - val_m['roc_cell'])
            history['gap_ap_cell'].append( train_m['ap_cell']  - val_m['ap_cell'])
            history['gap_roc_gene'].append(train_m['roc_gene'] - val_m['roc_gene'])
            history['gap_ap_gene'].append( train_m['ap_gene']  - val_m['ap_gene'])

            scheduler.step(val_m['roc_cell'])

            if epoch % (FLAGS.log_interval * 2) == 0:
                print(f"\nEpoch {epoch}/{epochs} | Loss {losses['total_loss'].item():.5f}")
                print(f"  Cell ROC  train={train_m['roc_cell']:.4f}  val={val_m['roc_cell']:.4f}")
                print(f"  Gene ROC  train={train_m['roc_gene']:.4f}  val={val_m['roc_gene']:.4f}")

            # Early stopping on cell-level validation ROC-AUC
            if val_m['roc_cell'] > best_val_roc:
                best_val_roc     = val_m['roc_cell']
                best_epoch       = epoch
                patience_counter = 0
                best_state       = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}
                if epoch % (FLAGS.log_interval * 2) == 0:
                    print('  New best model saved.')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping at epoch {epoch}')
                    break

    # Restore best model weights
    if best_state:
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f'\nBest model restored from epoch {best_epoch}')

    return model, history, best_epoch
