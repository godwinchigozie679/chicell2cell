"""Training history and score distribution plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_score_distribution(model, cell_features, cell_edge_index,
                             gene_features, gene_edge_index,
                             test_edges_pos, test_edges_neg,
                             save_path=None):
    """
    Plot decoder prediction score distributions for positive
    (true communication) and negative (absent) edge sets.
    """
    from chicell2cell.model.evaluate import get_link_predictions_decoder

    predictions, labels = get_link_predictions_decoder(
        model, cell_features, cell_edge_index,
        gene_features, gene_edge_index,
        test_edges_pos, test_edges_neg,
    )

    pos_preds = predictions[labels == 1]
    neg_preds = predictions[labels == 0]

    plt.figure(figsize=(10, 5))
    plt.hist(pos_preds, bins=50, alpha=0.5,
             label='Positive communications', color='green')
    plt.hist(neg_preds, bins=50, alpha=0.5,
             label='Negative communications', color='red')
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Scores')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()

    print(f"Positive — Mean: {pos_preds.mean():.4f}, Std: {pos_preds.std():.4f}")
    print(f"Negative — Mean: {neg_preds.mean():.4f}, Std: {neg_preds.std():.4f}")

    return pos_preds, neg_preds


def plot_training_history(history, total_epochs=500,
                          interpolate_val=True, save_path=None):
    """
    Three-panel training history figure:
    (A) total loss, (B) ROC-AUC, (C) Average Precision.
    Train and validation curves are shown for B and C.
    """
    actual_epochs = len(history['total_loss'])
    loss_x        = np.arange(1, actual_epochs + 1)
    n_pts         = len(history['train_roc_cell'])
    metric_x      = np.linspace(1, actual_epochs, n_pts)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(bottom=0.15, wspace=0.35)

    axes[0].plot(loss_x, history['total_loss'], color='tab:blue')
    axes[0].set_xlabel('Epoch', fontsize=12, labelpad=8)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_title('(A) Training Loss', fontsize=12, pad=10)
    axes[0].tick_params(labelsize=10)

    def _plot(ax, key, ylabel, label):
        if interpolate_val:
            tr = np.interp(loss_x, metric_x, history[f'train_{key}'])
            vl = np.interp(loss_x, metric_x, history[f'val_{key}'])
            ax.plot(loss_x, tr, label='Train', color='tab:blue')
            ax.plot(loss_x, vl, label='Val',   color='tab:orange')
        else:
            ax.plot(metric_x, history[f'train_{key}'], label='Train')
            ax.plot(metric_x, history[f'val_{key}'],   label='Val')
        ax.set_xlabel('Epoch', fontsize=12, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(label, fontsize=12, pad=10)
        ax.tick_params(labelsize=10)

    _plot(axes[1], 'roc_cell', 'ROC-AUC',          '(B) ROC-AUC')
    _plot(axes[2], 'ap_cell',  'Average Precision', '(C) Average Precision')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
