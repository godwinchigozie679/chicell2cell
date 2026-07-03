"""Dual-graph GAT-VAE model for cell-cell communication inference."""

import torch.nn as nn
from .config import FLAGS
from .layers import GATVAELayer, Decoder, ExpRec
from .data_utils import apply_edge_dropout


class CELL2CELLGATVAE(nn.Module):
    """
    Dual-graph GAT-VAE model.
    Encodes both the spatial cell graph and the ligand-receptor gene graph
    using separate GAT-VAE encoders, decodes each with a bilinear decoder,
    and reconstructs gene expression via linear projection heads.
    """

    def __init__(self, cell_features_dim, gene_features_dim,
                 n_cells, n_genes, hidden_dim=32, latent_dim=32):
        super().__init__()
        self.cell_encoder = GATVAELayer(cell_features_dim, hidden_dim, latent_dim,
                                        FLAGS.num_heads, FLAGS.dropout)
        self.gene_encoder = GATVAELayer(gene_features_dim, hidden_dim, latent_dim,
                                        FLAGS.num_heads, FLAGS.dropout)
        self.cell_decoder = Decoder(latent_dim, FLAGS.dropout)
        self.gene_decoder = Decoder(latent_dim, FLAGS.dropout)
        # Expression reconstruction heads
        self.exp_rec_cell = ExpRec(n_genes,  latent_dim, FLAGS.dropout)
        self.exp_rec_gene = ExpRec(n_cells, latent_dim, FLAGS.dropout)

    def forward(self, cell_features, cell_edge_index,
                gene_features, gene_edge_index):
        # Apply edge dropout during training for regularisation
        if self.training and FLAGS.edge_dropout > 0:
            cell_edge_index = apply_edge_dropout(cell_edge_index, FLAGS.edge_dropout)
            gene_edge_index = apply_edge_dropout(gene_edge_index, FLAGS.edge_dropout)

        z_cell, z_mean_cell, z_log_std_cell = self.cell_encoder(
            cell_features, cell_edge_index)
        z_gene, z_mean_gene, z_log_std_gene = self.gene_encoder(
            gene_features, gene_edge_index)

        return {
            'z_cell': z_cell, 'z_mean_cell': z_mean_cell,
            'z_log_std_cell': z_log_std_cell,
            'z_gene': z_gene, 'z_mean_gene': z_mean_gene,
            'z_log_std_gene': z_log_std_gene,
            'adj_pred_cell': self.cell_decoder(z_cell),
            'adj_pred_gene': self.gene_decoder(z_gene),
            'exp_rec_cell':  self.exp_rec_cell(z_mean_cell),
            'exp_rec_gene':  self.exp_rec_gene(z_mean_gene),
        }
