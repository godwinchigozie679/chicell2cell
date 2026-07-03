"""Spatial plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_spatial_graph(adata, cell_adj, figsize=(8, 6)):
    coords = adata.obsm['spatial']
    nc = cell_adj.sum(axis=1)
    iso = nc == 0

    plt.figure(figsize=figsize)
    if (~iso).any():
        plt.scatter(coords[~iso, 0], coords[~iso, 1],
                    c='blue', s=30, alpha=0.7,
                    label=f'Connected ({(~iso).sum()})')
    if iso.any():
        plt.scatter(coords[iso, 0], coords[iso, 1],
                    c='red', s=50, alpha=0.9, marker='X',
                    label=f'Isolated ({iso.sum()})')
    for i in range(len(cell_adj)):
        for j in range(i + 1, len(cell_adj)):
            if cell_adj[i, j]:
                plt.plot([coords[i, 0], coords[j, 0]],
                         [coords[i, 1], coords[j, 1]],
                         'gray', alpha=0.3, linewidth=0.5)
    plt.title('Spatial Cell Graph')
    plt.xlabel('Spatial X'); plt.ylabel('Spatial Y')
    plt.legend(); plt.axis('equal'); plt.tight_layout(); plt.show()


def plot_spatial_celltypes(adata, clustering='cell_type', figsize=(10, 8)):
    coords     = adata.obsm['spatial']
    cell_types = sorted(adata.obs[clustering].unique())
    colors     = plt.cm.tab10(np.arange(len(cell_types)))

    plt.figure(figsize=figsize)
    for idx, ct in enumerate(cell_types):
        mask = adata.obs[clustering] == ct
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    c=[colors[idx]], s=40, alpha=0.7,
                    edgecolors='none', label=ct)
    plt.xlabel('Spatial X'); plt.ylabel('Spatial Y')
    plt.title('Spatial Distribution of Cell Types')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('equal'); plt.tight_layout(); plt.show()
