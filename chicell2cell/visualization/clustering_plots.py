"""Spectral clustering analysis and 7-panel visualisation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, csgraph
from scipy.sparse.linalg import eigsh


def spectral_clustering_analysis(adata, adj_reconstructed, k, random_state=42):
    """
    Perform spectral clustering on the decoder-reconstructed adjacency matrix.
    Computes the unnormalised graph Laplacian, extracts the k smallest
    eigenvectors via LOBPCG, normalises to unit norm, then clusters.
    Cluster labels are stored in adata.obs['spectral_cluster'].
    """
    x_sparse  = csr_matrix(adj_reconstructed, dtype=np.float64)
    laplacian = csgraph.laplacian(x_sparse, normed=False)
    _, eigvecs = eigsh(laplacian, k=k, which='SM')
    embeddings = normalize(eigvecs)

    sc     = SpectralClustering(n_clusters=k, affinity='precomputed',
                                random_state=random_state)
    labels = sc.fit_predict(x_sparse.toarray())
    str_labels = [str(l) for l in labels]

    adata.obs['spectral_cluster'] = pd.Categorical(
        str_labels, categories=sorted(set(str_labels), key=int))
    return adata, embeddings


def plot_seven_panels(adata, embeddings, save_path=None):
    """
    Seven-panel figure summarising spectral clustering results:
    A — spatial distribution of spectral clusters
    B — spatial distribution of annotated cell types
    C — 2D spectral embedding (eigenvectors 1 and 2)
    D — 3D spectral embedding (eigenvectors 1, 2, 3)
    E — confusion matrix: spectral clusters vs cell types
    F — cluster size distribution
    G — cell type composition pie chart
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    cell_types  = sorted(adata.obs['cell_type'].unique()) \
        if 'cell_type' in adata.obs.columns else []
    type_colors = plt.cm.tab10(np.arange(len(cell_types)))
    clust_int   = adata.obs['spectral_cluster'].astype(int)

    fig = plt.figure(figsize=(20, 28))
    gs  = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.5)

    def _scatter_spatial(ax, col, labels, colors, title):
        coords = adata.obsm['spatial']
        for idx, label in enumerate(labels):
            mask = adata.obs[col] == label
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[colors[idx]], s=40, alpha=0.7,
                       edgecolors='none', label=str(label))
        ax.set_xlabel('Spatial X', fontsize=13)
        ax.set_ylabel('Spatial Y', fontsize=13)
        ax.set_title(title, fontsize=16, fontweight='bold', loc='left')
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)

    # A — spectral cluster spatial map
    ax1 = fig.add_subplot(gs[0, 0])
    unique_clusters = sorted(adata.obs['spectral_cluster'].unique(), key=int)
    _scatter_spatial(ax1, 'spectral_cluster', unique_clusters,
                     [plt.cm.tab10(int(c)) for c in unique_clusters], 'A')

    # B — annotated cell type spatial map
    ax2 = fig.add_subplot(gs[0, 1])
    if 'cell_type' in adata.obs.columns:
        _scatter_spatial(ax2, 'cell_type', cell_types,
                         list(type_colors), 'B')

    # C — 2D spectral embedding
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(embeddings[:, 0], embeddings[:, 1],
                c=clust_int, cmap='tab10', s=35, alpha=0.6, edgecolors='none')
    ax3.set_xlabel('Eigenvector 1')
    ax3.set_ylabel('Eigenvector 2')
    ax3.set_title('C', fontsize=16, fontweight='bold', loc='left')
    ax3.spines[['top', 'right']].set_visible(False)

    # D — 3D spectral embedding
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    ax4.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                c=clust_int, cmap='tab10', s=20, alpha=0.6, edgecolors='none')
    ax4.set_xlabel('Eigenvector 1')
    ax4.set_ylabel('Eigenvector 2')
    ax4.set_zlabel('Eigenvector 3')
    ax4.set_title('D', fontsize=16, fontweight='bold', loc='left')
    ax4.view_init(elev=20, azim=45)

    # E — confusion matrix
    ax5 = fig.add_subplot(gs[2, 0])
    if 'cell_type' in adata.obs.columns:
        conf = pd.crosstab(adata.obs['spectral_cluster'], adata.obs['cell_type'])
        sns.heatmap(conf, annot=True, fmt='d', cmap='YlOrRd',
                    ax=ax5, linewidths=0.5, linecolor='white',
                    annot_kws={'fontsize': 11, 'fontweight': 'bold'})
        ax5.set_xlabel('Known Cell Type')
        ax5.set_ylabel('Spectral Cluster')
        ax5.set_title('E', fontsize=16, fontweight='bold', loc='left')
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')

    # F — cluster size bar chart
    ax6 = fig.add_subplot(gs[2, 1])
    counts = adata.obs['spectral_cluster'].value_counts().sort_index()
    bars   = ax6.bar(counts.index.astype(int), counts.values,
                     color=plt.cm.tab10(np.arange(len(counts))),
                     edgecolor='black', alpha=0.8)
    for bar in bars:
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(int(bar.get_height())),
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax6.set_xlabel('Cluster ID')
    ax6.set_ylabel('Number of Cells')
    ax6.set_title('F', fontsize=16, fontweight='bold', loc='left')
    ax6.spines[['top', 'right']].set_visible(False)

    # G — cell type pie chart
    ax7 = fig.add_subplot(gs[3, :])
    if 'cell_type' in adata.obs.columns:
        ct_counts  = adata.obs['cell_type'].value_counts()
        pie_colors = [type_colors[cell_types.index(ct)]
                      for ct in ct_counts.index]
        wedges, _ = ax7.pie(ct_counts.values, labels=None,
                             colors=pie_colors, startangle=90,
                             wedgeprops=dict(linewidth=2, edgecolor='white'))
        ax7.legend(
            wedges,
            [f'{ct}  ({v / ct_counts.sum() * 100:.1f}%)'
             for ct, v in zip(ct_counts.index, ct_counts.values)],
            loc='center left', bbox_to_anchor=(1.02, 0.5),
            fontsize=11, frameon=True,
        )
        ax7.set_title('G', fontsize=16, fontweight='bold', loc='left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
