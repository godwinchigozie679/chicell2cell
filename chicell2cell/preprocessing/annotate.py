"""
Cell type annotation from cluster-to-cell-type mapping.

This is the critical step that adds 'cell_type' to adata.obs
so it is available in all downstream analysis including
adata_lr_filtered used for communication inference.
"""

import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import rcParams


# Default GBM marker genes from Neftel et al. 2019 and Liu et al. 2024
DEFAULT_MARKER_GENES = {
    'NPC-like-Tumor':       ['SOX4', 'SOX11', 'ASCL1', 'HES1'],
    'OPC-like-tumor':       ['OLIG1', 'PDGFRA', 'CSPG4', 'PLLP', 'TNR'],
    'AC-like':              ['MLC1', 'SLC15A2'],
    'MES-like':             ['CHI3L1', 'FN1', 'TGFBI', 'SERPINE1', 'C3', 'HILPDA', 'DDIT3'],
    'Microglia/Macrophages':['CD68', 'AIF1', 'CX3CR1', 'TMEM119', 'C1QA', 'TYROBP'],
    'Inhibition_neurons':   ['SST', 'NPY', 'CALB2', 'CCK'],
    'Endothelial':          ['PECAM1', 'CDH5', 'VWF', 'CLDN5', 'FLT1', 'KDR', 'TIE1', 'TEK', 'ESAM'],
    'Astrocyte':            ['GJA1', 'SLC1A2'],
    'Neurons':              ['MAP2', 'SYN1', 'NEFL', 'SNAP25'],
}

# Default cluster-to-cell-type mapping for GBM dataset
DEFAULT_CLUSTER_MAP = {
    '0': 'NPC-like-Tumor',
    '1': 'Microglia/Macrophages',
    '2': 'OPC-like-tumor',
    '3': 'AC-like',
    '4': 'Endothelial',
    '5': 'Inhibition_neurons',
    '6': 'MES-like',
    '7': 'Astrocyte',
    '8': 'Neurons',
}


def annotate_cell_types(
    adata,
    cluster_to_cell_type=None,
    marker_genes=None,
    clustering_key='clusters',
    plot=True,
):
    """
    Annotate clusters with cell type labels and add 'cell_type' to adata.obs.

    This function MUST be called before filter_lr_genes so that cell_type
    is preserved in adata_lr_filtered for downstream communication analysis.

    Parameters
    ----------
    adata : AnnData
        Must have clustering_key in .obs (e.g. 'clusters' from select_hvg_and_cluster)
    cluster_to_cell_type : dict or None
        Mapping from cluster ID (str) to cell type name.
        If None, uses DEFAULT_CLUSTER_MAP for GBM dataset.
    marker_genes : dict or None
        Marker genes per cell type for heatmap validation.
        If None, uses DEFAULT_MARKER_GENES.
    clustering_key : str
        Column in adata.obs containing cluster labels.
    plot : bool
        Show heatmap, matrixplot, and UMAP.

    Returns
    -------
    adata : AnnData with 'cell_type' in .obs
    """
    if cluster_to_cell_type is None:
        cluster_to_cell_type = DEFAULT_CLUSTER_MAP
        print("Using default GBM cluster-to-cell-type mapping.")

    if marker_genes is None:
        marker_genes = DEFAULT_MARKER_GENES

    if clustering_key not in adata.obs.columns:
        raise ValueError(
            f"Column '{clustering_key}' not found in adata.obs. "
            f"Run select_hvg_and_cluster first."
        )

    if plot:
        # Visualise marker genes across clusters
        available_markers = {
            ct: [g for g in genes if g in adata.var_names]
            for ct, genes in marker_genes.items()
        }
        available_markers = {ct: g for ct, g in available_markers.items() if g}

        if available_markers:
            sc.pl.heatmap(
                adata, available_markers, groupby=clustering_key,
                standard_scale='var', show_gene_labels=True, dendrogram=True,
            )
            sc.pl.matrixplot(
                adata, available_markers, groupby=clustering_key,
                standard_scale='var', dendrogram=True,
            )

    # Apply mapping
    adata.obs['cell_type'] = adata.obs[clustering_key].map(cluster_to_cell_type)

    # Check for unmapped clusters
    n_unmapped = adata.obs['cell_type'].isna().sum()
    if n_unmapped > 0:
        unmapped = adata.obs.loc[
            adata.obs['cell_type'].isna(), clustering_key
        ].unique().tolist()
        print(f"WARNING: {n_unmapped} cells have unmapped clusters: {unmapped}")
        print("Add these clusters to your cluster_to_cell_type mapping.")

    print("\nCell type counts:")
    print(adata.obs['cell_type'].value_counts())

    if plot:
        # Save rcParams
        orig = {k: rcParams[k] for k in
                ['legend.fontsize', 'axes.titlesize', 'axes.labelsize',
                 'xtick.labelsize', 'ytick.labelsize']}
        for k in orig:
            rcParams[k] = 16

        sc.pl.umap(
            adata, color='cell_type',
            title='UMAP — Cell Types', frameon=True,
        )

        for k, v in orig.items():
            rcParams[k] = v

    return adata
