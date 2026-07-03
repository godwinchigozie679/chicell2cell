"""Gene-level adjacency graph from ligand-receptor pair database."""

import numpy as np
import pandas as pd


def load_geneadj(adata_filtered, lr_pairs):
    """
    Build a directed gene adjacency matrix where entry [ligand, receptor] = 1
    for each LR pair present in both the database and the filtered adata.

    Parameters
    ----------
    adata_filtered : AnnData already subset to LR genes
    lr_pairs       : DataFrame with columns 'ligand' and 'receptor'

    Returns
    -------
    gene_adj   : pd.DataFrame (genes x genes, float32)
    edge_count : int
    """
    adata_filtered.var_names_make_unique()
    gene_list = list(adata_filtered.var_names)

    print(f"Genes in filtered adata : {len(gene_list)}")
    print(f"LR pairs                : {len(lr_pairs)}")

    gene_adj   = pd.DataFrame(0, index=gene_list,
                               columns=gene_list, dtype=np.float32)
    edge_count = 0

    for _, row in lr_pairs.iterrows():
        lig, rec = row['ligand'], row['receptor']
        if lig in gene_list and rec in gene_list:
            gene_adj.loc[lig, rec] = 1.0
            edge_count += 1

    print(f"Edges created : {edge_count}")
    return gene_adj, edge_count
