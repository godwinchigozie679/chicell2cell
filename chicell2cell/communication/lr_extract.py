"""Ligand-receptor interaction extraction for significant cell type pairs."""

import numpy as np
import pandas as pd
import anndata


def extract_significant_lr_communications(
    adata: anndata.AnnData,
    lr_pairs: pd.DataFrame,
    attention_weights: np.ndarray,
    edge_index_att: np.ndarray,
    uns_key: str = 'chicell2cell_cluster-cell_type-total-total',
    clustering: str = 'cell_type',
    p_value_cutoff: float = 0.05,
):
    """
    For each statistically significant cell type pair (p < cutoff),
    extract all LR interactions and compute mean ligand and receptor
    expression in the source and target cell types respectively.
    """
    print("=" * 80)
    print("EXTRACTING SIGNIFICANT L-R COMMUNICATIONS")
    print("=" * 80)

    comm_matrix = adata.uns[uns_key]['communication_strength']
    p_value_mat = adata.uns[uns_key]['communication_pvalue']
    obsp_key    = uns_key.split('_cluster-')[0] + '-total-total'
    X_filtered  = adata.obsp[obsp_key]

    # Rebuild attention matrix from edge index
    n_cells = adata.n_obs
    attn_matrix = np.zeros((n_cells, n_cells))
    for idx in range(edge_index_att.shape[1]):
        attn_matrix[edge_index_att[0, idx], edge_index_att[1, idx]] = \
            attention_weights[idx]

    # Identify significant cell type pairs
    sig_pairs = [
        {'source_type': src, 'target_type': tgt,
         'celltype_comm_score': comm_matrix.loc[src, tgt],
         'celltype_p_value':    p_value_mat.loc[src, tgt]}
        for src in comm_matrix.index
        for tgt in comm_matrix.columns
        if p_value_mat.loc[src, tgt] < p_value_cutoff
        and comm_matrix.loc[src, tgt] > 0
    ]

    print(f"  Significant cell-type pairs (p < {p_value_cutoff}): {len(sig_pairs)}")

    if not sig_pairs:
        print("  No significant pairs found.")
        return pd.DataFrame()

    cell_types  = adata.obs[clustering].values
    expr_matrix = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    gene_names  = list(adata.var_names)
    g2i         = {g: i for i, g in enumerate(gene_names)}

    results = []
    for pair in sig_pairs:
        src_ct  = pair['source_type']
        tgt_ct  = pair['target_type']
        src_idx = np.where(cell_types == src_ct)[0]
        tgt_idx = np.where(cell_types == tgt_ct)[0]

        # Average decoder score across all spatially connected pairs
        comms = [X_filtered[i, j]
                 for i in src_idx for j in tgt_idx
                 if X_filtered[i, j] > 0]
        attns = [attn_matrix[i, j]
                 for i in src_idx for j in tgt_idx
                 if X_filtered[i, j] > 0]

        if not comms:
            continue

        avg_comm = float(np.mean(comms))
        avg_attn = float(np.mean(attns))

        for _, lr_row in lr_pairs.iterrows():
            lig, rec = lr_row['ligand'], lr_row['receptor']
            if lig not in g2i or rec not in g2i:
                continue
            results.append({
                'source':              src_ct,
                'target':              tgt_ct,
                'ligand':              lig,
                'receptor':            rec,
                'ligand_expression':   float(expr_matrix[src_idx, g2i[lig]].mean()),
                'receptor_expression': float(expr_matrix[tgt_idx, g2i[rec]].mean()),
                'communication_score': avg_comm,
                'attn_score':          avg_attn,
                'p_value':             float(pair['celltype_p_value']),
                'pathway_name':        lr_row.get('pathway_name', 'Unknown'),
                'annotation':          lr_row.get('annotation', 'Unknown'),
                'evidence':            lr_row.get('evidence', 'Unknown'),
            })

    lr_comms_df = (pd.DataFrame(results)
                   .sort_values(['communication_score', 'attn_score'],
                                ascending=False)
                   .reset_index(drop=True))

    print(f"  Total L-R communications : {len(lr_comms_df):,}")
    print(f"  Unique cell-type pairs   : "
          f"{lr_comms_df[['source','target']].drop_duplicates().shape[0]}")
    print(f"  Unique L-R pairs         : "
          f"{lr_comms_df[['ligand','receptor']].drop_duplicates().shape[0]}")

    return lr_comms_df
