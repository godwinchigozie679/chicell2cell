"""Cluster-level cell-cell communication with permutation-based significance testing."""

import numpy as np
import pandas as pd
import anndata


def summarize_cluster_spatial(X, attention_matrix, clusterid, clusternames,
                               n_permutations=500):
    """
    Summarise decoder predictions to cluster level using only
    spatially connected cell pairs (attention weight > 0).
    Returns strength, density, total, count, and p-value DataFrames.
    """
    n = len(clusternames)
    comm_strength = np.empty([n, n], float)
    conn_density  = np.empty([n, n], float)
    comm_total    = np.empty([n, n], float)
    conn_count    = np.empty([n, n], int)

    for i in range(n):
        idx_i = np.where(clusterid == clusternames[i])[0]
        for j in range(n):
            idx_j        = np.where(clusterid == clusternames[j])[0]
            comm_values  = X[idx_i, :][:, idx_j]
            attn_values  = attention_matrix[idx_i, :][:, idx_j]
            spatial_mask = attn_values > 0
            n_spatial    = spatial_mask.sum()
            total_pairs  = comm_values.size

            if n_spatial > 0:
                comm_strength[i, j] = comm_values[spatial_mask].mean()
                conn_density[i, j]  = n_spatial / total_pairs
                comm_total[i, j]    = comm_strength[i, j] * conn_density[i, j]
                conn_count[i, j]    = n_spatial
            else:
                comm_strength[i, j] = 0
                conn_density[i, j]  = 0
                comm_total[i, j]    = 0
                conn_count[i, j]    = 0

    # Permutation test — shuffle cell type labels and recompute strength
    print(f"  Running {n_permutations} permutations...")
    p_cluster = np.zeros([n, n], float)

    for perm in range(n_permutations):
        if (perm + 1) % 100 == 0:
            print(f"  Permutation {perm + 1}/{n_permutations}", end='\r')
        perm_id = np.random.permutation(clusterid)
        cs_perm = np.empty([n, n], float)
        for j in range(n):
            idx_j = np.where(perm_id == clusternames[j])[0]
            for k in range(n):
                idx_k = np.where(perm_id == clusternames[k])[0]
                cv    = X[idx_j, :][:, idx_k]
                av    = attention_matrix[idx_j, :][:, idx_k]
                m     = av > 0
                cs_perm[j, k] = cv[m].mean() if m.sum() > 0 else 0
        p_cluster[cs_perm >= comm_strength] += 1.0

    print()
    p_cluster /= n_permutations

    mk = dict(index=clusternames, columns=clusternames)
    return (pd.DataFrame(comm_strength, **mk),
            pd.DataFrame(conn_density,  **mk),
            pd.DataFrame(comm_total,    **mk),
            pd.DataFrame(conn_count,    **mk),
            pd.DataFrame(p_cluster,     **mk))


def cluster_communication(
    adata: anndata.AnnData,
    decoder_pred: np.ndarray,
    attention_weights: np.ndarray,
    edge_index_att: np.ndarray,
    database_name: str = 'chicell2cell',
    clustering: str = 'cell_type',
    attention_threshold: float = 0.0,
    n_permutations: int = 500,
    random_seed: int = 42,
    copy: bool = False,
):
    """
    Infer cluster-level CCC by aggregating decoder predictions over
    spatially connected cell pairs. Significance is assessed via
    permutation testing on cell type labels.

    Requires 'cell_type' in adata.obs.
    """
    print("=" * 80)
    print("CLUSTER COMMUNICATION")
    print("=" * 80)

    if clustering not in adata.obs.columns:
        raise KeyError(
            f"Column '{clustering}' not found in adata.obs. "
            f"Run annotate_cell_types() before cluster_communication()."
        )

    np.random.seed(random_seed)
    n_cells   = decoder_pred.shape[0]
    celltypes = sorted([str(ct) for ct in adata.obs[clustering].unique()])
    clusterid = np.array(adata.obs[clustering], str)

    print(f"\n  Cells      : {n_cells}")
    print(f"  Cell types : {celltypes}")

    # Build dense attention matrix from sparse edge representation
    attention_matrix = np.zeros((n_cells, n_cells))
    for idx in range(edge_index_att.shape[1]):
        attention_matrix[edge_index_att[0, idx], edge_index_att[1, idx]] = \
            attention_weights[idx]

    # Filter to spatially connected cross-type pairs only
    X_filt    = decoder_pred.copy()
    attn_filt = attention_matrix.copy()
    for i in range(n_cells):
        for j in range(n_cells):
            if not (attention_matrix[i, j] > attention_threshold
                    and clusterid[i] != clusterid[j]):
                X_filt[i, j]    = 0
                attn_filt[i, j] = 0

    print(f"  Direct spatial edges (attn > {attention_threshold}): "
          f"{(X_filt > 0).sum():,}")

    obsp_key = f'{database_name}-total-total'
    adata.obsp[obsp_key]                     = X_filt
    adata.obsp[f'{database_name}-attention'] = attn_filt

    df_strength, df_density, df_total, df_count, df_p = summarize_cluster_spatial(
        X_filt, attn_filt, clusterid, celltypes,
        n_permutations=n_permutations)

    uns_key = f'{database_name}_cluster-{clustering}-total-total'
    adata.uns[uns_key] = {
        'communication_strength': df_strength,
        'connection_density':     df_density,
        'communication_total':    df_total,
        'connection_count':       df_count,
        'communication_pvalue':   df_p,
    }

    # Build summary DataFrame of significant pairs
    rows = []
    for src in celltypes:
        for tgt in celltypes:
            s = df_strength.loc[src, tgt]
            if s > 0:
                rows.append({
                    'source':        src,
                    'target':        tgt,
                    'comm_strength': s,
                    'conn_density':  df_density.loc[src, tgt],
                    'comm_total':    df_total.loc[src, tgt],
                    'n_connections': int(df_count.loc[src, tgt]),
                    'p_value':       df_p.loc[src, tgt],
                    'significant':   df_p.loc[src, tgt] < 0.05,
                })

    scores_df = (pd.DataFrame(rows)
                 .sort_values('comm_strength', ascending=False)
                 .reset_index(drop=True))
    adata.uns[uns_key]['communication_scores'] = scores_df

    n_sig = scores_df['significant'].sum()
    print(f"\n  Total pairs with connections : {len(scores_df)}")
    print(f"  Significant (p < 0.05)       : {n_sig}")
    print(f"\n  Top 10:")
    print(scores_df.head(10).to_string(index=False))

    return (adata, scores_df) if copy else scores_df


def get_cluster_communication_network(
    adata, uns_names, clustering='cell_type',
    metric='strength', p_value_cutoff=0.05,
    self_communication_off=True,
):
    """Extract significant cluster-level communications filtered by p-value."""
    metric_map = {
        'strength': 'communication_strength',
        'density':  'connection_density',
        'total':    'communication_total',
    }
    mkey   = metric_map.get(metric, 'communication_strength')
    base   = adata.uns[uns_names[0]][mkey].copy()
    labels = list(base.columns)
    X      = np.zeros_like(base.values, float)

    for key in uns_names:
        vals = adata.uns[key][mkey].values.copy()
        pvs  = adata.uns[key]['communication_pvalue'].values.copy()
        vals[pvs > p_value_cutoff] = 0
        X += vals
    X /= len(uns_names)

    if self_communication_off:
        np.fill_diagonal(X, 0)

    rows = []
    for i, src in enumerate(labels):
        for j, tgt in enumerate(labels):
            if X[i, j] > 0:
                rows.append({
                    'source':        src,
                    'target':        tgt,
                    'comm_strength': adata.uns[uns_names[0]]['communication_strength'].iloc[i, j],
                    'conn_density':  adata.uns[uns_names[0]]['connection_density'].iloc[i, j],
                    'comm_total':    adata.uns[uns_names[0]]['communication_total'].iloc[i, j],
                    'n_connections': int(adata.uns[uns_names[0]]['connection_count'].iloc[i, j]),
                    'p_value':       adata.uns[uns_names[0]]['communication_pvalue'].iloc[i, j],
                })

    return pd.DataFrame(rows).sort_values('comm_strength', ascending=False)
