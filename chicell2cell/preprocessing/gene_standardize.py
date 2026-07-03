"""
HGNC gene-name standardisation and LR gene filtering.
Preserves cell_type and all obs columns when subsetting.
"""

import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')


def standardize_gene_names(adata, species='human', output_path=None):
    """
    Map gene names to official HGNC symbols via MyGeneInfo.
    All .obs columns (including cell_type) are preserved.
    """
    try:
        import mygene
    except ImportError:
        raise ImportError("Install mygene: pip install mygene")

    mg = mygene.MyGeneInfo()
    original_genes = list(adata.var_names)
    print(f"Standardising {len(original_genes)} gene names...")

    results = mg.querymany(
        original_genes,
        scopes='symbol,alias,ensembl.gene',
        fields='symbol',
        species=species,
        returnall=True,
        verbose=False,
    )

    gene_mapping = {}
    unmapped = []
    for i, orig in enumerate(original_genes):
        r = results['out'][i] if i < len(results['out']) else {}
        if isinstance(r, dict) and 'symbol' in r:
            gene_mapping[orig] = r['symbol']
        else:
            gene_mapping[orig] = None
            unmapped.append(orig)

    # Retry unmapped
    if unmapped:
        retry = [g.split('.')[0] for g in unmapped]
        retry_results = mg.querymany(
            retry,
            scopes='ensembl.gene,symbol,alias',
            fields='symbol,name',
            species=species,
            returnall=True,
            verbose=False,
        )
        for i, orig in enumerate(unmapped):
            r = retry_results['out'][i] if i < len(retry_results['out']) else {}
            if isinstance(r, dict) and 'symbol' in r:
                gene_mapping[orig] = r['symbol']

    # Resolve duplicates
    symbol_to_originals = {}
    for orig, sym in gene_mapping.items():
        if sym is not None:
            symbol_to_originals.setdefault(sym, []).append(orig)

    final_mapping = {}
    for orig, sym in gene_mapping.items():
        if sym is None:
            final_mapping[orig] = None
        elif len(symbol_to_originals[sym]) == 1:
            final_mapping[orig] = sym
        else:
            final_mapping[orig] = sym if orig == sym else None

    genes_to_keep = [
        i for i, g in enumerate(original_genes)
        if final_mapping[g] is not None
    ]
    new_names = [final_mapping[original_genes[i]] for i in genes_to_keep]

    adata_std = adata[:, genes_to_keep].copy()
    adata_std.var_names = new_names
    adata_std.var_names_make_unique()
    adata_std.var['original_gene_name'] = [original_genes[i] for i in genes_to_keep]

    print(f"Original: {len(original_genes)} → Standardised: {len(genes_to_keep)} genes")
    print(f"Cell type preserved: {'cell_type' in adata_std.obs.columns}")

    if output_path:
        adata_std.write(output_path)
        print(f"Saved to {output_path}")

    return adata_std


def filter_lr_genes(adata, lr_pairs, output_path=None):
    """
    Subset AnnData to genes present in the LR database.
    IMPORTANT: All .obs columns including cell_type are preserved.

    Parameters
    ----------
    adata    : AnnData (HGNC-standardised, must have 'cell_type' in .obs)
    lr_pairs : DataFrame with columns 'ligand' and 'receptor'

    Returns
    -------
    adata_lr    : AnnData subset to LR genes only
    lr_filtered : DataFrame of LR pairs where both genes exist in adata
    """
    # Check cell_type exists before filtering
    if 'cell_type' not in adata.obs.columns:
        raise ValueError(
            "'cell_type' column not found in adata.obs. "
            "Run annotate_cell_types() before filter_lr_genes()."
        )

    genes = set(adata.var_names)
    lr_filtered = lr_pairs[
        lr_pairs['ligand'].isin(genes) &
        lr_pairs['receptor'].isin(genes)
    ].copy()

    lr_genes = set(lr_filtered['ligand']) | set(lr_filtered['receptor'])
    keep_genes = [g for g in adata.var_names if g in lr_genes]

    adata_lr = adata[:, keep_genes].copy()

    # Verify cell_type survived the subsetting
    assert 'cell_type' in adata_lr.obs.columns, \
        "cell_type was lost during subsetting — this should not happen."

    print(f"LR filtering: {adata.shape} → {adata_lr.shape}")
    print(f"LR pairs:     {len(lr_pairs)} → {len(lr_filtered)}")
    print(f"cell_type preserved: {adata_lr.obs['cell_type'].notna().sum()} / {adata_lr.n_obs} cells")
    print(f"Cell types: {sorted(adata_lr.obs['cell_type'].unique())}")

    if output_path:
        adata_lr.write(output_path)
        print(f"Saved to {output_path}")

    return adata_lr, lr_filtered
