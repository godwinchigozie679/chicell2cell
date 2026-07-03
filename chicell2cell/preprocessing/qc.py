import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

def run_qc(adata, plot=True):
    adata.var["mito"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], inplace=True)
    if plot:
        sc.pl.violin(adata, ["n_genes_by_counts","total_counts","pct_counts_mito"], jitter=0.4, multi_panel=True)
        sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mito")
    return adata

def filter_cells_genes(adata, min_genes=500, min_cells=3, max_pct_mito=10.0, plot=True):
    if plot:
        fig, axes = plt.subplots(1,2,figsize=(14,5))
        axes[0].hist(adata.obs["n_genes_by_counts"], bins=50, color="blue", alpha=0.7)
        axes[0].set_title("Before: Genes per Cell")
        axes[1].hist(adata.var["n_cells_by_counts"], bins=50, color="blue", alpha=0.7)
        axes[1].set_title("Before: Cells per Gene")
        plt.tight_layout(); plt.show()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata.obs["percent_mito"] = (adata[:, adata.var["mito"]].X.sum(axis=1) / adata.X.sum(axis=1)) * 100
    adata = adata[adata.obs["percent_mito"] < max_pct_mito, :]
    if plot:
        fig, axes = plt.subplots(1,2,figsize=(14,5))
        axes[0].hist(adata.obs["n_genes_by_counts"], bins=50, color="green", alpha=0.7)
        axes[0].set_title("After: Genes per Cell")
        axes[1].hist(adata.var["n_cells_by_counts"], bins=50, color="green", alpha=0.7)
        axes[1].set_title("After: Cells per Gene")
        plt.tight_layout(); plt.show()
    print(f"After filtering: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata

def remove_doublets(adata, threshold=0.25):
    try:
        import scrublet as scr
    except ImportError:
        raise ImportError("Install scrublet: pip install scrublet")
    scrub = scr.Scrublet(adata.X)
    doublet_scores, predicted_doublets = scrub.scrub_doublets()
    if predicted_doublets is None:
        predicted_doublets = scrub.call_doublets(threshold=threshold)
    adata.obs["doublet_score"] = doublet_scores.astype(float)
    adata.obs["predicted_doublet"] = predicted_doublets.astype(bool)
    n = predicted_doublets.sum()
    print(f"Detected {n} doublets ({n/len(predicted_doublets)*100:.1f}%)")
    return adata