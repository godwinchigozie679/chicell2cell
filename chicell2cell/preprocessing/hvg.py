import scanpy as sc

def select_hvg_and_cluster(adata, min_disp=0.5, min_mean=0.0125, max_mean=3.0,
    span=0.3, n_bins=20, flavor="seurat", n_pcs=31, n_neighbors=15,
    leiden_resolution=0.39, plot=True):
    sc.pp.highly_variable_genes(adata, min_disp=min_disp, min_mean=min_mean,
        max_mean=max_mean, span=span, n_bins=n_bins, flavor=flavor)
    if plot:
        sc.pl.highly_variable_genes(adata)
    sc.tl.pca(adata, svd_solver="arpack", zero_center=True, use_highly_variable=True)
    if plot:
        sc.pl.pca_variance_ratio(adata, log=True)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.leiden(adata, resolution=leiden_resolution, key_added="clusters")
    sc.tl.umap(adata)
    if plot:
        sc.pl.umap(adata, color=["clusters"])
    print(f"Leiden clustering done — {adata.obs["clusters"].nunique()} clusters")
    return adata