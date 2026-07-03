import scanpy as sc

def normalize_and_log(adata, target_sum=1e4):
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    print(f"Normalised to {target_sum:.0f} counts/cell and log1p-transformed.")
    return adata