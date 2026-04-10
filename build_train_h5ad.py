import scanpy as sc
import scipy.io
import scipy.sparse as sp
import pandas as pd
import anndata as ad
import numpy as np
import os

TIMEPOINTS = {
    0: {
        "matrix":   "/projectnb/ds596/projects/Team 8/data/additional_dataset/GSM3318004_P0_matrix.mtx.gz",
        "barcodes": "/projectnb/ds596/projects/Team 8/data/additional_dataset/GSM3318004_P0_barcodes.tsv.gz",
        "genes":    "/projectnb/ds596/projects/Team 8/data/additional_dataset/GSM3318004_P0_genes.tsv.gz",
    },
    14: {
        "matrix":   "/projectnb/ds596/projects/Team 8/data/additional_dataset/GSM3318007_P14_matrix.mtx.gz",
        "barcodes": "/projectnb/ds596/projects/Team 8/data/additional_dataset/GSM3318007_P14_barcodes.tsv.gz",
        "genes":    "/projectnb/ds596/projects/Team 8/data/additional_dataset/GSM3318007_P14_genes.tsv.gz",
    }
}

N_TOP_GENES   = 200    
MIN_GENES     = 200   
MIN_CELLS     = 3      
OUTPUT_TRAIN  = "squidiff_train.h5ad"
OUTPUT_TEST   = "squidiff_test.h5ad"   
TEST_FRACTION = 0.1                

def load_10x(matrix_path, barcodes_path, genes_path, day_label):

    matrix   = scipy.io.mmread(matrix_path).T.tocsr()
    barcodes = pd.read_csv(barcodes_path, header=None, names=["barcode"])
    genes    = pd.read_csv(genes_path, header=None, sep="\t")

    if genes.shape[1] >= 2:
        genes.columns = ["gene_id", "gene_name"] + list(genes.columns[2:])
    else:
        genes.columns = ["gene_name"]
        genes["gene_id"] = genes["gene_name"]

    adata = ad.AnnData(
        X   = matrix,
        obs = pd.DataFrame(index=barcodes["barcode"]),
        var = genes.set_index("gene_id")
    )

    # Unique barcodes across timepoints
    adata.obs_names = [f"day{day_label}_{bc}" for bc in adata.obs_names]

    # Timepoint label — key column SquiDiff uses for trajectory direction
    adata.obs["Group"] = str(day_label)   # required by SquiDiff
    adata.obs["day"]   = str(day_label)

    print(f"    → {adata.n_obs} cells, {adata.n_vars} genes")
    return adata

adatas = []
for day, paths in sorted(TIMEPOINTS.items()):
    a = load_10x(paths["matrix"], paths["barcodes"], paths["genes"], day_label=day)
    adatas.append(a)

adata = ad.concat(adatas, join="inner", merge="same")
adata.var_names_make_unique()

print(f"Combined: {adata.n_obs} cells × {adata.n_vars} genes")
print("Cells per timepoint:")
print(adata.obs["day"].value_counts().sort_index())

sc.pp.filter_cells(adata, min_genes=MIN_GENES)
sc.pp.filter_genes(adata, min_cells=MIN_CELLS)

# Mitochondrial QC
adata.var["mt"] = adata.var["gene_name"].str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# Remove cells with >20% mitochondrial reads
before = adata.n_obs
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
print(f"Removed {before - adata.n_obs} high-mito cells (>20% MT reads)")
print(f"After QC: {adata.n_obs} cells × {adata.n_vars} genes")

# Store raw counts before normalisation
adata.layers["counts"] = adata.X.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

print("Normalisation complete")

sc.pp.highly_variable_genes(
    adata,
    n_top_genes = N_TOP_GENES,
    batch_key   = "Group",      
    flavor      = "seurat",
)

adata = adata[:, adata.var["highly_variable"]].copy()
print(f"Selected {adata.n_vars} highly variable genes")

if sp.issparse(adata.X):
    adata.X = adata.X.toarray()

adata.X = adata.X.astype(np.float32)
print(f"Matrix dtype: {adata.X.dtype}, shape: {adata.X.shape}")

earliest_day = str(sorted(TIMEPOINTS.keys())[0])
latest_day   = str(sorted(TIMEPOINTS.keys())[-1])

source_mask = adata.obs["Group"] == earliest_day   # was "day"
source_idx  = np.where(source_mask)[0]

np.random.seed(42)
n_test      = max(1, int(len(source_idx) * TEST_FRACTION))
test_idx    = np.random.choice(source_idx, size=n_test, replace=False)
train_idx   = np.array([i for i in range(adata.n_obs) if i not in set(test_idx)])

adata_train = adata[train_idx].copy()
adata_test  = adata[test_idx].copy()

adata_train.write_h5ad(OUTPUT_TRAIN)
adata_test.write_h5ad(OUTPUT_TEST)

print(f"Training file : {OUTPUT_TRAIN}  ({adata_train.n_obs} cells × {adata_train.n_vars} genes)")
print(f"Test file     : {OUTPUT_TEST}   ({adata_test.n_obs} cells × {adata_test.n_vars} genes)")
