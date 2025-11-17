import h5py
import numpy as np
import pandas as pd
import anndata as ad

# Paths
file_path = "/home/absking/scratch/vcc/state/competition/prediction_test.h5ad"
csv_path = "/home/absking/scratch/vcc/state/competition_support_set/pert_counts_Test.csv"
output_path = "/home/absking/scratch/vcc/state/competition/prediction_test_filtered.h5ad"

# Read allowed genes from CSV
df = pd.read_csv(csv_path)
csv_genes = set(df['target_gene'].tolist())
csv_genes.add('non-targeting')  # Add 'non-targeting' to allowed genes

print(f"Genes from CSV (including 'non-targeting'): {len(csv_genes)}")
print(f"First few CSV genes: {sorted(list(csv_genes))[:5]}")

# Open the h5ad file with h5py and filter
with h5py.File(file_path, "r") as f:
    # Check X structure (dense or sparse)
    if isinstance(f['X'], h5py.Dataset):
        print(f"\nX is a dense matrix: shape {f['X'].shape}, dtype {f['X'].dtype}")
    elif isinstance(f['X'], h5py.Group):
        print(f"\nX is a sparse matrix (CSR format)")
    
    # Get target_gene information
    target_gene_categories = f['obs/target_gene/categories'][:]
    target_gene_codes = f['obs/target_gene/codes'][:]
    
    # Convert codes to actual gene names (these are bytes from h5py)
    target_gene_names = target_gene_categories[target_gene_codes]
    
    # Convert CSV genes to bytes for comparison
    csv_genes_bytes = {g.encode('utf-8') if isinstance(g, str) else g for g in csv_genes}
    
    # Also check what unique genes are actually in the file
    unique_genes_in_file = set(target_gene_names)
    unique_genes_in_file_str = {g.decode() if isinstance(g, bytes) else g for g in unique_genes_in_file}
    
    print(f"\nUnique genes in file: {len(unique_genes_in_file)}")
    print(f"'non-targeting' in file: {'non-targeting' in unique_genes_in_file_str}")
    
    # Create mask for cells to keep - compare bytes to bytes
    keep_mask = np.array([gene in csv_genes_bytes for gene in target_gene_names])
    keep_indices = np.where(keep_mask)[0]
    
    print(f"\nOriginal number of cells: {len(keep_mask)}")
    print(f"Number of cells to keep: {len(keep_indices)}")
    print(f"Number of cells to remove: {len(keep_mask) - len(keep_indices)}")
    
    # Print summary of kept genes
    kept_genes = set(target_gene_names[keep_indices])
    kept_genes_str = {g.decode() if isinstance(g, bytes) else g for g in kept_genes}
    print(f"\nUnique genes in filtered data: {len(kept_genes)}")
    print(f"'non-targeting' in filtered data: {'non-targeting' in kept_genes_str}")
    print(f"Kept genes: {sorted(kept_genes_str)[:15]}...")

# Load the full AnnData object and filter it
print("\nLoading AnnData object and filtering...")
adata = ad.read_h5ad(file_path)

# Filter to keep only allowed genes
adata_filtered = adata[keep_indices].copy()

print(f"Filtered AnnData shape: {adata_filtered.shape}")
print(f"Original shape: {adata.shape}")

# Save the filtered file
print(f"\nSaving filtered data to: {output_path}")
adata_filtered.write_h5ad(output_path)
print("Done!")

