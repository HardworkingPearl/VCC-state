#!/usr/bin/env python3
"""
Script to create competition_test_template.h5ad from control cells.

This script:
1. Loads controlled cells (unperturbed cells) from a source h5ad file
2. Reads target genes and n_cells from pert_counts_Test.csv
3. For each target gene, samples n_cells control cells (with replacement)
4. Creates a new AnnData object with the target gene as the perturbation
5. Saves as competition_test_template.h5ad
"""

import argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse
from pathlib import Path


def create_competition_test_template(
    source_adata_path: str,
    csv_path: str,
    output_path: str,
    pert_col: str = "target_gene",
    control_pert: str = "non-targeting",
    seed: int = 42,
):
    """
    Create competition test template from control cells.
    
    Args:
        source_adata_path: Path to source h5ad file containing control cells
        csv_path: Path to CSV file with target_gene and n_cells columns
        output_path: Path to save the output h5ad file
        pert_col: Column name for perturbations (default: "target_gene")
        control_pert: Label for control perturbation (default: "non-targeting")
        seed: Random seed for reproducibility
    """
    print("=" * 70)
    print("Creating Competition Test Template")
    print("=" * 70)
    breakpoint()
    # Set random seed
    rng = np.random.default_rng(seed)
    print(f"Random seed: {seed}")
    
    # Load source AnnData
    print(f"\nLoading source AnnData from: {source_adata_path}")
    adata = ad.read_h5ad(source_adata_path)
    print(f"  Shape: {adata.shape} (cells x genes)")
    print(f"  Obs columns: {list(adata.obs.columns)}")
    
    # Find perturbation column (could be "gene", "target_gene", etc.)
    # Try common column names
    possible_pert_cols = [pert_col, "gene", "target_gene", "perturbation"]
    pert_col_found = None
    for col in possible_pert_cols:
        if col in adata.obs.columns:
            pert_col_found = col
            break
    
    if pert_col_found is None:
        raise ValueError(
            f"Could not find perturbation column. Tried: {possible_pert_cols}. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    
    print(f"  Using perturbation column: {pert_col_found}")
    
    # Identify control cells
    print(f"\nIdentifying control cells with condition: '{control_pert}'")
    col_values = adata.obs[pert_col_found].values
    control_mask = col_values == control_pert
    control_indices = np.flatnonzero(control_mask)
    
    print(f"  Found {len(control_indices)} control cells out of {adata.n_obs} total cells")
    if len(control_indices) == 0:
        raise ValueError(
            f"No control cells found with condition '{control_pert}' in column '{pert_col_found}'"
        )
    
    # Load CSV file
    print(f"\nLoading target genes from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Found {len(df)} target genes")
    print(f"  Total cells to create: {df['n_cells'].sum()}")
    
    # Validate CSV columns
    if "target_gene" not in df.columns:
        raise ValueError("CSV file must have 'target_gene' column")
    if "n_cells" not in df.columns:
        raise ValueError("CSV file must have 'n_cells' column")
    
    # Collect all cells to create
    print("\nSampling control cells for each target gene...")
    all_cell_indices = []
    all_perturbations = []
    
    for idx, row in df.iterrows():
        target_gene = str(row["target_gene"])
        n_cells = int(row["n_cells"])
        
        if n_cells <= 0:
            continue
        
        # Sample control cells with replacement
        sampled_indices = rng.choice(control_indices, size=n_cells, replace=True)
        all_cell_indices.extend(sampled_indices)
        all_perturbations.extend([target_gene] * n_cells)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} target genes...")
    
    all_cell_indices = np.array(all_cell_indices)
    total_cells = len(all_cell_indices)
    print(f"\n  Total cells created: {total_cells}")
    
    # Create new AnnData object
    print("\nCreating new AnnData object...")
    
    # Copy X data (expression matrix)
    if issparse(adata.X):
        new_X = adata.X[all_cell_indices].copy()
    else:
        new_X = adata.X[all_cell_indices].copy()
    
    # Create new obs (cell metadata)
    new_obs = adata.obs.iloc[all_cell_indices].copy()
    # Update perturbation column to target genes
    new_obs[pert_col_found] = all_perturbations
    
    # Copy obsm (cell embeddings) if present
    new_obsm = {}
    for key, matrix in adata.obsm.items():
        if issparse(matrix):
            new_obsm[key] = matrix[all_cell_indices].copy()
        else:
            new_obsm[key] = matrix[all_cell_indices].copy()
    
    # Create new AnnData
    new_adata = ad.AnnData(
        X=new_X,
        obs=new_obs,
        var=adata.var.copy(),
        obsm=new_obsm,
        varm=adata.varm.copy() if hasattr(adata, 'varm') else {},
        uns=adata.uns.copy() if hasattr(adata, 'uns') else {},
    )
    
    print(f"  New AnnData shape: {new_adata.shape}")
    print(f"  Unique perturbations: {new_adata.obs[pert_col_found].nunique()}")
    
    # Save to file
    print(f"\nSaving to: {output_path}")
    new_adata.write_h5ad(output_path, compression="gzip")
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  Source file: {source_adata_path}")
    print(f"  Control cells used: {len(control_indices)}")
    print(f"  Target genes: {len(df)}")
    print(f"  Total cells created: {total_cells}")
    print(f"  Output file: {output_path}")
    print("=" * 70)
    
    return new_adata


def main():
    parser = argparse.ArgumentParser(
        description="Create competition test template from control cells"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source h5ad file containing control cells",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="competition_support_set/pert_counts_Test.csv",
        help="Path to CSV file with target_gene and n_cells columns",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="competition_test_template.h5ad",
        help="Path to save output h5ad file",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default="target_gene",
        help="Column name for perturbations (default: target_gene)",
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        default="non-targeting",
        help="Label for control perturbation (default: non-targeting)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    create_competition_test_template(
        source_adata_path=args.source,
        csv_path=args.csv,
        output_path=args.output,
        pert_col=args.pert_col,
        control_pert=args.control_pert,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

