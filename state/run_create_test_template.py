#!/usr/bin/env python3
"""
Simple script to run create_competition_test_template.py
with default paths for the competition.
"""

from create_competition_test_template import create_competition_test_template
from pathlib import Path
breakpoint()
# Default paths
base_dir = Path(__file__).parent / "competition_support_set"
source_adata = Path("/home/absking/scratch/vcc/state/competition_support_set/competition_train.h5")

csv_path = base_dir / "pert_counts_Test.csv"
output_path = base_dir / "competition_test_template.h5ad"

# Check if files exist
if not source_adata.exists():
    print(f"Error: Source file not found: {source_adata}")
    print("Please specify a different source file with --source argument")
    exit(1)

if not csv_path.exists():
    print(f"Error: CSV file not found: {csv_path}")
    exit(1)

print(f"Source: {source_adata}")
print(f"CSV: {csv_path}")
print(f"Output: {output_path}")
print()

# Run the creation
create_competition_test_template(
    source_adata_path=str(source_adata),
    csv_path=str(csv_path),
    output_path=str(output_path),
    pert_col="target_gene",
    control_pert="non-targeting",
    seed=42,
)

