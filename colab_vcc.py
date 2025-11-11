########################### download data #######################

# import requests
# from tqdm.auto import tqdm  # picks the best bar for the environment

# url = "https://storage.googleapis.com/vcc_data_prod/datasets/state/competition_support_set.zip"
# output_path = "competition_support_set.zip"

# # stream the download so we can track progress
# response = requests.get(url, stream=True)
# total = int(response.headers.get("content-length", 0))

# with open(output_path, "wb") as f, tqdm(
#     total=total, unit='B', unit_scale=True, desc="Downloading"
# ) as bar:
#     for chunk in response.iter_content(chunk_size=8192):
#         if not chunk:
#             break
#         f.write(chunk)
#         bar.update(len(chunk))

##################### set wandb #######################
# ENTITY="arcinstitute" && sed -i "s|entity: your_entity_name|entity: ${ENTITY}|g" src/state/configs/wandb/default.yaml

###### cell_load/dataset/_perturbation.py
#################### prepare training dataset ############
from zipfile import ZipFile
from tqdm.auto import tqdm
import os
output_path = "competition_support_set.zip"
out_dir  = "competition_support_set"

os.makedirs(out_dir, exist_ok=True)

with ZipFile(output_path, 'r') as z:
    for member in tqdm(z.infolist(), desc="Unzipping", unit="file"):
        z.extract(member, out_dir)

############### command to run training ##############
uv run state tx train   data.kwargs.toml_config_path="competition_support_set/starter.toml"   data.kwargs.num_workers=4   data.kwargs.batch_col="batch_var"   data.kwargs.pert_col="target_gene"   data.kwargs.cell_type_key="cell_type"   data.kwargs.control_pert="non-targeting"   data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt"   training.max_steps=40000   training.ckpt_every_n_steps=1000   model=state_sm   wandb.tags="[first_run]"   output_dir="competition"   name="first_run"
uv run state tx train   data.kwargs.toml_config_path="competition_support_set/starter.toml"   data.kwargs.num_workers=4   data.kwargs.batch_col="batch_var"   data.kwargs.pert_col="target_gene"   data.kwargs.cell_type_key="cell_type"   data.kwargs.control_pert="non-targeting"   data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt"   training.max_steps=420   training.ckpt_every_n_steps=210   model=state_sm   wandb.tags="[first_run]"   output_dir="competition"   name="first_run"
uv run state tx train   data.kwargs.toml_config_path="competition_support_set/starter.toml"   data.kwargs.num_workers=4   data.kwargs.batch_col="batch_var"   data.kwargs.pert_col="target_gene"   data.kwargs.cell_type_key="cell_type"   data.kwargs.control_pert="non-targeting"   data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt"   training.max_steps=40000   training.ckpt_every_n_steps=1000   model=state_sm   wandb.tags="state"   output_dir="competition"   name="state"
uv run state tx train   data.kwargs.toml_config_path="competition_support_set/starter.toml"   data.kwargs.num_workers=4   data.kwargs.batch_col="batch_var"   data.kwargs.pert_col="target_gene"   data.kwargs.cell_type_key="cell_type"   data.kwargs.control_pert="non-targeting"   data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt"   training.max_steps=40000   training.ckpt_every_n_steps=1000   model=tabicl   wandb.tags="tabicl"   output_dir="competition"   name="tabicl"

python -m src.state tx train data.kwargs.toml_config_path="competition_support_set/starter.toml"   data.kwargs.num_workers=4   data.kwargs.batch_col="batch_var"   data.kwargs.pert_col="target_gene"   data.kwargs.cell_type_key="cell_type"   data.kwargs.control_pert="non-targeting"   data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt"   training.max_steps=40000   training.ckpt_every_n_steps=1000   model=tabicl   wandb.tags="tabicl"   output_dir="competition"   name="tabicl"
python -m src.state tx train data.kwargs.toml_config_path="competition_support_set/starter.toml"   data.kwargs.num_workers=4   data.kwargs.batch_col="batch_var"   data.kwargs.pert_col="target_gene"   data.kwargs.cell_type_key="cell_type"   data.kwargs.control_pert="non-targeting"   data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt"   training.max_steps=213   training.ckpt_every_n_steps=213 training.val_freq=213 model=xgboost   wandb.tags="xgboost"   output_dir="competition"   name="xgboost"
python -m src.state tx train data.kwargs.toml_config_path="competition_support_set/starter.toml"   data.kwargs.num_workers=4   data.kwargs.batch_col="batch_var"   data.kwargs.pert_col="target_gene"   data.kwargs.cell_type_key="cell_type"   data.kwargs.control_pert="non-targeting"   data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt"   training.max_steps=213   training.ckpt_every_n_steps=213 training.val_freq=2 model=xgboost   wandb.tags="xgboost"   output_dir="competition"   name="xgboost"

SCIPY_ARRAY_API=1 python -m src.state tx train data.kwargs.toml_config_path="competition_support_set/starter.toml"   data.kwargs.num_workers=4   data.kwargs.batch_col="batch_var"   data.kwargs.pert_col="target_gene"   data.kwargs.cell_type_key="cell_type"   data.kwargs.control_pert="non-targeting"   data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt"   training.max_steps=213   training.ckpt_every_n_steps=213 training.val_freq=213 model=tabpfn   wandb.tags="tabpfn"   output_dir="competition"   name="tabpfn"
python -m src.state tx train data.kwargs.toml_config_path="competition_support_set/starter.toml"   data.kwargs.num_workers=4   data.kwargs.batch_col="batch_var"   data.kwargs.pert_col="target_gene"   data.kwargs.cell_type_key="cell_type"   data.kwargs.control_pert="non-targeting"   data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt"   tr             login01" 17:12 06-Nov-25aining.max_steps=213   training.ckpt_every_n_steps=213 training.val_freq=213 model=causalpfn   wandb.tags="causalpfn"   output_dir="competition"   name="causalpfn"

```bash
state tx train \
data.kwargs.toml_config_path="examples/fewshot.toml" \
data.kwargs.embed_key=X_hvg \
data.kwargs.num_workers=12 \
data.kwargs.batch_col=batch_var \
data.kwargs.pert_col=target_gene \
data.kwargs.cell_type_key=cell_type \
data.kwargs.control_pert=TARGET1 \
training.max_steps=40000 \
training.val_freq=100 \
training.ckpt_every_n_steps=100 \
training.batch_size=8 \
training.lr=1e-4 \
model.kwargs.cell_set_len=64 \
model.kwargs.hidden_dim=328 \
model=pertsets \
wandb.tags="[test]" \
output_dir="$HOME/state" \
name="test"
```
############### evaluation on checkpoints ###############
uv run state tx infer \
  --output "competition/prediction.h5ad" \
  --model-dir "competition/first_run" \
  --checkpoint "competition/first_run/checkpoints/final.ckpt" \
  --adata "competition_support_set/competition_val_template.h5ad" \
  --pert-col "target_gene"

python -m state tx infer \
  --output "competition/prediction.h5ad" \
  --model-dir "competition/causalpfn" \
  --checkpoint "competition/causalpfn/checkpoints/step=step=28968-val_loss=val_loss=4.4869.ckpt" \
  --adata "competition_support_set/competition_val_template.h5ad" \
  --pert-col "target_gene"
###################### cell eval #########################
curl -L -A "Mozilla/5.0" -e "https://plus.figshare.com" "https://plus.figshare.com/ndownloader/files/35775554" -o rpe1_normalized_singlecell_01.h5ad
curl -L -A "Mozilla/5.0" -e "https://www.ncbi.nlm.nih.gov" "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE264667&format=file&file=GSE264667%5Fhepg2%5Fraw%5Fsinglecell%5F01%2Eh5ad" -o GSE264667_hepg2_raw_singlecell_01.h5ad

export PYTHONPATH="$PWD/src:$PYTHONPATH"

uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval prep -i competition/prediction.h5ad -g competition_support_set/gene_names.csv
python -m src.cell_eval prep -i /home/absking/scratch/vcc/state/competition/prediction.h5ad -g /home/absking/scratch/vcc/state/competition_support_set/gene_names.csv
"""
  # 1. Get the source
cd ~/scratch
wget https://github.com/facebook/zstd/releases/download/v1.5.5/zstd-1.5.5.tar.gz
tar xf zstd-1.5.5.tar.gz
cd zstd-1.5.5

# 2. Build
make

# 3. Install into local prefix
PREFIX=~/local/zstd
make install PREFIX=$PREFIX

# 4. Add to PATH and LD_LIBRARY_PATH
export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH

# 5. Verify
which zstd
zstd --version
"""
OSError: libcudnn.so: cannot open shared object file
module load StdEnv/2023
module load cuda/12.6
module load cudnn
echo $EBROOTCUDNN
ls $EBROOTCUDNN/lib
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/CUDA/cuda12.6/cudnn/9.10.0.56/lib:$LD_LIBRARY_PATH
pip install '/home/absking/scratch/transformer_engine_cu12-2.3.0-py3-none-linux_x86_64.whl'


OSError: cublas.so: cannot open shared object file
export CUDA_HOME=$EBROOTCUDA
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

triton==3.2.0

module load r/4.5.0
module load StdEnv/2023

gunzip *.gz

module load hdf5
module load udunits

module load udunits/2.2.27.6_gcc11.3.0-rocky8
module load proj/7.2.1-rocky8
module load geos/3.7.1-rocky8
module load gdal/2.3.2-rocky8
R
install.packages("sf")
library("sf")
devtools::install_github('cole-trapnell-lab/monocle3')

# 1. Download the binary
curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip

# 2. Unzip it
unzip rclone-current-linux-amd64.zip
cd rclone-*-linux-amd64

# 3. Create a local bin folder if you don't have one
mkdir -p ~/bin

# 4. Copy the binary there
cp rclone ~/bin/

# 5. Add it to your PATH
echo 'export PATH=$PATH:~/bin' >> ~/.bashrc
source ~/.bashrc



################# notes to deal with dataset ########################
################# adata_for_cellflow_datasets_with_embeddings
import h5py
import numpy as np
from scipy.sparse import csr_matrix

with h5py.File("pbmc_parse.h5ad", "r") as f:
    data = f['X/data'][:]
    indices = f['X/indices'][:]
    indptr = f['X/indptr'][:]
    n_obs = f['obs/_index'].shape[0]    # number of cells (rows)
    n_vars = f['var/_index'].shape[0]   # number of genes (columns)
    X = csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))
len(set(aaa['var/_index'][:]).intersection(set(f['var/_index'][:]))) -> 1209
f.keys()<KeysViewHDF5 ['X', 'layers', 'obs', 'obsm', 'obsp', 'uns', 'var', 'varm', 'varp']>
f['obs'].keys()
<KeysViewHDF5 ['_index', 'bc1_well', 'bc1_wind', 'bc2_well', 'bc2_wind', 'bc3_well', 'bc3_wind', 'cell_type', 
'cytokine', 'cytokine_family', 'donor', 'gene_count', 'log1p_n_genes_by_counts', 'log1p_total_counts', 'log1p_total_counts_MT', 
'mread_count', 'pct_counts_MT', 'sample', 'species', 'total_counts_MT', 'treatment', 'tscp_count']>
f['obs/log1p_n_genes_by_counts'][:]array([7.71289096, ..., 7.39633529], shape=(9697974,))
{b'IL-17F', b'ADSF', b'IL-4', b'IFN-alpha1', b'IGF-1', b'CD27L', b'IL-12', b'4-1BBL', b'IFN-gamma', b'IL-9', b'EPO', b'IL-3', b'HGF', b'IL-20', b'GM-CSF', b'VEGF', b'IL-11', b'LIGHT', b'GITRL', b'IL-31', b'IL-35', b'IFN-beta', b'C5a', b'FLT3L', b'APRIL', b'BAFF', b'TGF-beta1', b'PBS', b'TRAIL', b'OSM', b'RANKL', b'TSLP', b'PRL', b'IL-32-beta', b'IL-33', b'CT-1', b'Decorin', b'IL-18', b'IL-36Ra', b'M-CSF', b'IL-26', b'IL-17D', b'IL-22', b'Noggin', b'IL-1-alpha', b'TPO', b'IFN-epsilon', b'IL-17B', b'LT-alpha1-beta2', b'IL-36-alpha', b'IL-21', b'FGF-beta', b'IL-5', b'IL-7', b'TWEAK', b'IL-13', b'IL-24', b'G-CSF', b'LT-alpha2-beta1', b'IL-17E', b'TL1A', b'SCF', b'OX40L', b'PSPN', b'IFN-lambda1', b'IL-15', b'IFN-lambda2', b'IL-27', b'IL-19', b'IL-17C', b'TNF-alpha', b'IL-1Ra', b'IL-10', b'CD30L', b'IL-2', b'IL-8', b'IL-34', b'LIF', b'IL-17A', b'EGF', b'FasL', b'IL-6', b'Leptin', b'IFN-lambda3', b'CD40L', b'IL-23', b'GDNF', b'C3a', b'IL-16', b'IFN-omega', b'IL-1-beta'}
f['obs/bc1_well'].keys() <KeysViewHDF5 ['categories', 'codes']>: f['obs/bc1_well/categories'] <HDF5 dataset "categories": shape (96,), type "|O">
f['obs/bc1_well/codes']: <HDF5 dataset "codes": shape (9697974,), type "|i1">
f['obs/cytokine/categories']<HDF5 dataset "categories": shape (91,), type "|O"> f['obs/cytokine/categories'][:] array([b'4-1BBL', b'ADSF' ...
f['obs/sample'].keys() <KeysViewHDF5 ['categories', 'codes']>
f['obs/cytokine/categories'][:] -> array([b'4-1BBL', b'ADSF', b'APRIL', b'BAFF'
f['uns'].keys() <KeysViewHDF5 ['donor_embeddings', 'esm2_embeddings', 'hvg', 'log1p']>

f['var'].keys()
<KeysViewHDF5 ['_index', 'dispersions', 'dispersions_norm', 'highly_variable', 'means', 'n_cells']>
f['var/_index'][:] array([b'CFH', b'KLHL13', b'TFPI', ..., b'ANKRD36BP2-1', b'ENSG00000291230', b'SOD2'], shape=(2000,), dtype=object)

from anndata import AnnData
import numpy as np
import scipy.sparse as sp

all_genes = sorted(set().union(*[ad.var_names for ad in adatas]))

def align_union(adata, all_genes):
    gene_index = {g: i for i, g in enumerate(all_genes)}
    idx = [gene_index[g] for g in adata.var_names]
    X_new = sp.lil_matrix((adata.n_obs, len(all_genes)))
    X_new[:, idx] = adata.X
    new = AnnData(X_new.tocsr(), obs=adata.obs, var=None)
    new.var_names = all_genes
    return new

adatas = [align_union(ad, all_genes) for ad in adatas]
# f['obs/gene_id/codes'][:]  shape=(18585,) categories (72,)
# X.shape (18585, 58347)
f['var/id'][:]: array([b'ENSG00000000003.14', b'ENSG00000000005.5', b'ENSG00000000419.12', ..., b'ENSG00000284748.1'], shape=(58347,), dtype=object)



# for datasets loading
/scratch/absking/vcc/state/.venv/lib/python3.11/site-packages/cell_load/utils/data_utils.py
/scratch/absking/vcc/state/.venv/lib/python3.11/site-packages/cell_load/data_modules/perturbation_dataloader.py
/scratch/absking/vcc/state/.venv/lib/python3.11/site-packages/cell_load/dataset/_perturbation.py

# for causalpfn
/home/absking/scratch/CausalPFN/src/causalpfn/causal_estimator.py
/home/absking/scratch/CausalPFN/src/causalpfn/models/icl_model.py
/scratch/absking/CausalPFN/src/causalpfn/models/model.py
Q: why same number of perturbed cells and control cells


causalFPN: module load StdEnv/2023 gcc/12.3 cuda/12.2
(vector) absking@klogin01:~/scratch/vcc$ module load faiss/1.7.4
export PYTHONPATH="/home/absking/scratch/vcc/cell-load/src:$PYTHONPATH"

pip install --no-deps huggingface_hub

python -m src.state tx train data.kwargs.toml_config_path="competition_support_set/starter.toml"   data.kwargs.num_workers=4   data.kwargs.batch_col="batch_var"   data.kwargs.pert_col="target_gene"   data.kwargs.cell_type_key="cell_type"   data.kwargs.control_pert="non-targeting"   data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt"   training.max_steps=213   training.ckpt_every_n_steps=213 training.val_freq=213 model=causalpfn   wandb.tags="causalpfn"   output_dir="competition"   name="causalpfn"