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
###################### cell eval #########################

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