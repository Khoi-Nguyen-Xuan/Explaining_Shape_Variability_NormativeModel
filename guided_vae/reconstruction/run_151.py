# main.py (NO OPTUNA) — fixed best hyperparameters + train 2000 epochs
# ---------------------------------------------------------------
import pickle
import argparse
import os
import os.path as osp
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from psbody.mesh import Mesh

from reconstruction import AE, run, eval_error
from datasets import MeshData
from utils import utils, writer as writer_mod, mesh_sampling, sap
from torch_geometric.loader import DataLoader  # PyG DataLoader


# ------------------------
# Args
# ------------------------
parser = argparse.ArgumentParser(description="mesh autoencoder (fixed run, no Optuna)")
parser.add_argument("--exp_name", type=str, default="interpolation_exp")
parser.add_argument("--dataset", type=str, default="CoMA")
parser.add_argument("--split", type=str, default="interpolation")
parser.add_argument("--test_exp", type=str, default="bareteeth")
parser.add_argument("--n_threads", type=int, default=4)
parser.add_argument("--device_idx", type=int, default=0)

# (kept for compatibility; we will override below)
parser.add_argument("--out_channels", nargs="+", default=[32, 32, 32, 64], type=int)
parser.add_argument("--latent_channels", type=int, default=16)
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--seq_length", type=int, default=[9, 9, 9, 9], nargs="+")
parser.add_argument("--dilation", type=int, default=[1, 1, 1, 1], nargs="+")

parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_decay", type=float, default=0.99)
parser.add_argument("--decay_step", type=int, default=1)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--weight_decay_c", type=float, default=1e-4)

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--beta", type=float, default=0.0)
parser.add_argument("--wcls", type=float, default=1.0)

parser.add_argument("--correlation_loss", type=bool, default=False)
parser.add_argument("--guided_contrastive_loss", type=bool, default=True)
parser.add_argument("--guided", type=bool, default=False)
parser.add_argument("--tc", type=bool, default=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--temperature", type=float, default=100.0)
parser.add_argument("--threshold", type=float, default=0.025001)

args = parser.parse_args()

# ------------------------
# Fixed run config (YOUR BEST TRIAL) + epochs=2000
# ------------------------
args.out_channels = [16, 16, 16, 32]
args.latent_channels = 128
args.in_channels = 3

args.seq_length = [49, 49, 49, 49]
args.dilation = [2, 2, 2, 2]

args.lr = 0.00022956047762143564
args.lr_decay = 0.92
args.decay_step = 37
args.weight_decay = 1e-5
args.weight_decay_c = 1e-4

args.batch_size = 16
args.epochs = 800  # <<< requested

args.beta = 0.30762065994572013
args.wcls = 34.91490763774021
args.w_leak = 0.0

args.temperature = 0.4045180229613066
args.threshold = 16
args.delta = 0.8

args.age_dim = 1
args.use_mu_for_guidance = True

args.guided = False
args.guided_contrastive_loss = True
args.correlation_loss = False
args.attribute_loss = False


# ------------------------
# Paths
# ------------------------
args.work_dir = "/scratch/xuankhoi/mesh/Explaining_Shape_Variability_CALSNIC/src/DeepLearning/compute_canada/guided_vae"
args.data_fp = osp.join(args.work_dir, "data", args.dataset)
args.out_dir = osp.join(args.work_dir, "data", "out", args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, "checkpoints")

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

writer = writer_mod.Writer(args)
device = torch.device("cuda", args.device_idx)
torch.set_num_threads(args.n_threads)


# ------------------------
# Determinism
# ------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


set_seed(args.seed)


# ------------------------
# Helpers for saving
# ------------------------
ROOT_SAVE = "/scratch/xuankhoi/mesh/Explaining_Shape_Variability_CALSNIC/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/151_fixed_run_800epochs"
SUMMARY_CSV = os.path.join(ROOT_SAVE, "final_summary.csv")


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _dump_args_json(base_dir: str, args_obj):
    try:
        with open(os.path.join(base_dir, "args.json"), "w") as f:
            json.dump(vars(args_obj), f, indent=2, default=str)
    except Exception:
        with open(os.path.join(base_dir, "args.txt"), "w") as f:
            f.write(repr(args_obj))


def _save_artifacts(base_dir: str, model, args, spiral_indices_list, up_transform_list,
                    down_transform_list, meshdata):
    _ensure_dir(base_dir)
    torch.save(model.state_dict(), os.path.join(base_dir, "model_state_dict.pt"))
    torch.save(args.in_channels, os.path.join(base_dir, "in_channels.pt"))
    torch.save(args.out_channels, os.path.join(base_dir, "out_channels.pt"))
    torch.save(args.latent_channels, os.path.join(base_dir, "latent_channels.pt"))
    torch.save(spiral_indices_list, os.path.join(base_dir, "spiral_indices_list.pt"))
    torch.save(up_transform_list, os.path.join(base_dir, "up_transform_list.pt"))
    torch.save(down_transform_list, os.path.join(base_dir, "down_transform_list.pt"))
    torch.save(meshdata.std, os.path.join(base_dir, "std.pt"))
    torch.save(meshdata.mean, os.path.join(base_dir, "mean.pt"))
    torch.save(meshdata.template_face, os.path.join(base_dir, "faces.pt"))

    # snapshot of index splits for traceability (if exists)
    src_split = os.path.join(args.data_fp, "processed", "train_val_test_files.pt")
    if os.path.exists(src_split):
        try:
            import shutil
            shutil.copy(src_split, os.path.join(base_dir, "train_val_test_files.pt"))
        except Exception:
            pass


# ------------------------
# Load dataset
# ------------------------
print("DATA FP:", args.data_fp)
template_fp = osp.join(args.data_fp, "template", "template.ply")
print("TEMPLATE:", template_fp)

meshdata = MeshData(
    args.data_fp,
    template_fp,
    split=args.split,
    test_exp=args.test_exp,
)

print("SPLIT SIZES (MeshData):",
      "train =", len(meshdata.train_dataset),
      "val   =", len(meshdata.val_dataset),
      "test  =", len(meshdata.test_dataset))


# ------------------------
# Generate / Load transform matrices
# ------------------------
transform_fp = osp.join(args.data_fp, "transform", "transform.pkl")
if not osp.exists(transform_fp):
    print("Generating transform matrices...")
    mesh = Mesh(filename=template_fp)
    ds_factors = [3, 3, 2, 2]
    _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
    tmp = {
        "vertices": V,
        "face": F,
        "adj": A,
        "down_transform": D,
        "up_transform": U,
    }
    utils.makedirs(osp.dirname(transform_fp))
    with open(transform_fp, "wb") as fp:
        pickle.dump(tmp, fp)
    print(f"Done! Saved to {transform_fp}")
else:
    with open(transform_fp, "rb") as f:
        tmp = pickle.load(f, encoding="latin1")

# NOTE: spiral indices are generated using args.seq_length and args.dilation
spiral_indices_list = [
    utils.preprocess_spiral(
        tmp["face"][idx],
        args.seq_length[idx],
        tmp["vertices"][idx],
        args.dilation[idx],
    ).to(device)
    for idx in range(len(tmp["face"]) - 1)
]

down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp["down_transform"]
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp["up_transform"]
]

# Sanity checks (same style you used)
print("spiral0 N:", spiral_indices_list[0].shape[0])
print("D0 shape:", down_transform_list[0].shape)
print("mean/std:", meshdata.mean.shape, meshdata.std.shape)
print("faces max idx:", int(meshdata.template_face.max()))
assert spiral_indices_list[0].shape[0] == int(meshdata.template_face.max()) + 1
assert meshdata.mean.shape == meshdata.std.shape == (spiral_indices_list[0].shape[0], 3)


# ------------------------
# Build model, optimizer, scheduler
# ------------------------
model = AE(
    args.in_channels,
    args.out_channels,
    args.latent_channels,
    spiral_indices_list,
    down_transform_list,
    up_transform_list,
).to(device)

print("Number of parameters:", utils.count_parameters(model))
print(model)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=args.decay_step,
    gamma=args.lr_decay,
)


# ------------------------
# DataLoaders
# ------------------------
train_loader = DataLoader(
    meshdata.train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    meshdata.val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
)
test_loader = DataLoader(
    meshdata.test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
)

print(
    f"Using batch_size={args.batch_size} -> "
    f"train batches={len(train_loader)}, val batches={len(val_loader)}, test batches={len(test_loader)}"
)


# ------------------------
# Train
# ------------------------
t0 = time.time()
run(
    model,
    train_loader,
    val_loader,
    args.epochs,
    optimizer,
    scheduler,
    writer,
    device,
    args.beta,
    args.wcls,
    args.w_leak,
    args.age_dim,
    args.use_mu_for_guidance,
    args.guided,
    args.guided_contrastive_loss,
    args.correlation_loss,
    args.latent_channels,
    args.weight_decay_c,
    args.temperature,
    args.delta,
    args.threshold,
)
t_train = time.time() - t0
print(f"Training done in {t_train/60.0:.2f} min")


# ------------------------
# Eval on VAL (reconstruction)
# ------------------------
euclidean_distance_val = eval_error(model, val_loader, device, meshdata, args.out_dir)
print("VAL Euclidean distance:", euclidean_distance_val)


# ------------------------
# Compute SAP on VAL (mu)
# ------------------------
latent_codes_list, ages_list = [], []
model.eval()
with torch.no_grad():
    for data in val_loader:
        x = data.x.to(device)
        y = data.y.to(device)

        _, mu, log_var, *_ = model(x)
        z = mu.view(mu.size(0), -1)
        z[torch.isnan(z) | torch.isinf(z)] = 0
        latent_codes_list.append(z.cpu())

        # ages -> [B,1]
        if y.dim() == 0:
            age = y.view(1, 1)
        elif y.dim() == 1:
            age = y.unsqueeze(1)
        else:
            age = y[..., 0]
        if age.dim() == 1:
            age = age.unsqueeze(1)
        else:
            age = age.view(age.size(0), 1)

        ages_list.append(age.cpu())

latent_codes = torch.cat(latent_codes_list, dim=0)
ages = torch.cat(ages_list, dim=0)

if latent_codes.shape[0] != ages.shape[0]:
    N = min(latent_codes.shape[0], ages.shape[0])
    print(f"[WARN] Mismatch codes/ages -> trimming to {N}")
    latent_codes = latent_codes[:N]
    ages = ages[:N]

sap_score_val = sap(
    factors=ages.numpy(),
    codes=latent_codes.numpy(),
    continuous_factors=True,
    regression=True,
)

print("VAL SAP:", sap_score_val)


# ------------------------
# Save final bundle
# ------------------------
_ensure_dir(ROOT_SAVE)
_dump_args_json(ROOT_SAVE, args)
_save_artifacts(ROOT_SAVE, model, args, spiral_indices_list, up_transform_list, down_transform_list, meshdata)

# Save val latent codes
torch.save(latent_codes, os.path.join(ROOT_SAVE, "val_latent_codes.pt"))
pd.DataFrame(latent_codes.numpy()).to_csv(os.path.join(ROOT_SAVE, "val_latent_codes.csv"), index=False)
pd.DataFrame(ages.numpy(), columns=["age"]).to_csv(os.path.join(ROOT_SAVE, "val_ages.csv"), index=False)

# Append / write a small summary CSV
row = {
    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    "epochs": int(args.epochs),
    "batch_size": int(args.batch_size),
    "latent_channels": int(args.latent_channels),
    "out_channels": str(args.out_channels),
    "seq_length": str(args.seq_length),
    "dilation": str(args.dilation),
    "lr": float(args.lr),
    "lr_decay": float(args.lr_decay),
    "decay_step": int(args.decay_step),
    "beta": float(args.beta),
    "wcls": float(args.wcls),
    "w_leak": float(args.w_leak),
    "temperature": float(args.temperature),
    "threshold": float(args.threshold),
    "delta": float(args.delta),
    "age_dim": int(args.age_dim),
    "use_mu_for_guidance": bool(args.use_mu_for_guidance),
    "guided_contrastive_loss": bool(args.guided_contrastive_loss),
    "euclidean_distance_val": float(euclidean_distance_val),
    "sap_score_val": float(sap_score_val),
    "train_minutes": float(t_train / 60.0),
}
df_row = pd.DataFrame([row])
if not os.path.exists(SUMMARY_CSV):
    df_row.to_csv(SUMMARY_CSV, index=False)
else:
    df_row.to_csv(SUMMARY_CSV, mode="a", header=False, index=False)

print(f"\nSaved final run to: {ROOT_SAVE}")
print(f"Summary CSV: {SUMMARY_CSV}")
