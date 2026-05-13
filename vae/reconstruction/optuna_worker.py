import os
import os.path as osp
import json
import time
import pickle
import shutil
import random
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from psbody.mesh import Mesh
from torch_geometric.loader import DataLoader
import optuna
from optuna.storages import JournalStorage, JournalFileStorage

from reconstruction import AE, run, eval_error
from datasets import MeshData
from utils import utils, writer, mesh_sampling


# ============================================================
# Parser
# ============================================================

parser = argparse.ArgumentParser(description="Healthy-population vanilla mesh VAE")

parser.add_argument("--n_trials", type=int, default=50)

parser.add_argument("--exp_name", type=str, default="healthy_vanilla_vae")
parser.add_argument("--dataset", type=str, default="CoMA")
parser.add_argument("--split", type=str, default="interpolation")
parser.add_argument("--test_exp", type=str, default="bareteeth")
parser.add_argument("--n_threads", type=int, default=4)
parser.add_argument("--device_idx", type=int, default=0)

# Network hyperparameters
parser.add_argument("--out_channels", nargs="+", default=[32, 32, 32, 64], type=int)
parser.add_argument("--latent_channels", type=int, default=16)
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--seq_length", type=int, default=[9, 9, 9, 9], nargs="+")
parser.add_argument("--dilation", type=int, default=[1, 1, 1, 1], nargs="+")

# Optimizer hyperparameters
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_decay", type=float, default=0.99)
parser.add_argument("--decay_step", type=int, default=1)
parser.add_argument("--weight_decay", type=float, default=1e-5)

# Training hyperparameters
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--beta", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=1)

args = parser.parse_args()


# ============================================================
# Paths
# ============================================================

args.work_dir = "/scratch/xuankhoi/mesh/Explaining_Shape_Variability_CALSNIC/src/DeepLearning/compute_canada/guided_vae"
args.data_fp = osp.join(args.work_dir, "data", args.dataset)
args.out_dir = osp.join(args.work_dir, "data", "out", args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, "checkpoints")

ROOT_SAVE = osp.join(
    args.work_dir,
    "data",
    args.dataset,
    "raw",
    "torus",
    "models_healthy_vanilla_vae",
)

SUMMARY_CSV = osp.join(ROOT_SAVE, "trials_summary.csv")
JOURNAL_PATH = "/scratch/xuankhoi/optuna_healthy_vanilla_vae.journal"
STUDY_NAME = "healthy_vanilla_vae_v1"


# ============================================================
# Setup
# ============================================================

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)
Path(ROOT_SAVE).mkdir(parents=True, exist_ok=True)

writer = writer.Writer(args)

device = torch.device("cuda", args.device_idx)
torch.set_num_threads(args.n_threads)


def set_seed(seed):
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


# ============================================================
# Load dataset
# ============================================================

print("Data path:", args.data_fp)

template_fp = osp.join(args.data_fp, "template", "template.ply")
print("Template path:", template_fp)

meshdata = MeshData(
    args.data_fp,
    template_fp,
    split=args.split,
    test_exp=args.test_exp,
)

print(
    "SPLIT SIZES:",
    "train =", len(meshdata.train_dataset),
    "val =", len(meshdata.val_dataset),
    "test =", len(meshdata.test_dataset),
)

# Important:
# At this stage, make sure MeshData already contains only healthy controls
# in train/val/test. This script does not filter diagnosis labels.


# ============================================================
# Generate/load transform matrices
# ============================================================

transform_fp = osp.join(args.data_fp, "transform", "transform.pkl")

if not osp.exists(transform_fp):
    print("Generating transform matrices...")

    mesh = Mesh(filename=template_fp)
    ds_factors = [3, 3, 2, 2]

    _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
        mesh,
        ds_factors,
    )

    tmp = {
        "vertices": V,
        "face": F,
        "adj": A,
        "down_transform": D,
        "up_transform": U,
    }

    with open(transform_fp, "wb") as fp:
        pickle.dump(tmp, fp)

    print(f"Transform matrices saved to {transform_fp}")

else:
    with open(transform_fp, "rb") as f:
        tmp = pickle.load(f, encoding="latin1")


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


# ============================================================
# Save helpers
# ============================================================

def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _dump_args_json(base_dir: str, args_obj):
    with open(osp.join(base_dir, "args.json"), "w") as f:
        json.dump(vars(args_obj), f, indent=2, default=str)


def _save_artifacts(
    base_dir: str,
    model,
    args,
    spiral_indices_list,
    up_transform_list,
    down_transform_list,
    meshdata,
):
    _ensure_dir(base_dir)

    torch.save(model.state_dict(), osp.join(base_dir, "model_state_dict.pt"))
    torch.save(args.in_channels, osp.join(base_dir, "in_channels.pt"))
    torch.save(args.out_channels, osp.join(base_dir, "out_channels.pt"))
    torch.save(args.latent_channels, osp.join(base_dir, "latent_channels.pt"))

    torch.save(spiral_indices_list, osp.join(base_dir, "spiral_indices_list.pt"))
    torch.save(up_transform_list, osp.join(base_dir, "up_transform_list.pt"))
    torch.save(down_transform_list, osp.join(base_dir, "down_transform_list.pt"))

    torch.save(meshdata.std, osp.join(base_dir, "std.pt"))
    torch.save(meshdata.mean, osp.join(base_dir, "mean.pt"))
    torch.save(meshdata.template_face, osp.join(base_dir, "faces.pt"))

    split_fp = osp.join(args.data_fp, "processed", "train_val_test_files.pt")
    if osp.exists(split_fp):
        shutil.copy(split_fp, osp.join(base_dir, "train_val_test_files.pt"))


def _save_latent_codes(model, loader, device, save_fp):
    model.eval()
    latent_codes = []

    with torch.no_grad():
        for data in loader:
            x = data.x.to(device)
            y = data.y.to(device)

            B = y.view(-1).size(0)
            N = x.size(0)

            if N % B != 0:
                raise RuntimeError(f"Cannot reshape latent batch: x={N}, B={B}")

            V = N // B
            x = x.view(B, V, -1)

            _, mu, log_var, *_ = model(x)

            z = mu.view(mu.size(0), -1)
            z[torch.isnan(z) | torch.isinf(z)] = 0

            latent_codes.append(z.cpu())

    latent_codes = torch.cat(latent_codes, dim=0)

    torch.save(latent_codes, save_fp)
    pd.DataFrame(latent_codes.numpy()).to_csv(
        save_fp.replace(".pt", ".csv"),
        index=False,
    )

    return latent_codes


def _save_full_trial_bundle(
    base_dir: str,
    trial_number: int,
    model,
    args,
    spiral_indices_list,
    up_transform_list,
    down_transform_list,
    meshdata,
    euclidean_distance: float,
    trial_params: dict,
    val_loader,
):
    _ensure_dir(base_dir)

    _save_artifacts(
        base_dir,
        model,
        args,
        spiral_indices_list,
        up_transform_list,
        down_transform_list,
        meshdata,
    )

    _dump_args_json(base_dir, args)

    latent_codes = _save_latent_codes(
        model,
        val_loader,
        device,
        osp.join(base_dir, "val_latent_codes.pt"),
    )

    meta = {
        "trial_number": int(trial_number),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": trial_params,
        "values": {
            "euclidean_distance_val": float(euclidean_distance),
        },
        "latent_codes_shape": list(latent_codes.shape),
    }

    with open(osp.join(base_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _append_summary_row(trial_number: int, euclid: float, params: dict):
    row = {
        "trial_number": int(trial_number),
        "euclidean_distance_val": float(euclid),
        "params_json": json.dumps(params, sort_keys=True),
    }

    df_row = pd.DataFrame([row])

    if not osp.exists(SUMMARY_CSV):
        df_row.to_csv(SUMMARY_CSV, index=False)
    else:
        df_row.to_csv(SUMMARY_CSV, mode="a", header=False, index=False)


# ============================================================
# Objective
# ============================================================

def objective(trial):
    set_seed(args.seed + trial.number)

    # Hyperparameters
    args.epochs = trial.suggest_int("epochs", 200, 1000, step=100)
    args.batch_size = trial.suggest_int("batch_size", 8, 32, step=4)

    args.beta = trial.suggest_float("beta", 1e-5, 1e-1, log=True)
    args.lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    args.lr_decay = trial.suggest_float("learning_rate_decay", 0.70, 0.99, step=0.01)
    args.decay_step = trial.suggest_int("decay_step", 1, 50)

    args.latent_channels = trial.suggest_categorical(
        "latent_channels",
        [8, 16, 32, 64,128,256,512],
    )

    sequence_length = trial.suggest_int("sequence_length", 5, 50)
    args.seq_length = [sequence_length] * 4

    dilation = trial.suggest_int("dilation", 1, 2)
    args.dilation = [dilation] * 4

    out_channel = trial.suggest_int("out_channel", 8, 64, step=8)
    args.out_channels = [
        out_channel,
        out_channel,
        out_channel,
        2 * out_channel,
    ]

    print("\n==============================")
    print(f"Trial {trial.number}")
    print(args)
    print("==============================\n")

    # Recompute spiral indices because seq_length/dilation are trial-dependent
    trial_spiral_indices_list = [
        utils.preprocess_spiral(
            tmp["face"][idx],
            args.seq_length[idx],
            tmp["vertices"][idx],
            args.dilation[idx],
        ).to(device)
        for idx in range(len(tmp["face"]) - 1)
    ]

    model = AE(
        args.in_channels,
        args.out_channels,
        args.latent_channels,
        trial_spiral_indices_list,
        down_transform_list,
        up_transform_list,
    ).to(device)

    print("Number of parameters:", utils.count_parameters(model))

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

    print(
        f"[Trial {trial.number}] batch_size={args.batch_size}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}"
    )

    # Vanilla VAE training only
    run(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
        device=device,
        beta=args.beta,
    )

    euclidean_distance = eval_error(
        model,
        val_loader,
        device,
        meshdata,
        args.out_dir,
    )

    trial_params = dict(trial.params)
    trial_params.update(
        {
            "out_channels": args.out_channels,
            "seq_length": args.seq_length,
            "dilation": args.dilation,
        }
    )

    _append_summary_row(
        trial_number=trial.number,
        euclid=euclidean_distance,
        params=trial_params,
    )

    trial_dir = osp.join(ROOT_SAVE, str(trial.number))

    _save_full_trial_bundle(
        base_dir=trial_dir,
        trial_number=trial.number,
        model=model,
        args=args,
        spiral_indices_list=trial_spiral_indices_list,
        up_transform_list=up_transform_list,
        down_transform_list=down_transform_list,
        meshdata=meshdata,
        euclidean_distance=euclidean_distance,
        trial_params=trial_params,
        val_loader=val_loader,
    )

    return euclidean_distance


# ============================================================
# Optuna
# ============================================================

class LogAfterEachTrial:
    def __init__(self, save_every=10):
        self.save_every = save_every

    def __call__(self, study, trial):
        if trial.number % self.save_every != 0:
            return

        torch.save(
            study.trials,
            osp.join(ROOT_SAVE, "intermediate_trials.pt"),
        )


sampler = optuna.samplers.TPESampler(seed=args.seed)

storage = JournalStorage(JournalFileStorage(JOURNAL_PATH))

study = optuna.create_study(
    study_name=STUDY_NAME,
    direction="minimize",
    sampler=sampler,
    storage=storage,
    load_if_exists=True,
)

study.optimize(
    objective,
    n_trials=args.n_trials,
    callbacks=[LogAfterEachTrial(save_every=10)],
)

print("\nBest trial:")
print("Trial number:", study.best_trial.number)
print("Validation Euclidean error:", study.best_trial.value)
print("Params:", study.best_trial.params)
