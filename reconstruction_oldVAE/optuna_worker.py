import pickle
import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from psbody.mesh import Mesh
from scipy import stats
from reconstruction import AE, run, eval_error
from datasets import MeshData
from utils import utils, writer, DataLoader, mesh_sampling, sap, point_biserial_correlation,sap_fixed_age
import optuna
from contextlib import redirect_stdout
import shutil
import random
from sklearn.metrics import accuracy_score
from torch.utils.data import ConcatDataset
from typing import Optional, Dict, Any
from torch_geometric.loader import DataLoader  # ← override with PyG DataLoader

import argparse



parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument("--n_trials", type=int, default=50)
cli = parser.parse_args()

parser.add_argument('--exp_name', type=str, default='interpolation_exp')
parser.add_argument('--dataset', type=str, default='CoMA')
parser.add_argument('--split', type=str, default='interpolation')
parser.add_argument('--test_exp', type=str, default='bareteeth')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=0)

# network hyperparameters
parser.add_argument('--out_channels', nargs='+', default=[32, 32, 32, 64], type=int)
parser.add_argument('--latent_channels', type=int, default=16)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--weight_decay_c', type=float, default=1e-4)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument('--wcls', type=int, default=1)

# others
parser.add_argument('--correlation_loss', type=bool, default=False)
parser.add_argument('--guided_contrastive_loss', type=bool, default=True)
parser.add_argument('--guided', type=bool, default=False)
parser.add_argument('--tc', type=bool, default=True)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--temperature', type=int, default=100)
parser.add_argument('--threshold', type=float, default=0.025001)

args = parser.parse_args()

#args.work_dir = osp.dirname(osp.realpath(__file__))
args.work_dir = "/scratch/xuankhoi/mesh/Explaining_Shape_Variability_CALSNIC/src/DeepLearning/compute_canada/guided_vae"
args.data_fp = osp.join(args.work_dir, 'data', args.dataset)
args.out_dir = osp.join(args.work_dir, 'data', 'out', args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
#print(args)

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

writer = writer.Writer(args)
device = torch.device('cuda', args.device_idx)
torch.set_num_threads(args.n_threads)

# deterministic
#torch.use_deterministic_algorithms(True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

set_seed(args.seed)

# load dataset
print(args.data_fp)
template_fp = osp.join(args.data_fp, 'template', 'template.ply')
print(template_fp)
meshdata = MeshData(args.data_fp,
                    template_fp,
                    split=args.split,
                    test_exp=args.test_exp)

print("SPLIT SIZES (MeshData):",
      "train =", len(meshdata.train_dataset),
      "val =", len(meshdata.val_dataset),
      "test =", len(meshdata.test_dataset))

for split_name, ds in [("train", meshdata.train_dataset),
                       ("val",   meshdata.val_dataset),
                       ("test",  meshdata.test_dataset)]:
    sample = ds[0]
    print(f"{split_name} sample[0].y shape:", sample.y.shape,
          "values:", sample.y[:5])
    break

# ===== OPTIONAL DEBUG: peek at labels with a temporary loader =====
print("\n=== DEBUG: sample labels from datasets ===")
for split_name, ds in [("train", meshdata.train_dataset),
                       ("val",   meshdata.val_dataset),
                       ("test",  meshdata.test_dataset)]:
    try:
        sample = ds[0]  # first subject in this split
        print(f"{split_name} sample[0].y shape:", sample.y.shape)
        print(f"{split_name} sample[0].y values:", sample.y)
    except Exception as e:
        print(f"{split_name} sample[0] label read FAILED:", e)

# small debug loader (NOT used for training/Optuna)
debug_loader = DataLoader(meshdata.train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True)
print("\n=== DEBUG: first batch from train_loader (batch_size =", args.batch_size, ") ===")
for batch in debug_loader:
    y_dbg = batch.y
    print("train batch.y shape:", y_dbg.shape)
    print("train batch.y values:", y_dbg)
    break  # only first batch
print("=== END DEBUG ===\n")
# ======================================

# generate/load transform matrices
transform_fp = osp.join(args.data_fp, 'transform', 'transform.pkl')
if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)
    ds_factors = [3, 3, 2, 2]
    _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
        mesh, ds_factors)
    tmp = {
        'vertices': V,
        'face': F,
        'adj': A,
        'down_transform': D,
        'up_transform': U
    }

    with open(transform_fp, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')
    print('Transform matrices are saved in \'{}\''.format(transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

spiral_indices_list = [
    utils.preprocess_spiral(tmp['face'][idx], args.seq_length[idx],
                            tmp['vertices'][idx],
                            args.dilation[idx]).to(device)
    for idx in range(len(tmp['face']) - 1)
]
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]

import os, json, shutil, time, math
from pathlib import Path
import torch
import pandas as pd

# --- config
ROOT_SAVE = "/scratch/xuankhoi/mesh/Explaining_Shape_Variability_CALSNIC/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_guided_1702logSAPparallel_latent16_leakage"
SAVE_IF_EUCLIDEAN_LT = 300.0  # strictly less than 300
SUMMARY_CSV = os.path.join(ROOT_SAVE, "trials_summary_0412.csv")  # the "sheet"

# --- utils
def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

_ensure_dir(ROOT_SAVE)  # make sure root exists once at import time

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
    # Model + config tensors
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

    # snapshot of index splits for traceability
    src_split = os.path.join(ROOT_SAVE, "..", "..", "..", "processed", "train_val_test_files.pt")
    if os.path.exists(src_split):
        shutil.copy(src_split, os.path.join(base_dir, "train_val_test_files.pt"))

    # optional: keep source files used for reconstruction (if present)
    for src, name in [
        ("/scratch/xuankhoi/mesh/Explaining_Shape_Variability_CALSNIC/src/DeepLearning/compute_canada/guided_vae/reconstruction/network.py", "network.py"),
        ("/scratch/xuankhoi/mesh/Explaining_Shape_VariABILITY_CALSNIC/src/DeepLearning/compute_canada/guided_vae/conv/spiralconv.py", "spiralconv.py"),
    ]:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(base_dir, name))


def _save_full_trial_bundle(
    base_dir: str,
    trial_number: int,
    model, args,
    spiral_indices_list, up_transform_list, down_transform_list, meshdata,
    euclidean_distance: float,
    trial_params: dict,
    latent_codes_t: Optional[torch.Tensor] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    _save_artifacts(base_dir, model, args, spiral_indices_list, up_transform_list, down_transform_list, meshdata)
    _dump_args_json(base_dir, args)

    # Optionally save latents
    if latent_codes_t is not None:
        torch.save(latent_codes_t.cpu(), os.path.join(base_dir, "latent_codes.pt"))
        pd.DataFrame(latent_codes_t.cpu().numpy()).to_csv(os.path.join(base_dir, "latent_codes.csv"), index=False)

    # Meta
    trial_meta = {
        "trial_number": trial_number,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": trial_params,
        "values": {
            "euclidean_distance": float(euclidean_distance),
        },
    }
    if extra:
        trial_meta["extra"] = extra

    with open(os.path.join(base_dir, "meta.json"), "w") as f:
        json.dump(trial_meta, f, indent=2)


def _append_summary_row(trial_number: int,
                        euclid: float,
                        sap_score: float,
                        params: dict):
    """
    Append (or create) a summary CSV:
      trial_number, euclidean_distance, sap_score, params_json
    """
    _ensure_dir(ROOT_SAVE)

    row = {
        "trial_number": int(trial_number),
        "euclidean_distance": float(euclid),
        "sap_score": float(sap_score),
        "params_json": json.dumps(params, sort_keys=True),
    }

    df_row = pd.DataFrame([row])
    if not os.path.exists(SUMMARY_CSV):
        df_row.to_csv(SUMMARY_CSV, index=False)
    else:
        df_row.to_csv(SUMMARY_CSV, mode="a", header=False, index=False)

def objective(trial):
    # ----- hyperparams suggested by Optuna -----
    args.epochs = trial.suggest_int("epochs", 200, 800, step=100)
    args.batch_size = trial.suggest_int("batch_size", 8, 32, step=4)

    args.wcls = trial.suggest_float("w_cls", 1e-2, 50.0, log=True)
    args.w_leak = trial.suggest_float("w_leak", 1e-2, 10.0, log=True)

    args.beta = trial.suggest_float("beta", 0.001, 1, log=True)
    args.lr = trial.suggest_float("learning_rate", 0.0001, 0.001, log=True)
    args.lr_decay = trial.suggest_float("learning_rate_decay", 0.70, 0.99, step=0.01)
    args.delta = trial.suggest_categorical("delta", [0.2, 0.4, 0.6, 0.8])
    args.decay_step = trial.suggest_int("decay_step", 1, 50)
    args.latent_channels = 16
    args.threshold = trial.suggest_categorical("threshold", [0.01, 0.02, 0.025, 0.05, 0.1])
    args.temperature = trial.suggest_float("temperature", 0.2, 2.0, log=True)
    # >>> enforce which dim is age
    args.age_dim = 1
    args.use_mu_for_guidance = True

    sequence_length = trial.suggest_int("sequence_length", 5, 50)
    args.seq_length = [sequence_length] * 4

    dilation = trial.suggest_int("dilation", 1, 2)
    args.dilation = [dilation] * 4

    out_channel = trial.suggest_int("out_channel", 8, 64, 8)
    args.out_channels = [out_channel, out_channel, out_channel, 2 * out_channel]

    print(args)
    print(
        "Split sizes:",
        "train =", len(meshdata.train_dataset),
        "val =", len(meshdata.val_dataset),
        "test =", len(meshdata.test_dataset),
    )

    # ----- model -----
    model = AE(
        args.in_channels,
        args.out_channels,
        args.latent_channels,
        spiral_indices_list,
        down_transform_list,
        up_transform_list,
    ).to(device)

    print("Number of parameters: {}".format(utils.count_parameters(model)))
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        args.decay_step,
        gamma=args.lr_decay,
    )

    # ----- BUILD LOADERS FOR THIS TRIAL USING args.batch_size -----
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
        f"[Trial {trial.number}] Using batch_size={args.batch_size} -> "
        f"train batches={len(train_loader)}, val batches={len(val_loader)}, "
        f"test batches={len(test_loader)}"
    )

    # ----- loss config -----
    args.guided = False
    args.guided_contrastive_loss = True
    args.attribute_loss = False
    args.correlation_loss = False

    # Sanity
    print("spiral0 N:", spiral_indices_list[0].shape[0])
    print("D0 shape:", down_transform_list[0].shape)
    print("mean/std:", meshdata.mean.shape, meshdata.std.shape)
    print("faces max idx:", int(meshdata.template_face.max()))

    assert spiral_indices_list[0].shape[0] == int(meshdata.template_face.max()) + 1
    assert meshdata.mean.shape == meshdata.std.shape == (spiral_indices_list[0].shape[0], 3)

    # ----- TRAIN -----
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

    # ----- EVAL: recon error on VAL -----
    euclidean_distance = eval_error(model, val_loader, device, meshdata, args.out_dir)

    # ----- Build latent codes + ages on VAL -----
    latent_codes_list, ages_list = [], []
    with torch.no_grad():
        for data in val_loader:
            x = data.x.to(device)
            y = data.y.to(device)
            B = y.view(-1).size(0)
            N = x.size(0)
            if N % B != 0:
                   raise RuntimeError(f"[VAL] Cannot reshape: x has {N} nodes, y batch {B}")
            V = N // B
            x = x.view(B, V, -1)
            _, mu, log_var, *_ = model(x)
            z = mu.view(mu.size(0), -1)
            z[torch.isnan(z) | torch.isinf(z)] = 0
            latent_codes_list.append(z.cpu())

            # force ages -> [B,1]
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

    latent_codes = torch.cat(latent_codes_list, dim=0)  # [N,D]
    ages = torch.cat(ages_list, dim=0)                  # [N,1]

    if latent_codes.shape[0] != ages.shape[0]:
        N = min(latent_codes.shape[0], ages.shape[0])
        print(
            f"[WARN] Mismatch in shapes: codes={latent_codes.shape[0]}, ages={ages.shape[0]} -> trimming to {N}"
        )
        latent_codes = latent_codes[:N]
        ages = ages[:N]

    print("Score debug:")
    print(" codes shape:", latent_codes.shape, "std:", latent_codes.std())
    print(" ages shape:", ages.shape, "std:", ages.std())

    sap_score = sap(
        factors=ages.numpy(),
        codes=latent_codes.numpy(),
        continuous_factors=True,
        regression=True,
    )

    # ----- log line per trial (only SAP now) -----
    out_error_fp = os.path.join(ROOT_SAVE, "test.txt")
    with open(out_error_fp, "a") as log_file:
        log_file.write(
            "Split=VAL | LatCh | Euclid | SAP | w_cls | w_leak | age_dim | Trial | "
            f"{args.latent_channels:d} | {euclidean_distance:.3f} | {sap_score:.6f} | "
            f"{args.wcls:.4g} | {args.w_leak:.4g} | {args.age_dim:d} | {trial.number:d}\n"
        )

    # ---- summary CSV (only SAP now) ----
    trial_params_for_log = dict(trial.params)  # keep optuna params
    # (optional) also store a couple of fixed args you care about:
    trial_params_for_log.update({
        "age_dim": args.age_dim,
        "use_mu_for_guidance": bool(args.use_mu_for_guidance),
        "guided_contrastive_loss": bool(args.guided_contrastive_loss),
    })

    _append_summary_row(
        trial_number=trial.number,
        euclid=euclidean_distance,
        sap_score=sap_score,
        params=trial_params_for_log,
    )

    # ----- Save full bundle for every trial (remove fixed_sap fields) -----
    trial_params_extra = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "wcls": args.wcls,
        "w_leak": args.w_leak,
        "age_dim": args.age_dim,
        "use_mu_for_guidance": args.use_mu_for_guidance,
        "beta": args.beta,
        "lr": args.lr,
        "lr_decay": args.lr_decay,   # <<< FIXED: you had a typo lr_decsap_scoreay
        "delta": args.delta,
        "decay_step": args.decay_step,
        "latent_channels": args.latent_channels,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "seq_length": args.seq_length,
        "dilation": args.dilation,
        "out_channels": args.out_channels,
        # scores
        "euclidean_distance_val": float(euclidean_distance),
        "sap_score_val": float(sap_score),
    }

    trial_dir = os.path.join(ROOT_SAVE, f"{trial.number}")
    _save_full_trial_bundle(
        base_dir=trial_dir,
        trial_number=trial.number,
        model=model,
        args=args,
        spiral_indices_list=spiral_indices_list,
        up_transform_list=up_transform_list,
        down_transform_list=down_transform_list,
        meshdata=meshdata,
        euclidean_distance=euclidean_distance,
        trial_params=trial.params,
        latent_codes_t=latent_codes,
        extra=trial_params_extra,
    )

    # IMPORTANT: directions=["minimize","maximize"] -> (euclid, sap)
    return euclidean_distance, sap_score

class LogAfterEachTrial:
    def __init__(self, save_every=10):
        self.save_every = save_every

    def __call__(self, study, trial):
        if trial.number % self.save_every != 0:
            return
        torch.save(
            study.trials,
            "/scratch/xuankhoi/mesh/Explaining_Shape_Variability_CALSNIC/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_contrastive_inhib/intermediate_trials.pt"
        )

log_trials = LogAfterEachTrial(save_every=10)


sampler = optuna.samplers.NSGAIISampler(
    population_size=32,   # 16–64; bigger = better search, slower per generation
    crossover_prob=0.9,
    mutation_prob=0.2,
)

import sqlite3

db_path = "/scratch/xuankhoi/optuna_multiobj_latent16_1702_correctSNN_leakage.db"

con = sqlite3.connect(db_path, timeout=60)
con.execute("PRAGMA journal_mode=WAL;")
con.execute("PRAGMA synchronous=NORMAL;")
con.execute("PRAGMA busy_timeout=60000;")
con.commit()
con.close()

STUDY_NAME = "calsnic_mo_v1"
DB_PATH = "/scratch/xuankhoi/optuna_multiobj_test_1702_correctSNN_leakage.db"
STORAGE = f"sqlite:///{DB_PATH}"

study = optuna.create_study(
    study_name=STUDY_NAME,
    directions=["minimize", "maximize"],
    sampler=sampler,
    storage=STORAGE,
    load_if_exists=True,
)


study.optimize(objective, n_trials=cli.n_trials, callbacks=[log_trials])


# Print Pareto front
print("\nPareto-optimal trials (best trade-offs):")
for t in study.best_trials:
    euc, sap_sc = t.values
    print(f"- Trial {t.number:4d} | Euclid: {euc:.4f} | SAP: {sap_sc:.4f} | params: {t.params}")

# Also show the single-metric winners
best_euclid_trial = min(study.trials, key=lambda tr: tr.values[0] if tr.values is not None else float('inf'))
best_sap_trial    = max(study.trials, key=lambda tr: tr.values[1] if tr.values is not None else float('-inf'))
print(f"\nBest Euclid -> Trial {best_euclid_trial.number}, Euclid={best_euclid_trial.values[0]:.4f}, SAP={best_euclid_trial.values[1]:.4f}")
print(f"Best SAP    -> Trial {best_sap_trial.number}, Euclid={best_sap_trial.values[0]:.4f}, SAP={best_sap_trial.values[1]:.4f}")
