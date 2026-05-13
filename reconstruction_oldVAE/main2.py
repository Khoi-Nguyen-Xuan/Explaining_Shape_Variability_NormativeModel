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
from utils import utils, writer, DataLoader, mesh_sampling, sap, point_biserial_correlation
import optuna
from contextlib import redirect_stdout
import shutil
import random
from sklearn.metrics import accuracy_score
from torch.utils.data import ConcatDataset
from typing import Optional, Dict, Any
import re 

parser = argparse.ArgumentParser(description='mesh autoencoder')
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

# --- assume imports & args parsing already done above ---
# (utils, writer, MeshData, Mesh, mesh_sampling, AE, run, eval_error, optuna, etc.)
# If you need typing: from typing import Optional

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
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"

set_seed(args.seed)

# -----------------------------
# load dataset
# -----------------------------
print(args.data_fp)
template_fp = osp.join(args.data_fp, 'template', 'template.ply')
print(template_fp)
meshdata = MeshData(args.data_fp,
                    template_fp,
                    split=args.split,
                    test_exp=args.test_exp)

# -------- USE FULL SPLITS (no slicing) ----------
from utils import DataLoader
train_loader = DataLoader(meshdata.train_dataset, batch_size=1, shuffle=False, drop_last=False)
val_loader   = DataLoader(meshdata.val_dataset,   batch_size=1, shuffle=False, drop_last=False)
test_loader  = DataLoader(meshdata.test_dataset,  batch_size=1, shuffle=False, drop_last=False)
print(f"Full splits -> train: {len(meshdata.train_dataset)} | val: {len(meshdata.val_dataset)} | test: {len(meshdata.test_dataset)}")

# -----------------------------
# generate/load transform matrices
# -----------------------------
transform_fp = osp.join(args.data_fp, 'transform', 'transform.pkl')
if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)
    ds_factors = [2, 2, 2, 2]
    _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
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
    print(f"Transform matrices are saved in '{transform_fp}'")
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

spiral_indices_list = [
    utils.preprocess_spiral(tmp['face'][idx], args.seq_length[idx],
                            tmp['vertices'][idx], args.dilation[idx]).to(device)
    for idx in range(len(tmp['face']) - 1)
]
down_transform_list = [utils.to_sparse(d).to(device) for d in tmp['down_transform']]
up_transform_list   = [utils.to_sparse(u).to(device) for u in tmp['up_transform']]

# -----------------------------
# trial I/O helpers (kept simple)
# -----------------------------
import os, json, shutil, time, math
from pathlib import Path
import pandas as pd
from typing import Optional

ROOT_SAVE = "/scratch/xuankhoi/mesh/Explaining_Shape_Variability_CALSNIC/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_guided_3_VAE"
SUMMARY_CSV = os.path.join(ROOT_SAVE, "trials_summary.csv")

def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

_ensure_dir(ROOT_SAVE)

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
    torch.save(args.in_channels,   os.path.join(base_dir, "in_channels.pt"))
    torch.save(args.out_channels,  os.path.join(base_dir, "out_channels.pt"))
    torch.save(args.latent_channels, os.path.join(base_dir, "latent_channels.pt"))
    torch.save(spiral_indices_list, os.path.join(base_dir, "spiral_indices_list.pt"))
    torch.save(up_transform_list,   os.path.join(base_dir, "up_transform_list.pt"))
    torch.save(down_transform_list, os.path.join(base_dir, "down_transform_list.pt"))
    torch.save(meshdata.std,        os.path.join(base_dir, "std.pt"))
    torch.save(meshdata.mean,       os.path.join(base_dir, "mean.pt"))
    torch.save(meshdata.template_face, os.path.join(base_dir, "faces.pt"))

    src_split = os.path.join(ROOT_SAVE, "..", "..", "..", "processed", "train_val_test_files.pt")
    if os.path.exists(src_split):
        shutil.copy(src_split, os.path.join(base_dir, "train_val_test_files.pt"))

    for src, name in [
        ("/scratch/xuankhoi/mesh/Explaining_Shape_Variability_CALSNIC/src/DeepLearning/compute_canada/guided_vae/reconstruction/network.py", "network.py"),
        ("/scratch/xuankhoi/mesh/Explaining_Shape_Variability_CALSNIC/src/DeepLearning/compute_canada/guided_vae/conv/spiralconv.py", "spiralconv.py"),
    ]:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(base_dir, name))

def _save_full_trial_bundle(
    base_dir: str,
    trial_number: int,
    model, args,
    spiral_indices_list, up_transform_list, down_transform_list, meshdata,
    euclidean_distance: float,
    sap_score: float,
    trial_params: dict,
    latent_codes_t: Optional[torch.Tensor] = None,
    extra: Optional[dict] = None,
)->None:
    _save_artifacts(base_dir, model, args, spiral_indices_list, up_transform_list, down_transform_list, meshdata)
    _dump_args_json(base_dir, args)

    if latent_codes_t is not None:
        torch.save(latent_codes_t.cpu(), os.path.join(base_dir, "latent_codes.pt"))
        pd.DataFrame(latent_codes_t.cpu().numpy()).to_csv(os.path.join(base_dir, "latent_codes.csv"), index=False)

    trial_meta = {
        "trial_number": trial_number,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": trial_params,
        "values": {
            "euclidean_distance": float(euclidean_distance),
            "sap_score": float(sap_score),
        },
    }
    if extra:
        trial_meta["extra"] = extra

    with open(os.path.join(base_dir, "meta.json"), "w") as f:
        json.dump(trial_meta, f, indent=2)

def _append_summary_row(trial_number: int, euclid: float, sap_score: float, params: dict)-> None:
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

# -----------------------------
# Multi-objective objective()
# -----------------------------
def objective(trial):
    # ----- hyperparams -----
    args.epochs        = trial.suggest_int("epochs", 100, 400, step=100)
    args.batch_size    = trial.suggest_int("batch_size", 4, 32, step=4)
    args.wcls          = trial.suggest_int("w_cls", 1, 100)
    args.beta          = trial.suggest_float("beta", 0.001, 0.3, log=True)
    args.lr            = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    args.lr_decay      = trial.suggest_float("learning_rate_decay", 0.70, 0.99, step=0.01)
    args.delta         = trial.suggest_float("delta", 0.1, 0.9, step=0.1)
    args.decay_step    = trial.suggest_int("decay_step", 1, 50)
    args.latent_channels = trial.suggest_int("latent_channels", 12, 32, step=4)
    args.threshold     = trial.suggest_float("threshold", 0.005, 0.05, step=0.005)
    args.temperature   = trial.suggest_int("temperature", 1, 200, step=20)

    sequence_length    = trial.suggest_int("sequence_length", 5, 50)
    args.seq_length    = [sequence_length]*4

    dilation           = trial.suggest_int("dilation", 1, 2)
    args.dilation      = [dilation]*4
    
    out_channel        = trial.suggest_int("out_channel", 8, 64, step=8)
    args.out_channels  = [out_channel, out_channel, out_channel, 2*out_channel]
    print(args)

    # ----- model/opt -----
    model = AE(
        args.in_channels, args.out_channels, args.latent_channels,
        spiral_indices_list, down_transform_list, up_transform_list
    ).to(device)

    print('Number of parameters: {}'.format(utils.count_parameters(model)))
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_step, gamma=args.lr_decay)

    # switches
    args.guided = False
    args.guided_contrastive_loss = True   # age-guided contrastive
    args.attribute_loss = False
    args.correlation_loss = False
    args.tc = False

    # ----- load labels -----
    LABELS_FP  = osp.join(args.data_fp,'raw', 'torus', 'labels.pt')
    labels_map = torch.load(LABELS_FP, map_location='cpu')
    label_keys = set(labels_map.keys())

    # small resolver to avoid KeyError(None)
    def _resolve_sid_from_data(data_obj) -> str:
        # try common attrs
        sid = getattr(data_obj, 'name', None) or getattr(data_obj, 'sid', None) or getattr(data_obj, 'subject', None)
        # fallback to file-like attrs
        if sid is None:
            p = getattr(data_obj, 'path', None) or getattr(data_obj, 'file', None) or getattr(data_obj, 'fname', None)
            if p is not None:
                bn = os.path.basename(str(p))
                sid = os.path.splitext(bn)[0]
        # if tensor/list, normalize
        if isinstance(sid, (list, tuple)):
            cand = None
            for s in sid:
                s = str(s)
                if s in label_keys:
                    cand = s; break
            if cand is None:
                # substring fallback
                for s in sid:
                    s = str(s)
                    cand = next((k for k in label_keys if k in s or s in k), None)
                    if cand is not None:
                        break
            return cand
        # single str
        if sid is None:
            return None
        sid = str(sid)
        if sid in label_keys:
            return sid
        # substring fallback
        return next((k for k in label_keys if k in sid or sid in k), None)

    # ----- train -----
    run(
        model, train_loader, val_loader, args.epochs, optimizer, scheduler,
        writer, device, args.beta, args.wcls, args.guided, args.guided_contrastive_loss,
        args.correlation_loss, args.attribute_loss, args.latent_channels,
        args.weight_decay_c, args.temperature, args.delta, args.threshold,
        labels_map
    )

    # ----- metric 1: Euclidean distance (↓)
    euclidean_distance = float(eval_error(model, val_loader, device, meshdata, args.out_dir))

    # ----- metric 2: SAP on AGE (↑) with safe SID resolution
    latent_codes = []
    ages = []
    warn_printed = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            x = data.x.to(device)

            out = model(x)
            if isinstance(out, (list, tuple)) and len(out) >= 3:
                recon, mu, log_var = out[0], out[1], out[2]
                z = model.reparameterize(mu, log_var)
            else:
                mu, log_var = model.encoder(x)
                z = mu
            z[z.isnan() | z.isinf()] = 0

            sid_key = _resolve_sid_from_data(data)
            if sid_key is None:
                if warn_printed < 6:
                    print(f"[WARN][SAP] Unmatched SID for val sample #{i}; skipping.")
                    warn_printed += 1
                continue

            age = float(labels_map[sid_key].item())
            latent_codes.append(z)
            ages.append(age)

    if len(ages) == 0:
        # if nothing matched, don't crash the trial — penalize SAP
        print("[WARN][SAP] No matched validation samples to labels.pt; setting SAP=0.0 for this trial.")
        sap_score = 0.0
    else:
        latent_codes = torch.cat(latent_codes, dim=0)
        ages_np = torch.tensor(ages, dtype=torch.float32).view(-1, 1).cpu().numpy()
        sap_score = float(sap(
            factors=ages_np,
            codes=latent_codes.cpu().numpy(),
            continuous_factors=True,
            regression=True
        ))

    # ----- log minimal text
    out_error_fp = os.path.join(ROOT_SAVE, "test.txt")
    _ensure_dir(os.path.dirname(out_error_fp))
    with open(out_error_fp, 'a') as log_file:
        log_file.write(
            "LatCh | Epochs | Euclid | SAP | Trial | "
            f"{args.latent_channels:d} | {args.epochs:d} | {euclidean_distance:.4f} | {sap_score:.4f} | {trial.number:d}\n"
        )

    # ----- save bundle
    trial_dir = os.path.join(ROOT_SAVE, f"{trial.number}")
    _ensure_dir(trial_dir)
    torch.save(euclidean_distance,       f"{trial_dir}/euclidean_distance.pt")
    torch.save(sap_score,                f"{trial_dir}/sap_score.pt")
    torch.save(model.state_dict(),       f"{trial_dir}/model_state_dict.pt")
    torch.save(args.in_channels,         f"{trial_dir}/in_channels.pt")
    torch.save(args.out_channels,        f"{trial_dir}/out_channels.pt")
    torch.save(args.latent_channels,     f"{trial_dir}/latent_channels.pt")
    torch.save(spiral_indices_list,      f"{trial_dir}/spiral_indices_list.pt")
    torch.save(up_transform_list,        f"{trial_dir}/up_transform_list.pt")
    torch.save(down_transform_list,      f"{trial_dir}/down_transform_list.pt")
    torch.save(meshdata.std,             f"{trial_dir}/std.pt")
    torch.save(meshdata.mean,            f"{trial_dir}/mean.pt")
    torch.save(meshdata.template_face,   f"{trial_dir}/faces.pt")
    split_src = os.path.join(args.data_fp, "processed", "train_val_test_files.pt")
    if os.path.exists(split_src):
        shutil.copy(split_src, f"{trial_dir}/train_val_test_files.pt")

    _append_summary_row(trial.number, euclidean_distance, sap_score, trial.params)

    # multi-objective: (Euclid ↓, SAP ↑)
    return euclidean_distance, sap_score


class LogAfterEachTrial:
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        trials = study.trials
        torch.save(
            trials,
            "/scratch/xuankhoi/mesh/Explaining_Shape_Variability_CALSNIC/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_contrastive_inhib/intermediate_trials_2.pt",
        )

log_trials = LogAfterEachTrial()
study = optuna.create_study(directions=['minimize', 'maximize'])
study.optimize(objective, n_trials=50, callbacks=[log_trials])

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
