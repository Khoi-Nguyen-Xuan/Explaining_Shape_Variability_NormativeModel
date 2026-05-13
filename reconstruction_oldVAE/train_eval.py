import time
import math
import os
import torch
import torch.nn.functional as F
from reconstruction import Regressor, Classifier
from reconstruction.loss import ClsCorrelationLoss, RegCorrelationLoss, SNNLoss, SNNRegLoss, WassersteinLoss, age_leakage_loss
from utils import DataLoader
from torch.utils.data import Subset
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
import math, numpy as np
from torch_geometric.loader import DataLoader as GeoDataLoader
import optuna
from typing import Tuple

def _flatten_to_2d(t: torch.Tensor) -> torch.Tensor:
    """
    Make sure tensor is [N, 3] for loss computation.
    Accepts [V, 3] or [B, V, 3].
    """
    if t.dim() == 3 and t.size(-1) == 3:
        # [B, V, 3] -> [B*V, 3]
        return t.reshape(-1, 3)
    elif t.dim() == 2 and t.size(1) == 3:
        # [N, 3] already
        return t
    else:
        raise RuntimeError(f"Unexpected tensor shape for loss: {t.shape}")


def _aligned_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Flatten a,b to [N,3], cut both to the same length, compute mean L1.
    Symmetric: _aligned_l1(a,b) == _aligned_l1(b,a).
    """
    a2 = _flatten_to_2d(a)
    b2 = _flatten_to_2d(b)

    if a2.size(1) != b2.size(1):
        raise RuntimeError(
            f"Channel mismatch in aligned L1: "
            f"a has C={a2.size(1)}, b has C={b2.size(1)}"
        )

    n = min(a2.size(0), b2.size(0))
    if n == 0:
        raise RuntimeError(
            f"No overlapping vertices: a={a2.shape}, b={b2.shape}"
        )

    a2 = a2[:n]
    b2 = b2[:n]
    return F.l1_loss(a2, b2, reduction='mean')


def loss_function(original, reconstruction, mu, log_var, beta):
    # reconstruction vs original, but shape-safe
    reconstruction_loss = _aligned_l1(reconstruction, original)

    # KL term (same as before)
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = reconstruction_loss + beta * KLD
    return total_loss


def _make_subset_loader(train_loader, delta):
    dataset = train_loader.dataset
    N = len(dataset)

    # delta can be a fraction (0..1] or an absolute count (>1)
    if 0 < float(delta) <= 1.0:
        subN = max(1, int(math.ceil(float(delta) * N)))
    else:
        subN = min(N, int(delta))

    idxs = np.random.choice(N, size=subN, replace=False).tolist()

    return DataLoader(
        dataset,
        batch_size=train_loader.batch_size,
        sampler=SubsetRandomSampler(idxs),
        shuffle=False,       # important when sampler is used
        drop_last=False,
        num_workers=getattr(train_loader, "num_workers", 0),
        pin_memory=getattr(train_loader, "pin_memory", False),
    )


def compute_fixed_sap_val(model, val_loader, device, age_dim, use_mu_for_guidance=True):
    model.eval()

    latent_codes_list, ages_list = [], []

    with torch.no_grad():
        for data in val_loader:
            x = data.x.to(device)
            y = data.y.to(device)

            _, mu, log_var, *_ = model(x)

            if use_mu_for_guidance:
                z = mu.view(mu.size(0), -1)
            else:
                # if you want sampled code, but mu is more stable
                z = model.reparameterize(mu, log_var).view(mu.size(0), -1)

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

    # trim if mismatch
    if latent_codes.shape[0] != ages.shape[0]:
        N = min(latent_codes.shape[0], ages.shape[0])
        latent_codes = latent_codes[:N]
        ages = ages[:N]

    fixed_score, r2_target, r2_best_other = sap_fixed_age(
        factors=ages.numpy(),
        codes=latent_codes.numpy(),
        age_dim=age_dim
    )
    return float(fixed_score), float(r2_target), float(r2_best_other)


def run(
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    scheduler,
    writer,
    device,
    beta,
    w_cls,
    w_leak,
    age_dim,
    use_mu_for_guidance,
    guided,
    guided_contrastive_loss,
    correlation_loss,
    latent_channels,
    weight_decay_c,
    temperature,
    delta,
    threshold,
):
    model_c = Classifier(latent_channels).to(device)
    optimizer_c = torch.optim.Adam(model_c.parameters(), lr=1e-3, weight_decay=weight_decay_c)

    model_c_2 = Regressor(latent_channels).to(device)
    optimizer_c_2 = torch.optim.Adam(model_c_2.parameters(), lr=1e-3, weight_decay=weight_decay_c)

    train_losses, test_losses = [], []

    base_dir = "/scratch/xuankhoi/mesh/Explaining_Shape_Variability_CALSNIC/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/models_con_base"
    os.makedirs(base_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t = time.time()

        train_loss = train(
            model,
            optimizer,
            model_c,
            optimizer_c,
            model_c_2,
            optimizer_c_2,
            train_loader,
            device,
            beta,
            w_cls,
            guided,
            guided_contrastive_loss,
            correlation_loss,
            temperature,
            delta,
            threshold,
            age_dim=age_dim,
            w_leak=w_leak,
            use_mu_for_guidance=use_mu_for_guidance,
        )

        t_duration = time.time() - t

        test_loss = test(model, val_loader, device, beta)

        scheduler.step()

        info = {
            "current_epoch": epoch,
            "epochs": epochs,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "t_duration": t_duration,
        }
        writer.print_info(info)

        writer.save_checkpoint(model, optimizer, scheduler, epoch)

        torch.save(
            model.state_dict(),
            os.path.join(base_dir, "model_state_dict.pt"),
        )
        torch.save(
            model_c.state_dict(),
            os.path.join(base_dir, "model_c_state_dict.pt"),
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)



import math
import random
import torch
from torch.utils.data import Subset

try:
    from torch_geometric.loader import DataLoader as GeoDataLoader
except Exception:
    GeoDataLoader = None


def train(
    model,
    optimizer,
    model_c, optimizer_c,
    model_c_2, optimizer_c_2,
    loader,
    device,
    beta,
    w_cls,
    guided,
    guided_contrastive_loss,
    correlation_loss,
    temp,
    delta,
    threshold,
    age_dim=1,                 # 0-indexed latent dim you want to reserve for age (used by leakage)
    w_leak=1.0,
    use_mu_for_guidance=True,
):
    model.train()
    if model_c is not None:
        model_c.train()
    if model_c_2 is not None:
        model_c_2.train()

    # Repo-style SNN that internally uses supervised_dim=1
    snn_loss_fn = SNNRegLoss(
        T=temp,
        threshold=threshold,
        lamda1=1.0,
        lamda2=1.0,
        supervised_dim=1,   # <-- EXPLICITLY supervise latent dim 1 inside the loss
    )

    total_loss = 0.0
    total_recon_kl = 0.0
    total_snn = 0.0
    total_snn_weighted = 0.0
    total_leak = 0.0
    total_leak_weighted = 0.0
    n_batches = 0

    dataset = loader.dataset
    total_data = len(dataset)

    # ---- delta subsample ----
    if delta is None or float(delta) <= 0:
        subset_loader = loader
    else:
        d = float(delta)
        if 0.0 < d <= 1.0:
            desired_data = math.ceil(d * total_data)
        else:
            desired_data = min(int(d), total_data)

        idx = list(range(total_data))
        random.shuffle(idx)
        subset_ds = Subset(dataset, idx[:desired_data])

        if GeoDataLoader is None:
            raise RuntimeError("torch_geometric GeoDataLoader not available but needed for your dataset.")
        subset_loader = GeoDataLoader(
            subset_ds,
            batch_size=loader.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=getattr(loader, "num_workers", 0),
            pin_memory=getattr(loader, "pin_memory", False),
        )

    print(f"[train] total_data={total_data}, using batches={len(subset_loader)}, batch_size={loader.batch_size}")

    for batch_idx, data in enumerate(subset_loader):
        x = data.x.to(device)
        y = data.y.to(device)

        # ---- age: [B] ----
        if y.dim() == 0:
            age = y.view(1)
        elif y.dim() == 1:
            age = y.contiguous().view(-1)
        elif y.dim() == 2 and y.size(1) == 1:
            age = y[:, 0].contiguous().view(-1)
        else:
            raise RuntimeError(f"Expected y to contain ONLY age with shape [B] or [B,1], got {tuple(y.shape)}")

        B = age.numel()

        # ---- reshape x using B ----
        N = x.size(0)
        if N % B != 0:
            raise RuntimeError(f"Cannot reshape: x has {N} nodes, but batch B={B} from y.shape={tuple(y.shape)}")
        V = N // B
        x = x.view(B, V, -1)

        optimizer.zero_grad()
        out, mu, log_var, *_ = model(x)

        recon_kl = loss_function(x, out, mu, log_var, beta)
        loss = recon_kl

        loss_snn_reg = x.new_tensor(0.0)
        loss_leak = x.new_tensor(0.0)

        if guided_contrastive_loss:
            # latent codes
            if use_mu_for_guidance:
                z_full = mu.view(mu.size(0), -1)  # [B, D]
            else:
                z_full = model.reparameterize(mu, log_var).view(mu.size(0), -1)

            D = z_full.size(1)
            if not (0 <= age_dim < D):
                raise RuntimeError(f"age_dim={age_dim} out of range for latent dim D={D}")

            # normalize age to match repo-style threshold scale
            age = age.float()
            age_n = (age - age.min()) / (age.max() - age.min() + 1e-8)

            # (1) repo-style SNN (uses supervised_dim=1 internally)
            loss_snn_reg = snn_loss_fn(z_full, age_n)

            # (2) leakage on ALL dims except age_dim (use normalized age for consistency)
            loss_leak = age_leakage_loss(z_full, age_n, age_dim=age_dim)

            # combine
            loss = recon_kl + (w_cls * loss_snn_reg) + (w_leak * loss_leak)

            if batch_idx == 0:
                # optional sanity: positives count
                with torch.no_grad():
                    abs_diff = (age_n.unsqueeze(0) - age_n.unsqueeze(1)).abs()
                    same = (abs_diff <= threshold).float()
                    same.fill_diagonal_(0)
                    avg_pos = same.sum(1).mean().item()
                print(f"[DEBUG] y.shape={tuple(y.shape)} age.shape={tuple(age.shape)}")
                print(f"[DEBUG] z_full={tuple(z_full.shape)} age_dim(leak-exempt)={age_dim}")
                print(f"[DEBUG] avg positives/anchor={avg_pos:.2f} (B={B}, threshold={threshold})")
                print(f"[DEBUG] SNN={loss_snn_reg.item():.4e}, LEAK={loss_leak.item():.4e}, w_cls={w_cls:.4g}, w_leak={w_leak:.4g}")

        loss.backward()
        optimizer.step()

        # ---- stats ----
        total_loss += float(loss.item())
        total_recon_kl += float(recon_kl.item())
        total_snn += float(loss_snn_reg.item())
        total_snn_weighted += float((loss_snn_reg * w_cls).item())
        total_leak += float(loss_leak.item())
        total_leak_weighted += float((loss_leak * w_leak).item())
        n_batches += 1

        if batch_idx < 3:
            print(
                f"[train batch {batch_idx}] "
                f"recon+KL={recon_kl.item():.4e}, "
                f"SNN={loss_snn_reg.item():.4e}, w*SNN={(loss_snn_reg*w_cls).item():.4e}, "
                f"LEAK={loss_leak.item():.4e}, w*LEAK={(loss_leak*w_leak).item():.4e}, "
                f"total={loss.item():.4e}"
            )

    if n_batches == 0:
        return 0.0

    avg_loss = total_loss / n_batches
    avg_recon_kl = total_recon_kl / n_batches
    avg_snn = total_snn / n_batches
    avg_snn_weighted = total_snn_weighted / n_batches
    avg_leak = total_leak / n_batches
    avg_leak_weighted = total_leak_weighted / n_batches

    print(
        f"[train epoch summary] "
        f"avg_total={avg_loss:.4e}, "
        f"avg_recon+KL={avg_recon_kl:.4e}, "
        f"avg_SNN={avg_snn:.4e}, avg_w*SNN={avg_snn_weighted:.4e}, "
        f"avg_LEAK={avg_leak:.4e}, avg_w*LEAK={avg_leak_weighted:.4e}"
    )

    return avg_loss


def test(model, loader, device, beta):
    model.eval()
    model.training = False

    total_loss = 0
    recon_loss = 0
    reg_loss = 0
    reg_loss_2 = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.x.to(device)
            has_nan = torch.isnan(x).any().item()
            if has_nan:
                continue
            y = data.y.to(device)
            B = y.view(-1).size(0)
            N = x.size(0)

            if N % B != 0:
                raise RuntimeError(f"Cannot reshape in test: x has {N} nodes, y has batch {B}")
            V = N // B
            x = x.view(B, V, -1)
            pred, mu, log_var, re, re_2 = model(x)
            has_nan_1 = torch.isnan(re).any().item()
            if has_nan_1:
                continue
            #print(re.shape)
            total_loss += loss_function(x, pred, mu, log_var, beta)
            recon_loss += _aligned_l1(pred, x)
            reg_loss += F.mse_loss(re.squeeze(-1), y[..., 0], reduction='mean')

    return total_loss / len(loader)

import os
import torch

def eval_error(model, loader, device, meshdata, out_dir):
    model.eval()

    mean = meshdata.mean.to(device)   # [V,3]
    std  = meshdata.std.to(device)    # [V,3]
    V = int(meshdata.num_nodes)

    errors = []

    with torch.no_grad():
        for data in loader:
            x = data.x.to(device)  # [N,3] (node-stacked)
            y = data.y.to(device)  # [B] or [B,1]  (graph labels)

            B = y.view(-1).size(0)
            N = x.size(0)

            if N % B != 0:
                raise RuntimeError(f"eval_error: Cannot reshape: x has N={N} nodes, batch B={B}")
            V_here = N // B
            if V_here != V:
                raise RuntimeError(f"eval_error: V mismatch: expected {V}, got {V_here} (N={N}, B={B})")

            x = x.view(B, V, 3)  # [B,V,3]

            pred, mu, log_var, *_ = model(x)  # pred should be [B,V,3] for AE

            if pred.shape != x.shape:
                raise RuntimeError(f"eval_error: pred shape {tuple(pred.shape)} != x shape {tuple(x.shape)}")

            # denorm in device space
            x_denorm = x * std.view(1, V, 3) + mean.view(1, V, 3)
            p_denorm = pred * std.view(1, V, 3) + mean.view(1, V, 3)

            # to mm (your factor)
            x_denorm = x_denorm * 300.0
            p_denorm = p_denorm * 300.0

            # per-vertex euclidean: [B,V]
            e = torch.sqrt(((p_denorm - x_denorm) ** 2).sum(dim=2))
            errors.append(e.detach().cpu())

    if len(errors) == 0:
        return 0.0

    all_e = torch.cat(errors, dim=0)          # [Ngraphs, V]
    mean_error = all_e.mean().item()
    std_error = all_e.std().item()
    median_error = all_e.median().item()

    message = f"Euclidean Error: {mean_error:.3f}+{std_error:.3f} | {median_error:.3f}"

    os.makedirs(out_dir, exist_ok=True)
    out_error_fp = os.path.join(out_dir, "euc_errors.txt")
    with open(out_error_fp, "a") as f:
        f.write(message + "\n")

    print("\n\n" + message + "\n\n")
    return mean_error
