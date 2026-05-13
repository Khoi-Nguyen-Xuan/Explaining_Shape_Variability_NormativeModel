import os
import time
import torch
import torch.nn.functional as F


# ============================================================
# Loss helpers
# ============================================================

def _flatten_to_2d(t: torch.Tensor) -> torch.Tensor:
    """
    Make sure tensor is [N, 3].
    Accepts [V, 3] or [B, V, 3].
    """
    if t.dim() == 3 and t.size(-1) == 3:
        return t.reshape(-1, 3)

    if t.dim() == 2 and t.size(1) == 3:
        return t

    raise RuntimeError(f"Unexpected tensor shape for loss: {tuple(t.shape)}")


def _aligned_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Flatten a,b to [N,3], cut both to same length, compute mean L1.
    """
    a2 = _flatten_to_2d(a)
    b2 = _flatten_to_2d(b)

    if a2.size(1) != b2.size(1):
        raise RuntimeError(
            f"Channel mismatch: a has {a2.size(1)}, b has {b2.size(1)}"
        )

    n = min(a2.size(0), b2.size(0))

    if n == 0:
        raise RuntimeError(f"No overlapping vertices: a={a2.shape}, b={b2.shape}")

    return F.l1_loss(a2[:n], b2[:n], reduction="mean")


def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Standard VAE KL divergence.

    mu/log_var shape: [B, latent_dim]
    """
    return -0.5 * torch.mean(
        torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp(),
            dim=1,
        )
    )


def loss_function(original, reconstruction, mu, log_var, beta):
    recon_loss = _aligned_l1(reconstruction, original)
    kld = kl_loss(mu, log_var)

    total_loss = recon_loss + beta * kld

    return total_loss, recon_loss, kld


# ============================================================
# Shape helper
# ============================================================

def reshape_pyg_batch(data, device):
    """
    PyG stores batched mesh nodes as [B*V, 3].
    This converts it back to [B, V, 3].

    We use data.y only to infer B.
    Age/labels are NOT used for training.
    """
    x = data.x.to(device)

    if not hasattr(data, "y") or data.y is None:
        raise RuntimeError(
            "data.y is required to infer batch size B. "
            "Even in vanilla VAE, labels are used only for reshaping."
        )

    y = data.y.to(device)
    B = y.view(-1).size(0)
    N = x.size(0)

    if N % B != 0:
        raise RuntimeError(
            f"Cannot reshape PyG batch: x has {N} nodes, but B={B}"
        )

    V = N // B
    x = x.view(B, V, -1)

    return x


# ============================================================
# Train / test
# ============================================================

def train(
    model,
    optimizer,
    loader,
    device,
    beta,
):
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch_idx, data in enumerate(loader):
        x = reshape_pyg_batch(data, device)

        optimizer.zero_grad()

        out, mu, log_var, *_ = model(x)

        loss, recon, kld = loss_function(
            original=x,
            reconstruction=out,
            mu=mu,
            log_var=log_var,
            beta=beta,
        )

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_recon += float(recon.item())
        total_kl += float(kld.item())
        n_batches += 1

        if batch_idx < 3:
            print(
                f"[train batch {batch_idx}] "
                f"loss={loss.item():.4e}, "
                f"recon={recon.item():.4e}, "
                f"KL={kld.item():.4e}, "
                f"beta*KL={(beta * kld).item():.4e}"
            )

    if n_batches == 0:
        return 0.0

    avg_loss = total_loss / n_batches
    avg_recon = total_recon / n_batches
    avg_kl = total_kl / n_batches

    print(
        f"[train summary] "
        f"avg_loss={avg_loss:.4e}, "
        f"avg_recon={avg_recon:.4e}, "
        f"avg_KL={avg_kl:.4e}, "
        f"beta={beta:.4g}"
    )

    return avg_loss


def test(
    model,
    loader,
    device,
    beta,
):
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    with torch.no_grad():
        for data in loader:
            x = reshape_pyg_batch(data, device)

            pred, mu, log_var, *_ = model(x)

            loss, recon, kld = loss_function(
                original=x,
                reconstruction=pred,
                mu=mu,
                log_var=log_var,
                beta=beta,
            )

            total_loss += float(loss.item())
            total_recon += float(recon.item())
            total_kl += float(kld.item())
            n_batches += 1

    if n_batches == 0:
        return 0.0

    avg_loss = total_loss / n_batches
    avg_recon = total_recon / n_batches
    avg_kl = total_kl / n_batches

    print(
        f"[val summary] "
        f"avg_loss={avg_loss:.4e}, "
        f"avg_recon={avg_recon:.4e}, "
        f"avg_KL={avg_kl:.4e}"
    )

    return avg_loss


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
):
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        t = time.time()

        train_loss = train(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            beta=beta,
        )

        val_loss = test(
            model=model,
            loader=val_loader,
            device=device,
            beta=beta,
        )

        scheduler.step()

        t_duration = time.time() - t

        info = {
            "current_epoch": epoch,
            "epochs": epochs,
            "train_loss": train_loss,
            "test_loss": val_loss,
            "t_duration": t_duration,
        }

        writer.print_info(info)
        writer.save_checkpoint(model, optimizer, scheduler, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses


# ============================================================
# Euclidean reconstruction error
# ============================================================

def eval_error(model, loader, device, meshdata, out_dir):
    model.eval()

    mean = meshdata.mean.to(device)
    std = meshdata.std.to(device)
    V = int(meshdata.num_nodes)

    errors = []

    with torch.no_grad():
        for data in loader:
            x = reshape_pyg_batch(data, device)

            B, V_here, C = x.shape

            if V_here != V:
                raise RuntimeError(
                    f"eval_error V mismatch: expected {V}, got {V_here}"
                )

            if C != 3:
                raise RuntimeError(f"Expected 3 channels, got {C}")

            pred, mu, log_var, *_ = model(x)

            if pred.shape != x.shape:
                raise RuntimeError(
                    f"eval_error: pred shape {tuple(pred.shape)} != x shape {tuple(x.shape)}"
                )

            x_denorm = x * std.view(1, V, 3) + mean.view(1, V, 3)
            p_denorm = pred * std.view(1, V, 3) + mean.view(1, V, 3)

            # Your original code used this scale factor.
            x_denorm = x_denorm * 300.0
            p_denorm = p_denorm * 300.0

            e = torch.sqrt(((p_denorm - x_denorm) ** 2).sum(dim=2))
            errors.append(e.detach().cpu())

    if len(errors) == 0:
        return 0.0

    all_e = torch.cat(errors, dim=0)

    mean_error = all_e.mean().item()
    std_error = all_e.std().item()
    median_error = all_e.median().item()

    message = f"Euclidean Error: {mean_error:.3f}+{std_error:.3f} | median={median_error:.3f}"

    os.makedirs(out_dir, exist_ok=True)

    out_error_fp = os.path.join(out_dir, "euc_errors.txt")
    with open(out_error_fp, "a") as f:
        f.write(message + "\n")

    print("\n" + message + "\n")

    return mean_error
