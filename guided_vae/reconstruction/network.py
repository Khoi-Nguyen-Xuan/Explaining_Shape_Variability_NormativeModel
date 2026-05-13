import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from conv import SpiralConv


DEBUG_POOL = False   # set False later


def Pool(x, trans):
    # x can be (V, C) or (B, V, C)
    if x.dim() == 2:
        # (V, C) -> (1, V, C)
        x = x.unsqueeze(0)
    B, V_in, C = x.shape

    # sparse stuff
    row, col = trans._indices()
    val = trans._values()
    device = x.device
    row = row.to(device)
    col = col.to(device)
    val = val.to(device)

    max_col = int(col.max().item())

    if DEBUG_POOL:
        print("[POOL DEBUG]")
        print(f"  x.shape      = {x.shape}")           # (B, V_in, C)
        print(f"  trans.size() = {trans.size()}")      # (V_out, V_in)
        print(f"  edges (nnz)  = {col.numel()}")
        print(f"  max col      = {max_col}")
        print(f"  V_in         = {V_in}")

    # safety check
    if max_col >= V_in:
        print("[POOL ERROR] transform wants vertex", max_col, "but x has", V_in)
        raise RuntimeError(f"Pool got out-of-range index {max_col} for V_in={V_in}")

    # gather / scatter
    col_b = col.unsqueeze(0).expand(B, -1)  # (B, E)
    gathered = x.gather(1, col_b.unsqueeze(-1).expand(-1, -1, C))  # (B, E, C)
    weighted = gathered * val.unsqueeze(0).unsqueeze(-1)           # (B, E, C)
    row_b = row.unsqueeze(0).expand(B, -1)
    out = scatter_add(weighted, row_b, dim=1, dim_size=trans.size(0))
    return out


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out


class AE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform, up_transform, training=True):
        super(AE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)
        self.training = training

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.en_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], 2 * latent_channels)
        )
        # self.en_layers.append(nn.Linear(8*latent_channels, 2*latent_channels))

        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels, self.num_vert * out_channels[-1])
        )
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx - 1],
                                  out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx],
                                  out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
        self.de_layers.append(
            SpiralConv(out_channels[0], in_channels, self.spiral_indices[0])
        )

        # Excitation
        self.cls_sq = nn.Sequential(
            nn.Linear(1, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        # Excitation 2
        self.reg_sq_2 = nn.Sequential(
            nn.Linear(1, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        # safer init: only use xavier for tensors with dim >= 2
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                if param.dim() < 2:
                    # e.g., BatchNorm weights or other 1D params
                    nn.init.ones_(param)
                else:
                    nn.init.xavier_uniform_(param)

    def encoder(self, x):
        # ORIGINAL behavior: only add batch dim if [V, C]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)

        mu = x[:, :self.latent_channels]
        log_var = x[:, self.latent_channels:]
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mu, log_var

    def decoder(self, x):
        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i])
            else:
                x = layer(x)
        return x

    def reparameterize(self, mu, log_var):
        if self.training:
            log_var = torch.clamp(log_var, min=-10.0, max=10.0)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def cls(self, z):  # first excitation
        z = torch.split(z, 1, 1)[0]
        was_training = self.cls_sq.training
        if was_training:
            self.cls_sq.eval()
        out = self.cls_sq(z)
        if was_training:
            self.cls_sq.train()
        return out

    def reg_2(self, z):  # second excitation
        z = torch.split(z, 1, 1)[1]
        was_training = self.reg_sq_2.training
        if was_training:
            self.reg_sq_2.eval()
        out = self.reg_sq_2(z)
        if was_training:
            self.reg_sq_2.train()
        return out

    def forward(self, x, *indices):
        # ORIGINAL behavior: only add batch dim if [V, C]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var, self.cls(z), self.reg_2(z)


class Classifier(nn.Module):
    def __init__(self, n_vae_dis):
        super(Classifier, self).__init__()

        self.cls_sq_n = nn.Sequential(
            nn.Linear(n_vae_dis - 1, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.cls_sq_n(x)


class Regressor(nn.Module):
    def __init__(self, n_vae_dis):
        super(Regressor, self).__init__()

        self.reg_sq = nn.Sequential(
            nn.Linear(n_vae_dis - 1, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.reg_sq(x)
