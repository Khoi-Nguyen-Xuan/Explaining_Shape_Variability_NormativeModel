import numpy as np 
import torch
import torch.nn as nn
import math
import scipy.optimize
from scipy.spatial.distance import cdist

class SNNLCrossEntropy():
    STABILITY_EPS = 0.00001
    def __init__(self,
               temperature=100.,
               factor=-10.,
               optimize_temperature=True,
               cos_distance=True):
        
        self.temperature = temperature
        self.factor = factor
        self.optimize_temperature = optimize_temperature
        self.cos_distance = cos_distance
    
    @staticmethod
    def pairwise_euclid_distance(A, B):
        """Pairwise Euclidean distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise Euclidean between A and B.
        """
        batchA = A.shape[0]
        batchB = B.shape[0]

        sqr_norm_A = torch.reshape(torch.pow(A, 2).sum(axis=1), [1, batchA])
        sqr_norm_B = torch.reshape(torch.pow(B, 2).sum(axis=1), [batchB, 1])
        inner_prod = torch.matmul(B, A.T)

        tile_1 = torch.tile(sqr_norm_A, [batchB, 1])
        tile_2 = torch.tile(sqr_norm_B, [1, batchA])
        return (tile_1 + tile_2 - 2 * inner_prod)
    
    @staticmethod
    def pairwise_cos_distance(A, B):
        
        """Pairwise cosine distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise cosine between A and B.
        """
        normalized_A = torch.nn.functional.normalize(A, dim=1)
        normalized_B = torch.nn.functional.normalize(B, dim=1)
        prod = torch.matmul(normalized_A, normalized_B.transpose(-2, -1).conj())
        return 1 - prod
    
    @staticmethod
    def fits(A, B, temp, cos_distance):
        if cos_distance:
            distance_matrix = SNNLCrossEntropy.pairwise_cos_distance(A, B)
        else:
            distance_matrix = SNNLCrossEntropy.pairwise_euclid_distance(A, B)
            
        return torch.exp(-(distance_matrix / temp))
    
    @staticmethod
    def pick_probability(x, temp, cos_distance):
        """Row normalized exponentiated pairwise distance between all the elements
        of x. Conceptualized as the probability of sampling a neighbor point for
        every element of x, proportional to the distance between the points.
        :param x: a matrix
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or euclidean distance
        :returns: A tensor for the row normalized exponentiated pairwise distance
                  between all the elements of x.
        """
        f = SNNLCrossEntropy.fits(x, x, temp, cos_distance) - torch.eye(x.shape[0], device='cuda:0')
        return f / (SNNLCrossEntropy.STABILITY_EPS + f.sum(axis=1).unsqueeze(1))
    
    @staticmethod
    def same_label_mask(y, y2):
        """Masking matrix such that element i,j is 1 iff y[i] == y2[i].
        :param y: a list of labels
        :param y2: a list of labels
        :returns: A tensor for the masking matrix.
        """
        return (y == y2.unsqueeze(1)).squeeze().to(torch.float32)
    
    @staticmethod
    def masked_pick_probability(x, y, temp, cos_distance):
        """The pairwise sampling probabilities for the elements of x for neighbor
        points which share labels.
        :param x: a matrix
        :param y: a list of labels for each element of x
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or Euclidean distance
        :returns: A tensor for the pairwise sampling probabilities.
        """
        return SNNLCrossEntropy.pick_probability(x, temp, cos_distance) * \
                                    SNNLCrossEntropy.same_label_mask(y, y)
    
    @staticmethod
    def SNNL(x, y, temp=100, cos_distance=True):
        """Soft Nearest Neighbor Loss
        :param x: a matrix.
        :param y: a list of labels for each element of x.
        :param temp: Temperature.
        :cos_distance: Boolean for using cosine or Euclidean distance.
        :returns: A tensor for the Soft Nearest Neighbor Loss of the points
                  in x with labels y.
        """
        summed_masked_pick_prob = SNNLCrossEntropy.masked_pick_probability(x, y, temp, cos_distance).sum(axis=1)
        return -torch.log(SNNLCrossEntropy.STABILITY_EPS + summed_masked_pick_prob).mean()



class ClsCorrelationLoss(nn.Module):
    def __init__(self):
        super(ClsCorrelationLoss, self).__init__()

    def forward(self, z_batch, y_batch):
        # Split z_batch and y_batch into categories
        z_1 = z_batch[y_batch.flatten() == 1.0]
        z_0 = z_batch[y_batch.flatten() == 0.0]
        n_1 = len(z_1)
        n_0 = len(z_0)
        n = n_1 + n_0

        # Calculate means for the two categories
        mean_z_1 = torch.mean(z_1[:, 0])
        mean_z_0 = torch.mean(z_0[:, 0])

        # Multiplier
        mlt = math.sqrt((n_1 * n_0) / (n**2))

        # Calculate point biserial correlation
        r_pb = (mean_z_1 - mean_z_0) / torch.std(z_batch[:, 0]) * mlt

        # Calculate correlation of other dimensions with y
        other_dim_corrs = torch.zeros_like(z_batch[:, 1])
        for i in range(1, z_batch.shape[1]):
            other_dim_corrs[i-1] = (torch.mean(z_1[:, i]) - torch.mean(z_0[:, i])) / torch.std(z_batch[:, i]) * mlt

        # Loss components
        ncc_loss = 1 - torch.abs(r_pb)  # Minimize correlation
        other_dims_loss = torch.mean(torch.abs(other_dim_corrs))  # Minimize other dimension correlations

        # Combine losses with weights
        total_loss = ncc_loss + other_dims_loss

        return total_loss
    
#Pearson correlation

class RegCorrelationLoss(nn.Module):
    def __init__(self):
        super(RegCorrelationLoss, self).__init__()

    def forward(self, z_batch, y_batch):
        # Calculate the means of x and y
        y_batch = y_batch.squeeze()
        mean_z = torch.mean(z_batch[:, 1])
        mean_y = torch.mean(y_batch)
        # Calculate the differences from the means
        diff_z = z_batch[:, 1] - mean_z
        diff_y = y_batch - mean_y
        
        # Calculate the sum of squared differences
        sum_squared_diff_z = torch.sum(diff_z ** 2)
        sum_squared_diff_y = torch.sum(diff_y ** 2)
        
        # Calculate the cross-product of differences
        cross_product = torch.sum(diff_z * diff_y)
        
        # Calculate the denominator (product of standard deviations)
        denominator = torch.sqrt(sum_squared_diff_z * sum_squared_diff_y)
        
        # Calculate the Pearson correlation coefficient
        r_p = cross_product / denominator

        # Calculate correlation of other dimensions with y
        other_dim_corrs = torch.zeros_like(z_batch[:, 0])
        #first element
        mean_z, mean_y = torch.mean(z_batch[:, 0]), torch.mean(y_batch)
        diff_z, diff_y = z_batch[:, 0] - mean_z, y_batch - mean_y
        sum_squared_diff_z, sum_squared_diff_y = torch.sum(diff_z ** 2), torch.sum(diff_y ** 2)
        other_dim_corrs[0] = torch.sum(diff_z * diff_y) / torch.sqrt(sum_squared_diff_z * sum_squared_diff_y)
        #remaining element
        for i in range(2, z_batch.shape[1]):
            mean_z, mean_y = torch.mean(z_batch[:, i]), torch.mean(y_batch)
            diff_z, diff_y = z_batch[:, i] - mean_z, y_batch - mean_y
            sum_squared_diff_z, sum_squared_diff_y = torch.sum(diff_z ** 2), torch.sum(diff_y ** 2)
            other_dim_corrs[i-1] = torch.sum(diff_z * diff_y) / torch.sqrt(sum_squared_diff_z * sum_squared_diff_y)

        # Loss components
        ncc_loss = 1 - torch.abs(r_p)  # Minimize correlation
        other_dims_loss = torch.mean(torch.abs(other_dim_corrs))  # Minimize other dimension correlations

        # Combine losses with weights
        total_loss = ncc_loss + other_dims_loss

        return total_loss
    
# SNNL loss modified fast
class SNNLoss(nn.Module):
    def __init__(self, T):
        super(SNNLoss, self).__init__()
        self.T = T
        self.STABILITY_EPS = 0.00001

    def forward(self, x, y):
        b = x.size(0)  # Batch size
        y = y.squeeze()

        x_expanded = x[:,0].unsqueeze(1)  # Expand dimensions for broadcasting
        y_expanded = y.unsqueeze(0)

        same_class_mask = y_expanded == y_expanded.t()

        squared_distances = (x_expanded - x_expanded.t()) ** 2
        exp_distances = torch.exp(-(squared_distances / self.T))
        exp_distances = exp_distances * (1 - torch.eye(b, device='cuda:0'))
        #print(exp_distances)

        numerator = exp_distances * same_class_mask
        denominator = exp_distances
        # remaining elements
        exp_distances_all = torch.zeros_like(exp_distances, device='cuda:0')
        for i in range(1, x.shape[1]):
            x_expanded = x[:,i].unsqueeze(1)
            squared_distances = (x_expanded - x_expanded.t()) ** 2
            exp_distances = torch.exp(-(squared_distances / self.T))
            exp_distances = exp_distances * (1 - torch.eye(b, device='cuda:0'))
            exp_distances = exp_distances * same_class_mask
            exp_distances_all = exp_distances_all + exp_distances

        
        denominator1 = exp_distances_all/float(x.shape[1]-1)
        #print(denominator)

        lsn_loss = -torch.log(self.STABILITY_EPS + (numerator.sum(dim=1) / (self.STABILITY_EPS + (0.5*denominator.sum(dim=1)) + (0.5*denominator1.sum(dim=1))))).mean()

        return lsn_loss
    
# SNNL loss reg modified fast
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class SNNRegLoss(nn.Module):
    """
    Matches the original repo logic:
      - x is [B, D] full latent
      - supervised_dim is used for numerator/denominator (repo used x[:,1])
      - denominator1 uses all OTHER dims (repo used dims 0,2..D-1)
      - dot-product similarity with exp(sim/T)
      - class_counts normalization
      - lamda1, lamda2 structure
    """
    def __init__(self, T: float, lamda1: float, lamda2: float, threshold: float, stability_eps: float = 1e-5, supervised_dim: int = 1):
        super().__init__()
        self.T = float(T)
        self.lamda1 = float(lamda1)
        self.lamda2 = float(lamda2)
        self.threshold = float(threshold)
        self.eps = float(stability_eps)
        self.supervised_dim = int(supervised_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]
        y: [B] (age)
        """
        if x.dim() != 2:
            raise RuntimeError(f"SNNRegLossRepo expects x [B,D], got {tuple(x.shape)}")

        B, D = x.shape
        if B < 2:
            return x.new_tensor(0.0)
        if D < 2:
            # original logic assumes there exist "other dims" besides supervised_dim
            raise RuntimeError(f"SNNRegLossRepo expects D>=2, got D={D}")

        y = y.view(-1)
        if y.numel() != B:
            raise RuntimeError(f"SNNRegLossRepo expects y [B], got {tuple(y.shape)} for B={B}")

        if not (0 <= self.supervised_dim < D):
            raise RuntimeError(f"supervised_dim={self.supervised_dim} out of range for D={D}")

        device, dtype = x.device, x.dtype
        eye = torch.eye(B, device=device, dtype=dtype)

        # same_class_mask: |age_i - age_j| <= threshold, remove diagonal
        abs_diff = (y.unsqueeze(0) - y.unsqueeze(1)).abs()
        same = (abs_diff <= self.threshold).to(dtype)
        same = same * (1.0 - eye)

        # ----- numerator/denominator computed from supervised dim ONLY -----
        x_sup = x[:, self.supervised_dim:self.supervised_dim+1]          # [B,1]
        sim_sup = (x_sup @ x_sup.t()) / max(self.T, 1e-8)                # [B,B]
        exp_sup = torch.exp(sim_sup) * (1.0 - eye)                       # zero diagonal

        numerator = exp_sup * same                                       # [B,B]
        denominator = exp_sup                                            # [B,B]

        # ----- denominator1: average over OTHER dims, masked by same -----
        exp_all = torch.zeros_like(exp_sup)                              # [B,B]

        # repo included dim 0, then dims 2..end (excluding supervised dim 1)
        for d in range(D):
            if d == self.supervised_dim:
                continue
            x_d = x[:, d:d+1]
            sim_d = (x_d @ x_d.t()) / max(self.T, 1e-8)
            exp_d = torch.exp(sim_d) * (1.0 - eye)
            exp_all = exp_all + (exp_d * same)

        denom1 = exp_all / float(D - 1)

        # class_counts (positives per anchor)
        class_counts = same.sum(dim=1).clamp_min(1.0)

        frac = numerator.sum(dim=1) / (self.eps + self.lamda1 * denominator.sum(dim=1) + self.lamda2 * denom1.sum(dim=1))
        loss = ((-torch.log(self.eps + frac)) / class_counts).mean()

        return loss

def _corrcoef_1d(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    a, b: [B]
    returns Pearson correlation (scalar)
    """
    a = a - a.mean()
    b = b - b.mean()
    a_std = a.std(unbiased=False) + eps
    b_std = b.std(unbiased=False) + eps
    return (a * b).mean() / (a_std * b_std)

def age_leakage_loss(z: torch.Tensor, age: torch.Tensor, age_dim: int = 1) -> torch.Tensor:
    """
    Penalize correlation between age and ALL dims except age_dim.
    z:   [B, D]
    age: [B]
    Returns mean(r^2) across other dims.
    """
    B, D = z.shape
    if D <= 1:
        return z.new_tensor(0.0)

    age = age.view(-1)
    assert 0 <= age_dim < D

    # gather other dims
    if age_dim == 0:
        z_other = z[:, 1:]
    elif age_dim == D - 1:
        z_other = z[:, :-1]
    else:
        z_other = torch.cat([z[:, :age_dim], z[:, age_dim+1:]], dim=1)  # [B, D-1]

    # compute mean squared correlation
    losses = []
    for j in range(z_other.shape[1]):
        r = _corrcoef_1d(z_other[:, j], age)
        losses.append(r * r)

    return torch.stack(losses).mean() if losses else z.new_tensor(0.0)


# Wasserstein loss proposed by nilanjan
class WassersteinLoss(nn.Module):
    def __init__(self, delta):
        super(WassersteinLoss, self).__init__()
        self.delta = delta
        self.h_loss = torch.nn.HuberLoss(reduction='mean', delta=delta)

    def linear_assignment(self, x, u):
        dist_matrix = cdist(x, u)
        _, col_ind = scipy.optimize.linear_sum_assignment(dist_matrix)
        return col_ind

    def forward(self, x):
        bsize = x.shape[0]
        dim = x.shape[1]

        u = x[torch.randperm(bsize), 0:1]
        for i in range(dim - 1):
            u = torch.cat((u, x[torch.randperm(bsize), i + 1:i + 2]), dim=1)

        with torch.no_grad():
            ind = self.linear_assignment(x.cpu().detach().numpy(), u.cpu().detach().numpy())

        loss = self.h_loss(x, u[ind])

        return loss
    
# Attribute VAE loss
class AttributeLoss(nn.Module):
    def __init__(self, factor=1.0):
        super(AttributeLoss, self).__init__()
        self.factor = factor
        self.loss_fn = nn.L1Loss()

    def forward(self, latent_code, attribute):
        # compute latent distance matrix
        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

        # compute attribute distance matrix
        attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
        attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

        # compute regularization loss
        lc_tanh = torch.tanh(lc_dist_mat * self.factor)
        attribute_sign = torch.sign(attribute_dist_mat)
        attribute_loss = self.loss_fn(lc_tanh, attribute_sign.float())

        return attribute_loss


'''
# SNNL loss modified slow
class SNNLoss(nn.Module):
    def __init__(self, T):
        super(SNNLoss, self).__init__()
        self.T = T

    def forward(self, x, y):
        b = x.size(0)  # Batch size
        lsn_loss = 0.0
        
        for i in range(b):
            xi = x[i]
            yi = y[i]
            
            numerator = 0.0
            denominator = 0.0
            
            for j in range(b):
                if j != i and y[j] == yi:
                    numerator += torch.exp(-((xi - x[j])**2).sum() / self.T)
                    
                denominator += torch.exp(-((xi - x[j])**2).sum() / self.T)
            
            lsn_loss += -torch.log(numerator / denominator)
        
        return lsn_loss / b
'''
