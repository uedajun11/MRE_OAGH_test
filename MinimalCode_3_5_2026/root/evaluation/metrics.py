# metrics.py
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim_np

# -------------------------------
# Basic regression metrics
# -------------------------------

def compute_mae(pred, gt):
    """
    Mean Absolute Error
    """
    return torch.mean(torch.abs(pred - gt)).item()


def compute_rmse(pred, gt):
    """
    Root Mean Squared Error
    """
    return torch.sqrt(torch.mean((pred - gt) ** 2)).item()


# +
def compute_ssim(pred, gt, data_range=None, crop_phantom=False):
    pred_np = pred.squeeze().cpu().numpy()
    gt_np   = gt.squeeze().cpu().numpy()

    def safe_data_range(g):
        dr = g.max() - g.min()
        return dr if dr > 0 else 1.0

    def crop(arr):
        if arr.ndim == 2:
            return arr[51:205, 5:251]
        return arr

    if crop_phantom:
        pred_np = crop(pred_np)
        gt_np   = crop(gt_np)

    if pred_np.ndim == 2:
        return ssim_np(gt_np, pred_np,
                       data_range=data_range or safe_data_range(gt_np))
    else:
        ssim_vals = []
        for p, g in zip(pred_np, gt_np):
            ssim_vals.append(ssim_np(g, p,
                             data_range=data_range or safe_data_range(g)))
        return float(np.mean(ssim_vals))

# def compute_ssim(pred, gt, data_range=None):
#     """
#     Structural Similarity Index (SSIM)
#     Uses skimage implementation via numpy
#     Assumes pred and gt are torch tensors of shape [B,1,H,W] or [B,H,W]
#     """
#     pred_np = pred.squeeze().cpu().numpy()
#     gt_np = gt.squeeze().cpu().numpy()

#     # Handle single-channel 2D images
#     if pred_np.ndim == 2:
#         return ssim_np(gt_np, pred_np, data_range=data_range or (gt_np.max() - gt_np.min()))
#     else:
#         # Batch of images
#         ssim_vals = []
#         for p, g in zip(pred_np, gt_np):
#             ssim_vals.append(ssim_np(g, p, data_range=data_range or (g.max() - g.min())))
#         return float(np.mean(ssim_vals))


# -

# -------------------------------
# Contrast-to-Noise Ratio (CNR)
# -------------------------------

def compute_cnr(pred, inc_mask, bg_mask, epsilon=1e-6):
    """
    Compute contrast-to-noise ratio:
        CNR = |mean_inclusion - mean_background| / sqrt(var_inclusion + var_background)
    pred: [H,W] torch tensor
    inc_mask, bg_mask: boolean masks
    """
    pred = pred.float()

    mu_inc = pred[inc_mask].mean()
    mu_bg  = pred[bg_mask].mean()

    var_inc = pred[inc_mask].var(unbiased=False)
    var_bg  = pred[bg_mask].var(unbiased=False)

    cnr_val = torch.abs(mu_inc - mu_bg) / torch.sqrt(var_inc + var_bg + epsilon)
    return cnr_val.item()


# -------------------------------
# Helper: get inclusion/background masks
# -------------------------------

def get_masks(gt, threshold=None):
    """
    Create binary masks for inclusion and background.
    Example: simple thresholding on gt for demonstration.
    You can replace this with your real inclusion segmentation logic.
    """
    if threshold is None:
        threshold = gt.mean()  # simple heuristic

    inc_mask = gt > threshold
    bg_mask  = gt <= threshold

    return inc_mask.squeeze(), bg_mask.squeeze()
