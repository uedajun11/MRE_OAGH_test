#!/usr/bin/env python3
"""
Direct Inversion (DI) baseline evaluation for MRE benchmark.

Runs classical Helmholtz direct inversion on test datasets and saves
metrics in the same CSV format as the ML model evaluator, so results
can be compared side-by-side with trained models.

Usage:
    python run_direct_inversion.py \
        --test-dirs /path/to/Test1 /path/to/Test2 ... \
        --output-dir /path/to/results/
"""

import argparse
import csv
import os
import sys
import time
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader

# ---- Resolve project root so imports work from anywhere ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Data_loader import WDataset, PDataset
from losses.homogeneous import MREHelmholtzLoss


# ------------------------------------------------------------------ #
#  Metric helpers (same as ModelEvaluator)
# ------------------------------------------------------------------ #
def compute_mae(pred, gt):
    return torch.mean(torch.abs(pred.float() - gt.float())).item()

def compute_rmse(pred, gt):
    return torch.sqrt(torch.mean((pred.float() - gt.float()) ** 2)).item()

def compute_ssim(pred, gt):
    """Simplified structural similarity."""
    from torch.nn.functional import avg_pool2d
    p = pred.float().unsqueeze(0).unsqueeze(0) if pred.dim() == 2 else pred.float().unsqueeze(0)
    g = gt.float().unsqueeze(0).unsqueeze(0) if gt.dim() == 2 else gt.float().unsqueeze(0)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p = avg_pool2d(p, 11, stride=1, padding=5)
    mu_g = avg_pool2d(g, 11, stride=1, padding=5)
    sig_p = avg_pool2d(p ** 2, 11, stride=1, padding=5) - mu_p ** 2
    sig_g = avg_pool2d(g ** 2, 11, stride=1, padding=5) - mu_g ** 2
    sig_pg = avg_pool2d(p * g, 11, stride=1, padding=5) - mu_p * mu_g
    ssim_map = ((2 * mu_p * mu_g + C1) * (2 * sig_pg + C2)) / \
               ((mu_p ** 2 + mu_g ** 2 + C1) * (sig_p + sig_g + C2))
    return ssim_map.mean().item()

def get_masks(mu_map):
    """Binary masks for inclusion vs background based on mu value."""
    mu = mu_map.squeeze()
    bg_val = mu[0, 0].item()
    inc_mask = (mu != bg_val)
    bg_mask = (mu == bg_val)
    return inc_mask, bg_mask

def compute_cnr(pred_map, inc_mask, bg_mask):
    pred = pred_map.squeeze().float()
    if inc_mask.sum() == 0 or bg_mask.sum() == 0:
        return 0.0
    mu_inc = pred[inc_mask].mean()
    mu_bg = pred[bg_mask].mean()
    std_inc = pred[inc_mask].std()
    std_bg = pred[bg_mask].std()
    denom = torch.sqrt(std_inc ** 2 + std_bg ** 2 + 1e-8)
    return (torch.abs(mu_inc - mu_bg) / denom).item()


# ------------------------------------------------------------------ #
#  Direct inversion runner
# ------------------------------------------------------------------ #
def evaluate_direct_inversion(test_dirs, output_dir, fov=0.2, offsets=8,
                               k_filter=1000, rho=1000, snr_levels=None):
    """
    Run direct inversion on each test directory and save aggregate metrics.
    """
    if snr_levels is None:
        snr_levels = [30, 25, 20, 15]

    os.makedirs(output_dir, exist_ok=True)

    # Helmholtz inverter
    hom = MREHelmholtzLoss(
        density=rho,
        fov=(fov, fov),
        residual_type='raw',
        k_filter=k_filter,
        epsilon=1e-10,
        verbose=False
    )

    rows = []
    model_label = "DirectInversion"

    for test_dir in test_dirs:
        dataset_name = os.path.basename(test_dir.rstrip('/'))
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}  ({test_dir})")
        print(f"{'='*60}")

        # Detect file format (.pt or .mat)
        pt_files = [f for f in os.listdir(test_dir) if f.endswith('.pt')]
        mat_files = [f for f in os.listdir(test_dir) if f.endswith('.mat')]

        if pt_files:
            print(f"  Found {len(pt_files)} .pt files")
            dataset = PDataset(
                dir_input=test_dir,
                offsets=offsets,
                fov=fov,
                extension='.pt'
            )
        elif mat_files:
            mat_ids = sorted([os.path.splitext(f)[0] for f in mat_files])
            print(f"  Found {len(mat_ids)} .mat files")
            dataset = WDataset(
                ids=mat_ids,
                dir_input=test_dir + '/',
                offsets=offsets,
                fov=fov,
                extension='.mat'
            )
        else:
            print(f"  WARNING: No .pt or .mat files in {test_dir}, skipping.")
            continue
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        for snr_db in snr_levels:
            metrics = {
                'mae_k': [], 'rmse_k': [], 'ssim_k': [],
                'mae_mu': [], 'rmse_mu': [], 'ssim_mu': [],
                'cnr': []
            }

            t0 = time.time()
            for batch in loader:
                wave_input, mu_gt_np, k_gt, mfre, fov_val, idx = batch
                B = wave_input.shape[0]

                # Add noise for this SNR level
                if snr_db < 100:
                    noise_std = wave_input.abs().mean() * 10 ** (-snr_db / 20.0)
                    wave_noisy = wave_input + noise_std * torch.randn_like(wave_input)
                else:
                    wave_noisy = wave_input

                # directInverse → extract_fundamental_frequency_batch expects (B, C, H, W, T)
                # PDataset returns (1, H, W, T) per sample → batched (B, 1, H, W, T) — already correct
                # WDataset returns (1, T, H, W) per sample → batched (B, 1, T, H, W) — needs permute
                wave_5d = wave_noisy
                if wave_5d.shape[-1] != offsets:
                    # WDataset layout (B, 1, T, H, W) → permute to (B, 1, H, W, T)
                    wave_for_di = wave_5d.permute(0, 1, 3, 4, 2)
                else:
                    # PDataset layout (B, 1, H, W, T) — already correct
                    wave_for_di = wave_5d

                # Run direct inversion → returns (sm_kpa, k_di)
                with torch.no_grad():
                    sm_kpa, k_di = hom.directInverse(wave_for_di, mfre, apply_medfilt=True)
                    # sm_kpa shape: (B, H, W) in kPa
                    # k_di shape:   (B, 1, H, W) complex wave number

                # Convert to Pa for comparison with mu_gt (which is in Pa)
                mu_pred_pa = sm_kpa.float() * 1000.0  # kPa → Pa

                # Convert mu_gt from numpy to tensor
                mu_gt_t = torch.tensor(mu_gt_np).float()
                if mu_gt_t.dim() == 3:
                    mu_gt_t = mu_gt_t  # (B, H, W)
                elif mu_gt_t.dim() == 4:
                    mu_gt_t = mu_gt_t.squeeze(1)

                # Use k directly from directInverse (physical wave number in rad/m).
                # k_di shape: (B, 1, H, W) — squeeze channel dim to (B, H, W).
                # k_gt from PDataset is also in rad/m: k = omega * sqrt(rho/mu).
                # NOTE: Previous version incorrectly used WDataset convention
                # (k = fov * omega / sqrt(mu_kPa)) which is off by a factor of fov.
                k_pred = torch.abs(k_di.squeeze(1).float())  # abs since sqrt of complex can produce negative real parts

                k_gt_sq = k_gt.squeeze(1)  # (B, H, W)

                for b in range(B):
                    metrics['mae_k'].append(compute_mae(k_pred[b], k_gt_sq[b]))
                    metrics['rmse_k'].append(compute_rmse(k_pred[b], k_gt_sq[b]))
                    metrics['ssim_k'].append(compute_ssim(k_pred[b], k_gt_sq[b]))
                    metrics['mae_mu'].append(compute_mae(mu_pred_pa[b], mu_gt_t[b]))
                    metrics['rmse_mu'].append(compute_rmse(mu_pred_pa[b], mu_gt_t[b]))
                    metrics['ssim_mu'].append(compute_ssim(mu_pred_pa[b], mu_gt_t[b]))

                    # CNR only for inclusion datasets
                    if 'Inclusion' in dataset_name or 'inclusion' in dataset_name:
                        inc_mask, bg_mask = get_masks(mu_gt_t[b])
                        metrics['cnr'].append(compute_cnr(mu_pred_pa[b], inc_mask, bg_mask))

            elapsed = time.time() - t0

            # Aggregate
            row = {
                'model': model_label,
                'snr_db': snr_db,
                'dataset': dataset_name,
                'mae_k': np.mean(metrics['mae_k']) if metrics['mae_k'] else '',
                'rmse_k': np.mean(metrics['rmse_k']) if metrics['rmse_k'] else '',
                'ssim_k': np.mean(metrics['ssim_k']) if metrics['ssim_k'] else '',
                'mae_mu': np.mean(metrics['mae_mu']) if metrics['mae_mu'] else '',
                'rmse_mu': np.mean(metrics['rmse_mu']) if metrics['rmse_mu'] else '',
                'ssim_mu': np.mean(metrics['ssim_mu']) if metrics['ssim_mu'] else '',
                'cnr': np.mean(metrics['cnr']) if metrics['cnr'] else '',
                'time_s': f"{elapsed:.1f}"
            }
            rows.append(row)

            print(f"  SNR={snr_db:2d}dB  |  mae_k={row['mae_k']:>10.4f}  "
                  f"mae_mu={row['mae_mu']:>10.4f}  "
                  f"time={elapsed:.1f}s")

    # Save CSV
    csv_path = os.path.join(output_dir, 'direct_inversion_metrics.csv')
    fieldnames = ['model', 'snr_db', 'dataset', 'mae_k', 'rmse_k', 'ssim_k',
                  'mae_mu', 'rmse_mu', 'ssim_mu', 'cnr', 'time_s']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to: {csv_path}")
    return csv_path


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Direct Inversion baseline for MRE benchmark')
    parser.add_argument('--test-dirs', nargs='+', required=True,
                        help='Paths to test dataset directories containing .mat files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--fov', type=float, default=0.2,
                        help='Field of view in meters (default: 0.2)')
    parser.add_argument('--offsets', type=int, default=8,
                        help='Number of time offsets (default: 8)')
    parser.add_argument('--k-filter', type=float, default=1000,
                        help='Butterworth filter cutoff (default: 1000)')
    parser.add_argument('--snr-levels', nargs='+', type=int, default=[30, 25, 20, 15],
                        help='SNR levels to evaluate (default: 30 25 20 15)')

    args = parser.parse_args()
    evaluate_direct_inversion(
        test_dirs=args.test_dirs,
        output_dir=args.output_dir,
        fov=args.fov,
        offsets=args.offsets,
        k_filter=args.k_filter,
        snr_levels=args.snr_levels
    )
