#!/usr/bin/env python3
"""
Direct Inversion (DI) baseline evaluation for MRE benchmark.

Runs classical Helmholtz direct inversion on test datasets and saves
metrics in the same CSV format as the ML model evaluator, so results
can be compared side-by-side with trained models.

Supports parameter sweeps over k_filter (Butterworth cutoff) and
medfilt_kernel (post-inversion median filter size) to find the optimal
DI configuration at each SNR level.

Usage:
    python run_direct_inversion.py \
        --test-dirs /path/to/Test1 /path/to/Test2 ... \
        --output-dir /path/to/results/

    # Parameter sweep:
    python run_direct_inversion.py \
        --test-dirs /path/to/Test1 ... \
        --output-dir /path/to/results/ \
        --k-filter-sweep 300 500 700 1000 \
        --medfilt-sweep 3 5 7
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
from scipy.signal import medfilt2d

# ---- Resolve project root so imports work from anywhere ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Data_loader import WDataset, PDataset
from losses.homogeneous import MREHelmholtzLoss

try:
    from skimage.metrics import structural_similarity as ssim_np
except ImportError:
    ssim_np = None
    print("WARNING: skimage not available — SSIM will be reported as NaN")


# ------------------------------------------------------------------ #
#  Metric helpers (same as ModelEvaluator)
# ------------------------------------------------------------------ #

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

    if ssim_np is None:
        return float('nan')

    if pred_np.ndim == 2:
        return ssim_np(gt_np, pred_np,
                       data_range=data_range or safe_data_range(gt_np))
    else:
        ssim_vals = []
        for p, g in zip(pred_np, gt_np):
            ssim_vals.append(ssim_np(g, p,
                             data_range=data_range or safe_data_range(g)))
        return float(np.mean(ssim_vals))

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

# ------------------------------------------------------------------ #
#  Data loading helper (shared across sweep configs)
# ------------------------------------------------------------------ #
def _load_datasets(test_dirs, offsets, fov):
    """
    Pre-load all datasets once so they can be reused across parameter
    sweep iterations without re-reading from disk.
    Returns list of (dataset_name, loader) tuples.
    """
    loaded = []
    for test_dir in test_dirs:
        dataset_name = os.path.basename(test_dir.rstrip('/'))
        pt_files = [f for f in os.listdir(test_dir) if f.endswith('.pt')]
        mat_files = [f for f in os.listdir(test_dir) if f.endswith('.mat')]

        if pt_files:
            print(f"  [{dataset_name}] Found {len(pt_files)} .pt files")
            dataset = PDataset(
                dir_input=test_dir, offsets=offsets,
                fov=fov, extension='.pt'
            )
        elif mat_files:
            mat_ids = sorted([os.path.splitext(f)[0] for f in mat_files])
            print(f"  [{dataset_name}] Found {len(mat_ids)} .mat files")
            dataset = WDataset(
                ids=mat_ids, dir_input=test_dir + '/',
                offsets=offsets, fov=fov, extension='.mat'
            )
        else:
            print(f"  WARNING: No .pt or .mat files in {test_dir}, skipping.")
            continue
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        loaded.append((dataset_name, loader))
    return loaded


# ------------------------------------------------------------------ #
#  Direct inversion runner (single configuration)
# ------------------------------------------------------------------ #
def evaluate_direct_inversion(test_dirs, output_dir, fov=0.2, offsets=8,
                               k_filter=1000, medfilt_kernel=3, rho=1000,
                               snr_levels=None, _preloaded=None):
    """
    Run direct inversion on each test directory and save aggregate metrics.

    Args:
        k_filter: Butterworth lowpass cutoff (cycles/m). None = no filtering.
        medfilt_kernel: median filter kernel size applied to mu map (odd int).
                        0 or None = skip median filter.
        _preloaded: pre-loaded (dataset_name, loader) list to avoid re-reading
                    data from disk on every sweep iteration.
    """
    if snr_levels is None:
        snr_levels = [30, 25, 20, 15]

    os.makedirs(output_dir, exist_ok=True)

    # Helmholtz inverter — apply_medfilt=False so we control it externally
    hom = MREHelmholtzLoss(
        density=rho,
        fov=(fov, fov),
        residual_type='raw',
        k_filter=k_filter,
        epsilon=1e-10,
        verbose=False
    )

    rows = []
    kf_str = f"{int(k_filter)}" if k_filter else "None"
    mf_str = f"{medfilt_kernel}" if medfilt_kernel else "0"
    model_label = f"DI_kf{kf_str}_mf{mf_str}"

    # Load data (or reuse preloaded)
    if _preloaded is not None:
        datasets_loaded = _preloaded
    else:
        datasets_loaded = _load_datasets(test_dirs, offsets, fov)

    for dataset_name, loader in datasets_loaded:
        print(f"\n{'='*60}")
        print(f"Config: k_filter={kf_str}, medfilt={mf_str}  |  Dataset: {dataset_name}")
        print(f"{'='*60}")

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

                # Shape handling:
                # PDataset: (B, 1, H, W, T) — last dim is T, already correct
                # WDataset: (B, 1, T, H, W) — needs permute to (B, 1, H, W, T)
                wave_5d = wave_noisy
                if wave_5d.shape[-1] != offsets:
                    wave_for_di = wave_5d.permute(0, 1, 3, 4, 2)
                else:
                    wave_for_di = wave_5d

                # Run direct inversion WITHOUT internal median filter
                with torch.no_grad():
                    sm_pa, k_di = hom.directInverse(wave_for_di, mfre, apply_medfilt=False)
                    # sm_pa: (B, 1, H, W) in Pa when apply_medfilt=False
                    # k_di:  (B, 1, H, W) wave number

                # Squeeze channel dim: (B, 1, H, W) → (B, H, W)
                sm_pa = sm_pa.squeeze(1)

                # Apply external median filter with configurable kernel size
                if medfilt_kernel and medfilt_kernel > 1:
                    sm_np = sm_pa.cpu().numpy()
                    for b in range(B):
                        sm_np[b] = medfilt2d(sm_np[b], kernel_size=medfilt_kernel)
                    sm_pa = torch.tensor(sm_np, device=sm_pa.device)

                mu_pred_pa = sm_pa.float()

                # Ground truth mu
                mu_gt_t = torch.tensor(mu_gt_np).float()
                if mu_gt_t.dim() == 4:
                    mu_gt_t = mu_gt_t.squeeze(1)

                # k prediction (physical rad/m, from directInverse)
                k_pred = torch.abs(k_di.squeeze(1).float())
                k_gt_sq = k_gt.squeeze(1)

                for b in range(B):
                    metrics['mae_k'].append(compute_mae(k_pred[b], k_gt_sq[b]))
                    metrics['rmse_k'].append(compute_rmse(k_pred[b], k_gt_sq[b]))
                    metrics['ssim_k'].append(compute_ssim(k_pred[b], k_gt_sq[b]))
                    metrics['mae_mu'].append(compute_mae(mu_pred_pa[b], mu_gt_t[b]))
                    metrics['rmse_mu'].append(compute_rmse(mu_pred_pa[b], mu_gt_t[b]))
                    metrics['ssim_mu'].append(compute_ssim(mu_pred_pa[b], mu_gt_t[b]))

                    if 'Inclusion' in dataset_name or 'inclusion' in dataset_name:
                        inc_mask, bg_mask = get_masks(mu_gt_t[b])
                        metrics['cnr'].append(compute_cnr(mu_pred_pa[b], inc_mask, bg_mask))

            elapsed = time.time() - t0

            row = {
                'model': model_label,
                'k_filter': kf_str,
                'medfilt_kernel': mf_str,
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

    return rows


# ------------------------------------------------------------------ #
#  Parameter sweep runner
# ------------------------------------------------------------------ #
def run_sweep(test_dirs, output_dir, fov=0.2, offsets=8, rho=1000,
              snr_levels=None, k_filter_values=None, medfilt_values=None):
    """
    Sweep over k_filter × medfilt_kernel combinations and save all
    results into a single CSV for easy comparison.
    """
    if k_filter_values is None:
        k_filter_values = [1000]
    if medfilt_values is None:
        medfilt_values = [3]

    os.makedirs(output_dir, exist_ok=True)

    # Pre-load data ONCE
    print("Loading datasets...")
    preloaded = _load_datasets(test_dirs, offsets, fov)
    print(f"Loaded {len(preloaded)} datasets.\n")

    all_rows = []
    n_configs = len(k_filter_values) * len(medfilt_values)
    config_idx = 0

    for kf in k_filter_values:
        for mf in medfilt_values:
            config_idx += 1
            print(f"\n{'#'*60}")
            print(f"  CONFIG {config_idx}/{n_configs}: k_filter={kf}, medfilt={mf}")
            print(f"{'#'*60}")

            rows = evaluate_direct_inversion(
                test_dirs=test_dirs,
                output_dir=output_dir,
                fov=fov, offsets=offsets,
                k_filter=kf, medfilt_kernel=mf,
                rho=rho, snr_levels=snr_levels,
                _preloaded=preloaded
            )
            all_rows.extend(rows)

    # Save combined CSV
    csv_path = os.path.join(output_dir, 'direct_inversion_sweep.csv')
    fieldnames = ['model', 'k_filter', 'medfilt_kernel', 'snr_db', 'dataset',
                  'mae_k', 'rmse_k', 'ssim_k', 'mae_mu', 'rmse_mu', 'ssim_mu',
                  'cnr', 'time_s']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n{'='*60}")
    print(f"Sweep complete! {len(all_rows)} rows saved to: {csv_path}")
    print(f"{'='*60}")

    # ---- Print quick summary: best config per dataset+SNR ----
    print(f"\n{'='*60}")
    print("BEST CONFIG PER DATASET + SNR (by mae_mu)")
    print(f"{'='*60}")
    # Group by dataset+snr, find min mae_mu
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_rows:
        key = (r['dataset'], r['snr_db'])
        if r['mae_mu'] != '':
            groups[key].append(r)
    for (ds, snr), rows_g in sorted(groups.items()):
        best = min(rows_g, key=lambda x: float(x['mae_mu']))
        print(f"  {ds:<22} SNR={snr:2d}  →  {best['model']:<25}  "
              f"mae_mu={float(best['mae_mu']):>8.1f}  "
              f"ssim_mu={float(best['ssim_mu']):>6.4f}")

    return csv_path


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Direct Inversion baseline for MRE benchmark (with parameter sweep)')
    parser.add_argument('--test-dirs', nargs='+', required=True,
                        help='Paths to test dataset directories')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--fov', type=float, default=0.2,
                        help='Field of view in meters (default: 0.2)')
    parser.add_argument('--offsets', type=int, default=8,
                        help='Number of time offsets (default: 8)')
    parser.add_argument('--snr-levels', nargs='+', type=int,
                        default=[30, 25, 20, 15],
                        help='SNR levels to evaluate (default: 30 25 20 15)')

    # Single-config mode (backward compatible)
    parser.add_argument('--k-filter', type=float, default=None,
                        help='Single Butterworth cutoff (use --k-filter-sweep for sweep)')
    parser.add_argument('--medfilt-kernel', type=int, default=None,
                        help='Single median filter kernel (use --medfilt-sweep for sweep)')

    # Sweep mode
    parser.add_argument('--k-filter-sweep', nargs='+', type=float, default=None,
                        help='Butterworth cutoffs to sweep (e.g. 300 500 700 1000)')
    parser.add_argument('--medfilt-sweep', nargs='+', type=int, default=None,
                        help='Median filter kernels to sweep (e.g. 3 5 7)')

    args = parser.parse_args()

    # Determine mode
    sweep_mode = (args.k_filter_sweep is not None) or (args.medfilt_sweep is not None)

    if sweep_mode:
        kf_vals = args.k_filter_sweep or [1000]
        mf_vals = args.medfilt_sweep or [3]
        run_sweep(
            test_dirs=args.test_dirs,
            output_dir=args.output_dir,
            fov=args.fov, offsets=args.offsets,
            snr_levels=args.snr_levels,
            k_filter_values=kf_vals,
            medfilt_values=mf_vals
        )
    else:
        # Single config (backward compatible)
        kf = args.k_filter if args.k_filter is not None else 1000
        mf = args.medfilt_kernel if args.medfilt_kernel is not None else 3
        rows = evaluate_direct_inversion(
            test_dirs=args.test_dirs,
            output_dir=args.output_dir,
            fov=args.fov, offsets=args.offsets,
            k_filter=kf, medfilt_kernel=mf,
            snr_levels=args.snr_levels
        )
        # Save CSV
        csv_path = os.path.join(args.output_dir, 'direct_inversion_metrics.csv')
        fieldnames = ['model', 'k_filter', 'medfilt_kernel', 'snr_db', 'dataset',
                      'mae_k', 'rmse_k', 'ssim_k', 'mae_mu', 'rmse_mu', 'ssim_mu',
                      'cnr', 'time_s']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to: {csv_path}")
