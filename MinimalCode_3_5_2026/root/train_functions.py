# +
import torch
import numpy as np
import os
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader,random_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import gc
from optparse import OptionParser
from tqdm import tqdm
import time
import csv
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



from Data_loader import *
from architectures.UpdatedNetwork import Net
# Note: metrics_util and other architecture imports (FNO3dTo2d, LTAE_Net, etc.)
# are not present in this minimal code package. Add them back when needed.
import losses.homogeneous as hom
from losses.combinedLoss import CombinedLoss
from losses.ratio_loss import RatioLoss, CombinedRatioLoss
from losses.residual_losses import ResidualLoss, CombinedResidualLoss
from losses.MSELoss import MSELoss
from losses.oagh_harmonizer import OAGHHarmonizer, extract_flat_grad, set_flat_grad


# -

def train_net(net, device, train_loader, optimizer, grad_scaler, loss_f, batch_size, use_physics=False):
    net.train()
    
    # Accumulate each component separately
    total_loss_accum = 0.0
    data_loss_accum = 0.0
    physics_loss_accum = 0.0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training batches")):
        inputs, mu_gt, k_gt, mfre, fov, index = batch
        inputs, mu_gt, k_gt, mfre = inputs.to(device), mu_gt.to(device), k_gt.to(device), mfre.to(device, dtype=inputs.dtype)
        if isinstance(fov, (int, float)):
            fov = torch.tensor([fov] * inputs.shape[0], device=device, dtype=inputs.dtype)
        else:
            fov = fov.to(device, dtype=inputs.dtype)
            if fov.dim() == 0:  # scalar tensor
                fov = fov.repeat(inputs.shape[0])
        inputs = inputs.permute(4, 0, 1, 2, 3)
        inputs = inputs.permute(1,2,0,3,4) # If not jiaying architecture, then we must have B, 1, 8, 256, 256
        
        #print(f"Input shape: {inputs.shape}, dtype: {inputs.dtype}, device: {inputs.device}")

        
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # Mixed precision training
            k_pred = net(inputs)
            #print(f"Outputs shape: {outputs.shape}, GT shape: {gt.shape}")
            mu_pred = wave_number_to_shear_stiffness(k_pred, mfre, fov)

            if use_physics:
                if not torch.isfinite(k_pred).all():
                    print("❌ Non-finite network outputs")
                    print("outputs stats:", k_pred.min(), k_pred.max())
                    raise RuntimeError("Bad network output")

                # Check stiffness conversion
                if not torch.isfinite(mu_pred).all():
                    print("❌ Non-finite predicted_stiffness")
                    print("predicted_stiffness stats:",
                          mu_pred.min(),
                          mu_pred.max())
                    raise RuntimeError("Bad stiffness conversion")

                # Check ground truth
                if not torch.isfinite(mu_gt).all():
                    print("❌ Non-finite ground truth")
                    print("gt stats:", mu_gt.min(), mu_gt.max())
                    raise RuntimeError("Bad Mu GT")

                # For physics-informed loss, pass outputs, ground truth, and mfre
                inputs_physics = inputs.permute(0,1,3,4,2) # Needed fro CombinedRatio, CombinedResidual Losses
                total_loss, data_loss, physics_loss_val = loss_f(k_pred, k_gt, mu_pred, mu_gt, inputs_physics, mfre,fov)
            else:
                # For regular MSE / other losses
                total_loss, data_loss, physics_loss_val = loss_f(k_pred, k_gt, mu_pred, mu_gt)
        
        if not torch.isfinite(total_loss):
            print("❌ Non-finite loss detected!")
            print(f"  total_loss   = {total_loss}")
            print(f"  data_loss    = {data_loss}")
            print(f"  physics_loss = {physics_loss_val}")
            print(f"  k_pred range: [{k_pred.min():.3e}, {k_pred.max():.3e}]")
            print(f"  mu_pred range: [{mu_pred.min():.3e}, {mu_pred.max():.3e}]")
            raise RuntimeError("Non-finite loss, stopping training")

        # 🔎 PRINT DEBUG (first batch + periodic)
        if batch_idx == 0 or batch_idx % 10 == 0:
            print(
                f"[Train] Batch {batch_idx}: "
                f"total={total_loss.item():.4e}, "
                f"data={data_loss.item():.4e}, "
                f"physics={physics_loss_val.item():.4e}"
            )
        if batch_idx == 0:
                print(f"  k_pred: mean={k_pred.mean():.3f}, std={k_pred.std():.3f}")
                print(f"  k_gt: mean={k_gt.mean():.3f}, std={k_gt.std():.3f}")
                print(f"  mu_pred: mean={mu_pred.mean():.3f} Pa, std={mu_pred.std():.3f}")
                print(f"  mu_gt: mean={mu_gt.mean():.3f} Pa, std={mu_gt.std():.3f}")
        
        grad_scaler.scale(total_loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        total_loss_accum += total_loss.item()
        data_loss_accum += data_loss.item()
        physics_loss_accum += physics_loss_val.item()
        #progress_bar.set_postfix(loss=loss.item())
        
    n_batches = len(train_loader)
    return total_loss_accum / n_batches, data_loss_accum / n_batches, physics_loss_accum / n_batches


def train_net_oagh(net, device, train_loader, optimizer, grad_scaler, loss_f,
                   batch_size, harmonizer, use_physics=True):
    """
    OAGH/OAGH-C training loop with per-task gradient harmonization.

    Instead of computing total_loss = lambda_data * L_data + lambda_physics * L_physics
    and doing a single backward pass, this function:
      1. Computes L_data and L_physics via the existing loss function
      2. Backwards each independently to get per-task gradients
      3. Harmonizes gradients using OAGH or OAGH-C
      4. Sets the combined gradient and steps the optimizer

    Note: GradScaler and autocast are NOT used in this path. OAGH requires
    separate backward passes for each loss, and float16 losses from autocast
    produce gradients that underflow to zero (especially physics losses with
    large rho*omega^2 terms). The forward pass runs in float32 instead.

    Args:
        net: neural network
        device: torch device
        train_loader: training DataLoader
        optimizer: optimizer
        grad_scaler: GradScaler (passed for API compat; not used for backward)
        loss_f: loss function returning (total_loss, data_loss, physics_loss)
        batch_size: batch size
        harmonizer: OAGHHarmonizer instance
        use_physics: must be True for OAGH (requires two loss components)

    Returns:
        (avg_total_loss, avg_data_loss, avg_physics_loss, avg_diagnostics)
    """
    net.train()

    total_loss_accum = 0.0
    data_loss_accum = 0.0
    physics_loss_accum = 0.0
    alpha_accum = 0.0
    f_alpha_accum = 0.0
    alpha_conflict_accum = 0.0
    alpha_drift_accum = 0.0

    params = list(net.parameters())

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training (OAGH)")):
        inputs, mu_gt, k_gt, mfre, fov_batch, index = batch
        inputs, mu_gt, k_gt, mfre = (
            inputs.to(device), mu_gt.to(device),
            k_gt.to(device), mfre.to(device, dtype=inputs.dtype)
        )
        if isinstance(fov_batch, (int, float)):
            fov_batch = torch.tensor([fov_batch] * inputs.shape[0], device=device, dtype=inputs.dtype)
        else:
            fov_batch = fov_batch.to(device, dtype=inputs.dtype)
            if fov_batch.dim() == 0:
                fov_batch = fov_batch.repeat(inputs.shape[0])

        inputs = inputs.permute(4, 0, 1, 2, 3)
        inputs = inputs.permute(1, 2, 0, 3, 4)  # B, 1, 8, 256, 256

        optimizer.zero_grad()

        # ---- Forward pass (NO autocast for OAGH) ----
        # OAGH requires per-task backward passes without GradScaler.
        # autocast produces float16 losses whose gradients underflow to zero
        # during backward (especially physics losses with large rho*omega^2 terms).
        # Using float32 throughout avoids this issue.
        k_pred = net(inputs.float())
        mu_pred = wave_number_to_shear_stiffness(k_pred, mfre, fov_batch)

        if use_physics:
            if not torch.isfinite(k_pred).all():
                print("Non-finite network outputs")
                raise RuntimeError("Bad network output")
            if not torch.isfinite(mu_pred).all():
                print("Non-finite predicted stiffness")
                raise RuntimeError("Bad stiffness conversion")

            inputs_physics = inputs.permute(0, 1, 3, 4, 2).float()
            total_loss, data_loss, physics_loss = loss_f(
                k_pred, k_gt, mu_pred, mu_gt, inputs_physics, mfre, fov_batch
            )
        else:
            total_loss, data_loss, physics_loss = loss_f(
                k_pred, k_gt, mu_pred, mu_gt
            )

        # Skip batch if non-finite
        if not torch.isfinite(total_loss):
            print(f"[OAGH] Non-finite loss at batch {batch_idx}, skipping")
            optimizer.zero_grad()
            continue

        # ---- Per-task gradient extraction ----
        # Backward data_loss (retain graph for second backward)
        data_loss.backward(retain_graph=True)
        g_data = extract_flat_grad(params)
        optimizer.zero_grad()

        # Backward physics_loss
        physics_loss.backward()
        g_physics = extract_flat_grad(params)
        optimizer.zero_grad()

        # ---- Harmonize ----
        g_harmonized, diag = harmonizer([g_data, g_physics])

        # ---- Set gradient and step ----
        if torch.isfinite(g_harmonized).all():
            set_flat_grad(params, g_harmonized)
            optimizer.step()
        else:
            print(f"[OAGH] Non-finite harmonized gradient at batch {batch_idx}, skipping step")

        # ---- Accumulate metrics ----
        total_loss_accum += (data_loss.item() + physics_loss.item())
        data_loss_accum += data_loss.item()
        physics_loss_accum += physics_loss.item()
        alpha_accum += diag['alpha']
        f_alpha_accum += diag['f_alpha']
        alpha_conflict_accum += diag['alpha_conflict']
        alpha_drift_accum += diag['alpha_drift']

        # Debug printing
        if batch_idx == 0 or batch_idx % 10 == 0:
            print(
                f"[OAGH] Batch {batch_idx}: "
                f"data={data_loss.item():.4e}, "
                f"physics={physics_loss.item():.4e}, "
                f"alpha={diag['alpha']:.3f}, f(alpha)={diag['f_alpha']:.3f}, "
                f"tier={diag['tier']}"
            )
        if batch_idx == 0:
            print(f"  k_pred: mean={k_pred.mean():.3f}, std={k_pred.std():.3f}")
            print(f"  k_gt: mean={k_gt.mean():.3f}, std={k_gt.std():.3f}")
            print(f"  mu_pred: mean={mu_pred.mean():.3f} Pa, std={mu_pred.std():.3f}")
            print(f"  mu_gt: mean={mu_gt.mean():.3f} Pa, std={mu_gt.std():.3f}")
            print(f"  g_data norm: {torch.norm(g_data):.4e}")
            print(f"  g_physics norm: {torch.norm(g_physics):.4e}")
            print(f"  g_harmonized norm: {torch.norm(g_harmonized):.4e}")

    n_batches = len(train_loader)
    avg_diag = {
        'alpha': alpha_accum / max(n_batches, 1),
        'f_alpha': f_alpha_accum / max(n_batches, 1),
        'alpha_conflict': alpha_conflict_accum / max(n_batches, 1),
        'alpha_drift': alpha_drift_accum / max(n_batches, 1),
    }
    return (total_loss_accum / max(n_batches, 1),
            data_loss_accum / max(n_batches, 1),
            physics_loss_accum / max(n_batches, 1),
            avg_diag)


def val_net(net, device, val_loader, loss_f, batch_size, use_physics=False):
    net.eval()
    
    total_loss_accum = 0.0
    data_loss_accum = 0.0
    physics_loss_accum = 0.0

    with torch.no_grad():
        for batch in val_loader:
            inputs, mu_gt, k_gt, mfre, fov, index = batch
            inputs, mu_gt, k_gt, mfre = inputs.to(device), mu_gt.to(device), k_gt.to(device), mfre.to(device, dtype=inputs.dtype)
            if isinstance(fov, (int, float)):
                fov = torch.tensor([fov] * inputs.shape[0], device=device, dtype=inputs.dtype)
            else:
                fov = fov.to(device, dtype=inputs.dtype)
                if fov.dim() == 0:  # scalar tensor
                    fov = fov.repeat(inputs.shape[0])            
            
            inputs = inputs.permute(4, 0, 1, 2, 3)
            
            inputs = inputs.permute(1,2,0,3,4) # If not jiaying architecture
            #with torch.cuda.amp.autocast():  # Mixed precision training
            k_pred = net(inputs)
            mu_pred = wave_number_to_shear_stiffness(k_pred, mfre, fov)

            if use_physics:
                # For physics-informed loss, pass outputs, ground truth, and mfre
                inputs_physics = inputs.permute(0,1,3,4,2)
                total_loss, data_loss, physics_loss_val = loss_f(k_pred, k_gt, mu_pred, mu_gt, inputs_physics, mfre,fov)
            else:
                # For regular MSE / other losses
                total_loss, data_loss, physics_loss_val = loss_f(k_pred, k_gt, mu_pred, mu_gt)

            total_loss_accum += total_loss.item()
            data_loss_accum += data_loss.item()
            physics_loss_accum += physics_loss_val.item()

    n_batches = len(val_loader)
    if n_batches == 0:
        print("WARNING: Validation loader is empty (0 batches). Check val_input path.")
        return 0.0, 0.0, 0.0
    return total_loss_accum / n_batches, data_loss_accum / n_batches, physics_loss_accum / n_batches


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# +
# run of the training and validation
def setup_and_run_train(
    train_input,
    val_input,
    dir_model,
    offsets,
    fov,
    batch_size,
    epochs,
    lr,
    arch_type,
    arch_subtype=None,
    loss_type = 'mse',
    #physics_params=None,
    lambda_physics = 1.0,
    lambda_data = 1.0,
    #lambda_ratio=0.1
    heterogeneous=False,
    orthogonalize_het=False,
    mse_in_k_space=False,
    harmonization='none',
    warmup_epochs=10
):
    """
    Setup and run training with unified loss interface.

    Args:
        loss_type: 'mse', 'residual', or 'ratio'
        lambda_data: weight for MSE loss (set to 0 to disable)
        lambda_physics: weight for physics loss (residual or ratio)
        heterogeneous: for residual loss, use het term
        orthogonalize_het: for ratio loss, orthogonalize T_het
        harmonization: 'none', 'oagh', or 'oagh_c'
            When enabled, replaces fixed lambda weighting with adaptive
            gradient harmonization. Requires loss_type != 'mse'.
    """
    set_seed(42)

    # Validate harmonization config
    assert harmonization in ('none', 'oagh', 'oagh_c'), \
        f"Unknown harmonization: {harmonization}. Choose from: none, oagh, oagh_c"
    if harmonization != 'none' and loss_type == 'mse':
        raise ValueError(
            f"OAGH harmonization requires a physics loss (residual or ratio), "
            f"but loss_type='{loss_type}'. Use --loss-type residual or ratio."
        )

    print(f"""
    ============================================
    Training Configuration
    ============================================
    Train Input:     {train_input}
    Validation Input:{val_input}
    Model Dir:       {dir_model}
    Batch Size:      {batch_size}
    Epochs:          {epochs}
    Learning Rate:   {lr}
    Architecture:    {arch_type} / {arch_subtype}
    --------------------------------------------
    Loss Configuration:
    Type:            {loss_type}
    MSE in k-space:  {mse_in_k_space}
    Heterogeneous:   {heterogeneous} (for residual)
    Orthogonalize:   {orthogonalize_het} (for ratio)
    Harmonization:   {harmonization}
    --------------------------------------------
    Loss Weights:
    lambda_data:     {lambda_data}{' (ignored by OAGH)' if harmonization != 'none' else ''}
    lambda_physics:  {lambda_physics}{' (ignored by OAGH)' if harmonization != 'none' else ''}
    ============================================
    """)
    viz=False
    time_start = time.time()
    batch_size = validate_batch_size(batch_size, torch.cuda.device_count())
    net, device = setup_model(arch_type, arch_subtype)
    
    train_loader = get_Pdataloader_for_train(train_input, offsets, fov[0], batch_size, snr_db=20)
    val_loader = get_Pdataloader_for_val(val_input, offsets, fov[0], batch_size, snr_db=20)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = torch.optim.Adam  (net.parameters(), lr=lr, weight_decay=0.05)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(2e3), eta_min=0, last_epoch=-1)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    if loss_type == 'mse':
        # Simple MSE loss (no physics)
        loss_f = MSELoss(mse_in_k_space=mse_in_k_space)
        use_physics = False
        print("Using MSE-only loss (no physics)")

    elif loss_type == 'residual':
        loss_f = CombinedResidualLoss(
            fov=fov,
            rho=1000,
            lambda_data=lambda_data,
            lambda_physics=lambda_physics,
            heterogeneous=heterogeneous,
            mse_in_k_space=mse_in_k_space,
            viz=viz,
            diagnostics=False
        )
        use_physics = True
        mode = "heterogeneous" if heterogeneous else "homogeneous"
        print(f"Using {mode} Residual Loss")
        print(f"  λ_data={lambda_data}, λ_physics={lambda_physics}")
        
#         if physics_params is None:
#             physics_params = {}  # Use default params in MREHelmholtzLoss

#         # Create Physics Loss
#         hom_loss = hom.MREHelmholtzLoss(**physics_params)
        
#         # Wrap it in CombinedLoss
#         loss_f = CombinedLoss(
#             lambda_hom=lambda_hom,  # Adjust weight as needed
#             lambda_data=lambda_data,
#             physics_loss=hom_loss
#         )
#         use_physics = True
#         print(f"Using CombinedLoss with Helmholtz physics, params: {physics_params}")
    elif loss_type == 'ratio':
        loss_f = CombinedRatioLoss(
            fov=fov,
            rho=1000,
            lambda_data=lambda_data,
            lambda_ratio=lambda_physics,  # Using lambda_physics as lambda_ratio
            orthogonalize_het=orthogonalize_het,
            mse_in_k_space=mse_in_k_space,
            viz=viz,
            diagnostics=False
        )
        use_physics = True
        ortho_str = "with" if orthogonalize_het else "without"
        print(f"Using Ratio Loss ({ortho_str} orthogonalization)")
        print(f"  λ_data={lambda_data}, λ_ratio={lambda_physics}")
        
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'mse', 'residual', or 'ratio'")

    # --- OAGH / OAGH-C Harmonizer Setup ---
    harmonizer = None
    if harmonization != 'none':
        harmonizer = OAGHHarmonizer(method=harmonization)
        print(f"OAGH Harmonizer: method={harmonization}")
        print(f"  Gradient harmonization will adaptively replace fixed lambda weighting.")

    #header = ['epoch', 'learning rate', 'train loss', 'val loss', 'time cost now/second']

    # Directory setup
    best_loss = float('inf')
    start_epoch = 0
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build directory structure
    lambda_data_str = str(lambda_data).replace('.', '_')
    lambda_physics_str = str(lambda_physics).replace('.', '_')
    
    path_components = [dir_model, arch_type]
    if arch_subtype:
        path_components.append(arch_subtype)
    
    path_components.append(loss_type)
    
    if loss_type == 'residual':
        res_type = 'het' if heterogeneous else 'hom'
        path_components.append(res_type)
    elif loss_type == 'ratio':
        ortho_tag = 'ortho' if orthogonalize_het else 'noortho'
        path_components.append(ortho_tag)
    
    # Add harmonization tag
    if harmonization != 'none':
        path_components.append(harmonization)

    # Add lambda tags
    path_components.append(f'lam_data{lambda_data_str}')
    path_components.append(f'lam_phys{lambda_physics_str}')
    path_components.append(f'bs{batch_size}')
    
    dir_model = os.path.join(*path_components)
    os.makedirs(dir_model, exist_ok=True)

    tb_run_dir = os.path.join(
        dir_model,
        "tensorboard_logs",
        f"run_{run_id}"
    )
    os.makedirs(tb_run_dir, exist_ok=True)
    
    with open(os.path.join(tb_run_dir, f"meta.txt"), "w") as f:
        f.write(f"run_id: {run_id}\n")
        f.write(f"loss_type: {loss_type}\n")
        f.write(f"mse_in_k_space: {mse_in_k_space}\n")
        f.write(f"heterogeneous: {heterogeneous}\n")
        f.write(f"orthogonalize_het: {orthogonalize_het}\n")
        f.write(f"lambda_data: {lambda_data}\n")
        f.write(f"lambda_physics: {lambda_physics}\n")
        f.write(f"harmonization: {harmonization}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"fov: {fov}\n")
    
    writer = SummaryWriter(log_dir=tb_run_dir)
    comment = f"Loss: {loss_type}, lambda_data={lambda_data},lambda_ratio={lambda_physics}, Batch={batch_size}, LR={lr}"
    writer.add_text("Run Notes", comment, global_step=0)
    writer.add_text("Run Info", f"run_id: {run_id}", 0)

    hparams = {
    "run_id": run_id,
    "epochs": epochs,
    "batch_size": batch_size,
    "lr": lr,
    "loss_type": loss_type,
    "mse_in_k_space": mse_in_k_space,
    "heterogeneous": heterogeneous,
    "orthogonalize_het": orthogonalize_het,
    "lambda_data": lambda_data,
    "lambda_physics": lambda_physics,
    "harmonization": harmonization,
    "fov": str(fov)
    }
    
    # --- OAGH warm-up configuration ---
    # When using OAGH, run standard training for the first warmup_epochs
    # to move k_pred into a physical range. At initialization k_pred ≈ 0,
    # which makes mu = rho*omega^2/k^2 → infinity, clamped to 15000 Pa.
    # This saturates the clamp and kills physics gradients (g_physics=0).
    # A warm-up with combined loss (single backward pass) lets the data loss
    # pull k_pred toward reasonable values, after which OAGH can function.
    warmup_epochs = warmup_epochs if harmonizer is not None else 0
    if warmup_epochs > 0:
        print(f"\n{'='*50}")
        print(f"OAGH Warm-up: {warmup_epochs} epochs of standard training")
        print(f"  (to bring k_pred into physical range before OAGH)")
        print(f"{'='*50}")

    for epoch in tqdm(range(start_epoch, epochs)):

        # --- Train one epoch ---
        # Use standard training during warm-up, then switch to OAGH
        use_oagh_this_epoch = (harmonizer is not None) and (epoch >= warmup_epochs)

        if use_oagh_this_epoch:
            # OAGH / OAGH-C path: per-task gradient harmonization
            train_total, train_data, train_physics, oagh_diag = train_net_oagh(
                net, device, train_loader, optimizer, grad_scaler, loss_f,
                batch_size, harmonizer, use_physics=use_physics
            )
        else:
            # Standard path: fixed lambda weighting (also used for warm-up)
            if harmonizer is not None and epoch == 0:
                print(f"[Warm-up] Epoch {epoch}: standard training (combined loss)")
            if harmonizer is not None and epoch == warmup_epochs - 1:
                print(f"[Warm-up] Last warm-up epoch. Switching to OAGH next epoch.")
            train_total, train_data, train_physics = train_net(
                net, device, train_loader, optimizer, grad_scaler, loss_f,
                batch_size, use_physics=use_physics
            )
            oagh_diag = None

        val_total, val_data, val_physics = val_net(net, device, val_loader, loss_f, batch_size, use_physics=use_physics)
        scheduler.step()
        print('\Learning rate = ', optimizer.param_groups[0]['lr'], end=' ')
        time_cost_now = time.time() - time_start

        # TensorBoard logging
        writer.add_scalar('Loss/Train_Total', train_total, epoch)
        writer.add_scalar('Loss/Train_Data', train_data, epoch)
        writer.add_scalar('Loss/Train_Data_Weighted', lambda_data*train_data, epoch)
        writer.add_scalar('Loss/Train_Physics', train_physics, epoch)
        writer.add_scalar('Loss/Train_Physics_Weighted', lambda_physics  * train_physics, epoch)

        writer.add_scalar('Loss/Val_Total', val_total, epoch)
        writer.add_scalar('Loss/Val_Data', val_data, epoch)
        writer.add_scalar('Loss/Val_Physics', val_physics, epoch)
        writer.add_scalar('Loss/Val_Physics_Weighted', lambda_physics  * val_physics, epoch)
        writer.add_scalar('Loss/Val_Data_Weighted', lambda_data*val_data, epoch)

        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Time/Elapsed', time_cost_now, epoch)

        # OAGH diagnostics to TensorBoard
        if oagh_diag is not None:
            writer.add_scalar('OAGH/alpha', oagh_diag['alpha'], epoch)
            writer.add_scalar('OAGH/f_alpha', oagh_diag['f_alpha'], epoch)
            writer.add_scalar('OAGH/alpha_conflict', oagh_diag['alpha_conflict'], epoch)
            writer.add_scalar('OAGH/alpha_drift', oagh_diag['alpha_drift'], epoch)
        
        # Save model
        if val_total < best_loss:
            best_loss = val_total
            # Store entire checkpoint dictionary - checkpoint = torch.load(dir_model, map_location=device)
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'train_total_loss': train_total,
                    'train_data_loss': train_data,
                    'train_physics_loss': train_physics,
                    'val_total_loss': val_total,
                    'val_data_loss': val_data,
                    'val_physics_loss': val_physics,
                    'best_loss': best_loss,
                    'loss_type': loss_type,
                    'mse_in_k_space': mse_in_k_space,
                    'heterogeneous': heterogeneous,
                    'orthogonalize_het': orthogonalize_het,
                    'lambda_data': lambda_data,
                    'lambda_physics': lambda_physics,
                    'harmonization': harmonization,
                    'fov': fov,
                    'run_id': run_id
                }, os.path.join(dir_model, "weights.pth"))
    time_all = time.time() - time_start
    final_metrics = {"val_loss": best_loss,
                    "total_time": time_all}
    writer.add_hparams(hparams, final_metrics)
    writer.close()
    del net
    del optimizer
    del scheduler
    del grad_scaler
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"{'='*50}")
    print(f"Total time: {time_all:.2f} seconds ({time_all/60:.2f} minutes)")
    print(f"Best validation loss: {best_loss:.4e}")
    print(f"Model saved to: {dir_model}")
    print(f"TensorBoard logs: {tb_run_dir}")
    print(f"{'='*50}\n")


# -

def setup_model(arch_type, arch_subtype=None):
    """
    Proper model setup with DataParallel support
    """
    set_seed(42)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Create model
    if arch_type == 'FDTDNet':
        net = Net()
    elif arch_type == 'SimpleTFusionUNet3D':
        net = SimpleTFusionUNet3D()
    elif arch_type == 'MRE_UTAE':
        net = MRE_UTAE()
    elif arch_type == 'FNO3dTo2d':
        net = FNO3dTo2d(fusion=f'{arch_subtype}')
    elif arch_type == 'UNO':
        net = UNONet(hidden_channels=16, temporal_mode=f'{arch_subtype}')
    else:
        raise ValueError('Incorrect Architecture Type')
    

    # Handle multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        # Apply SyncBatchNorm for better multi-GPU training (if model has BatchNorm layers)
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = nn.DataParallel(net)
    
    # Move to device (only once, at the end)
    net = net.to(device)
    if not test_model_forward_pass(net, device):
        print("Model forward pass test failed. Aborting training.")
        return
    
    return net, device


def test_model_forward_pass(net, device, sample_shape=(64, 1, 256, 256, 8)):
    """
    Test the model with a dummy input to catch errors early
    """
    print("Testing model forward pass...")
    try:
        # Create dummy input matching your expected shape
        dummy_input = torch.randn(sample_shape).to(device)
        
        # Apply your permutations
        dummy_input = dummy_input.permute(4, 0, 1, 2, 3)
        dummy_input = dummy_input.permute(1, 2, 0, 3, 4)  # B, 1, 8, 256, 256
        
        print(f"Dummy input shape: {dummy_input.shape}")
        
        net.eval()
        with torch.no_grad():
            output = net(dummy_input)
            print(f"Forward pass successful! Output shape: {output.shape}")
            return True
    except Exception as e:
        print(f"Forward pass test failed: {e}")
        return False


def validate_batch_size(batch_size, num_gpus):
    """
    Validate and adjust batch size for DataParallel
    """
    if num_gpus > 1:
        if batch_size % num_gpus != 0:
            adjusted_batch_size = batch_size + (num_gpus - batch_size % num_gpus)
            print(f"Adjusting batch size from {batch_size} to {adjusted_batch_size} for {num_gpus} GPUs")
            return adjusted_batch_size
        else:
            print(f"Batch size {batch_size} is compatible with {num_gpus} GPUs")
    return batch_size


def _softplus_lower(x, lo, sharpness=10.0):
    """Soft lower clamp: smoothly enforces x >= lo while preserving gradients.
    Uses softplus: result = lo + softplus(x - lo) ≈ x when x >> lo."""
    return lo + torch.nn.functional.softplus(x - lo, beta=sharpness)


def _soft_clamp(x, lo, hi, sharpness=10.0):
    """Soft clamp to [lo, hi] that preserves gradients everywhere.
    Composes two softplus operations for differentiable clamping."""
    # soft lower bound
    x = _softplus_lower(x, lo, sharpness)
    # soft upper bound: hi - softplus(hi - x)
    x = hi - torch.nn.functional.softplus(hi - x, beta=sharpness)
    return x


def wave_number_to_shear_stiffness(k_pred, mfre, fov, rho=1000.0, eps=1e-6, clamp_mu=(100, 15000)):
    """
    Convert predicted wave number to shear stiffness (mu)

    Args:
        k_pred (Tensor): [B, 1, H, W] predicted wave number
        mfre (Tensor):   [B,1,1,1] mechanical frequency (Hz)
        fov (Tensor):    [B] field of view scalar
        eps (float):     numerical stability constant
        clamp_mu (tuple): (min, max) for soft clamping mu

    Returns:
        mu (Tensor): [B, 1, H, W] shear stiffness map

    Note: Uses soft (differentiable) clamping so that gradients flow through
    even when values are near the clamp boundaries. Hard torch.clamp() kills
    gradients when values are outside the range, which blocks physics loss
    gradients during early training when k_pred is near zero.
    """
    # Ensure correct shape for broadcasting
    mfre = mfre.view(-1, 1, 1, 1)  # [B,1,1,1]
    fov = fov.view(-1, 1, 1, 1)  # [B,1,1,1]

    # Soft lower bound on k to avoid division by zero
    # softplus(k_pred - 0.1) + 0.1 ≈ k_pred when k_pred >> 0.1
    k_safe = _softplus_lower(k_pred, 0.1, sharpness=10.0)

    omega = 2 * torch.pi * mfre
    mu_pa = rho * omega**2 / k_safe**2

    # Soft clamp mu to physical range [100, 15000] Pa
    mu_pa = _soft_clamp(mu_pa, clamp_mu[0], clamp_mu[1], sharpness=0.01)
    return mu_pa

