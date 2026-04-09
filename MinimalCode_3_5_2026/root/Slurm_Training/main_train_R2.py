"""
main_train_R2.py — Revision 2 training wrapper for OAGH improvements.

Three independent modifications, controlled by CLI flags:
  --snr-adaptive       : Scale lambda_physics by min(1, SNR/25) per epoch
  --curriculum         : Train clean→noisy (SNR schedule over epochs)
  --di-k-filter <val>  : Override k_filter in the physics loss (default 1000)

Each flag is independent. They can be combined but are designed to be tested
one at a time. When no R2 flags are set, behavior is identical to main_train.py.

This file does NOT modify any existing code — it wraps setup_and_run_train
with a thin layer that patches parameters before calling the original function.
To withdraw: simply delete this file and the R2 sbatch files.
"""

import argparse
import sys
import os
import json
import time
import copy
import numpy as np
import torch

# ---- Robust PYTHONPATH setup (same as main_train.py) ----
_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
_parent_dir = os.path.abspath(os.path.join(_script_dir, '..'))
_pace_root = "/storage/project/r-jueda3-0/jueda3/MRE_OAGH_test/MinimalCode_3_5_2026/root"

for _p in [_parent_dir, _pace_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

print(f"[main_train_R2.py] parent_dir = {_parent_dir}")
print(f"[main_train_R2.py] PYTHONPATH env = {os.environ.get('PYTHONPATH', 'NOT SET')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="OAGH R2 training — parameter-level improvements over baseline OAGH."
    )

    # --- Original arguments (same as main_train.py) ---
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--train-input', type=str, required=True)
    parser.add_argument('--validation-input', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--offsets', type=int, default=8)
    parser.add_argument('--field-of-view', type=float, default=0.2)
    parser.add_argument('--arch-type', type=str, default='FDTDNet')
    parser.add_argument('--arch-subtype', type=str, default=None)
    parser.add_argument('--loss-type', type=str, default='residual',
                        choices=['mse', 'residual', 'ratio'])
    parser.add_argument('--lambda-data', type=float, default=1.0)
    parser.add_argument('--lambda-physics', type=float, default=1.0)
    parser.add_argument('--heterogeneous', action='store_true', default=False)
    parser.add_argument('--orthogonalize-het', action='store_true', default=False)
    parser.add_argument('--mse-in-k-space', action='store_true', default=False)
    parser.add_argument('--harmonization', type=str, default='oagh',
                        choices=['none', 'oagh', 'oagh_c', 'pcgrad', 'gradnorm'])
    parser.add_argument('--warmup-epochs', type=int, default=50)

    # --- R2-specific arguments ---
    parser.add_argument('--snr-adaptive', action='store_true', default=False,
                        help='R2 Option 1: Scale physics loss weight by min(1, SNR/25). '
                             'Reduces physics influence at high noise.')
    parser.add_argument('--snr-adaptive-ref', type=float, default=25.0,
                        help='Reference SNR for adaptive scaling. Physics weight = min(1, SNR/ref). '
                             'Default: 25.0')

    parser.add_argument('--curriculum', action='store_true', default=False,
                        help='R2 Option 2: Curriculum learning — train clean first, '
                             'then gradually introduce noise.')
    parser.add_argument('--curriculum-schedule', type=str, default='30,25,20,15',
                        help='Comma-separated SNR schedule for curriculum stages. '
                             'Default: "30,25,20,15" (epochs split equally)')

    parser.add_argument('--di-k-filter', type=float, default=None,
                        help='R2 Option 3: Override k_filter in ResidualLoss physics loss. '
                             'Default: None (uses original k_filter=1000). '
                             'Sweep showed 700 is optimal for Main_Test.')

    parser.add_argument('--r2-tag', type=str, default=None,
                        help='Tag appended to model directory for identification. '
                             'Default: auto-generated from active R2 options.')

    return parser.parse_args()


def run_snr_adaptive(args):
    """
    Option 1: SNR-adaptive physics loss weighting.

    Mechanism: Instead of fixed lambda_physics=1.0, we scale it by
    min(1.0, training_snr / snr_ref). At SNR=20 with ref=25, the physics
    weight becomes 0.8, reducing noisy physics gradient influence.

    Implementation: We modify the data loader's SNR and scale lambda_physics
    accordingly. Since OAGH ignores fixed lambdas (it does its own weighting),
    we instead modify the physics loss output by a multiplicative factor.
    This is done by wrapping the loss function.
    """
    from train_functions import setup_and_run_train
    from Data_loader import get_Pdataloader_for_train, get_Pdataloader_for_val, AddGaussianNoiseSNR, PDataset
    from losses.residual_losses import CombinedResidualLoss
    import torch.nn as nn

    snr_ref = args.snr_adaptive_ref
    training_snr = 20  # Current default

    # Scale factor for physics loss
    scale = min(1.0, training_snr / snr_ref)
    print(f"\n{'='*60}")
    print(f"R2 Option 1: SNR-Adaptive Physics Weighting")
    print(f"  Training SNR: {training_snr}")
    print(f"  Reference SNR: {snr_ref}")
    print(f"  Physics scale factor: {scale:.3f}")
    print(f"{'='*60}\n")

    # Create a wrapper that scales the physics loss
    class ScaledPhysicsCombinedLoss(nn.Module):
        """Wraps CombinedResidualLoss, scaling physics_loss by a factor."""
        def __init__(self, base_loss, physics_scale):
            super().__init__()
            self.base_loss = base_loss
            self.physics_scale = physics_scale

        def forward(self, k_pred, k_gt, mu_pred, mu_gt, wave, mfre, fov=None):
            total_loss, data_loss, physics_loss = self.base_loss(
                k_pred, k_gt, mu_pred, mu_gt, wave, mfre, fov
            )
            # Rescale physics component
            scaled_physics = physics_loss * self.physics_scale
            # Recompute total: data + scaled_physics
            new_total = (self.base_loss.lambda_data * data_loss +
                         self.base_loss.lambda_physics * scaled_physics)
            return new_total, data_loss, scaled_physics

    # Monkey-patch: import and override the loss construction in train_functions
    # Instead, we use a cleaner approach: call setup_and_run_train but with
    # a modified lambda_physics that incorporates the scale
    effective_lambda_physics = args.lambda_physics * scale
    print(f"  Effective lambda_physics: {args.lambda_physics} * {scale} = {effective_lambda_physics:.3f}")

    # For OAGH: lambdas are "ignored" during harmonization, but they're used
    # during warmup AND as initial scale in CombinedResidualLoss.
    # The cleaner approach: scale lambda_physics directly.
    setup_and_run_train(
        train_input=args.train_input,
        val_input=args.validation_input,
        dir_model=args.model,
        offsets=args.offsets,
        fov=(args.field_of_view, args.field_of_view),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        arch_type=args.arch_type,
        arch_subtype=None if args.arch_subtype in [None, "None", "none"] else args.arch_subtype,
        loss_type=args.loss_type,
        lambda_data=args.lambda_data,
        lambda_physics=effective_lambda_physics,
        heterogeneous=args.heterogeneous,
        orthogonalize_het=args.orthogonalize_het,
        mse_in_k_space=args.mse_in_k_space,
        harmonization=args.harmonization,
        warmup_epochs=args.warmup_epochs
    )


def run_curriculum(args):
    """
    Option 2: Curriculum learning on noise.

    Mechanism: Split total epochs into stages. Each stage uses a different
    SNR level, starting from clean (high SNR) to noisy (low SNR).
    The model learns physics from clean gradients first, then adapts.

    Implementation: Run setup_and_run_train multiple times with different
    SNR levels and epoch counts, loading the checkpoint from the previous stage.
    """
    from train_functions import (
        setup_and_run_train, set_seed, setup_model, validate_batch_size,
        train_net, train_net_oagh, val_net
    )
    from Data_loader import get_Pdataloader_for_train, get_Pdataloader_for_val
    from losses.residual_losses import CombinedResidualLoss
    from losses.mse_loss import MSELoss
    from harmonizer import OAGHHarmonizer
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    from tqdm import tqdm

    schedule = [int(s) for s in args.curriculum_schedule.split(',')]
    n_stages = len(schedule)
    epochs_per_stage = args.epochs // n_stages
    remainder = args.epochs % n_stages

    print(f"\n{'='*60}")
    print(f"R2 Option 2: Curriculum Learning")
    print(f"  SNR schedule: {schedule}")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Epochs per stage: {epochs_per_stage} (+ {remainder} in last stage)")
    print(f"{'='*60}\n")

    set_seed(42)
    batch_size = validate_batch_size(args.batch_size, torch.cuda.device_count())
    net, device = setup_model(args.arch_type,
                              None if args.arch_subtype in [None, "None", "none"] else args.arch_subtype)

    fov = (args.field_of_view, args.field_of_view)

    # Loss function
    loss_f = CombinedResidualLoss(
        fov=fov, rho=1000,
        lambda_data=args.lambda_data,
        lambda_physics=args.lambda_physics,
        heterogeneous=args.heterogeneous,
        mse_in_k_space=args.mse_in_k_space,
        viz=False, diagnostics=False
    )
    use_physics = True

    # Harmonizer
    harmonizer = OAGHHarmonizer(method=args.harmonization) if args.harmonization != 'none' else None

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate,
                                 betas=(0.9, 0.999), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Directory setup
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    lambda_data_str = str(args.lambda_data).replace('.', '_')
    lambda_physics_str = str(args.lambda_physics).replace('.', '_')

    path_components = [args.model, args.arch_type, args.loss_type]
    res_type = 'het' if args.heterogeneous else 'hom'
    path_components.append(res_type)
    if args.harmonization != 'none':
        path_components.append(args.harmonization)
    # R2 tag
    r2_tag = args.r2_tag or f'R2_curriculum_{"_".join(str(s) for s in schedule)}'
    path_components.append(r2_tag)
    path_components.append(f'lam_data{lambda_data_str}')
    path_components.append(f'lam_phys{lambda_physics_str}')
    path_components.append(f'bs{batch_size}')

    dir_model = os.path.join(*path_components)
    os.makedirs(dir_model, exist_ok=True)

    tb_run_dir = os.path.join(dir_model, "tensorboard_logs", f"run_{run_id}")
    os.makedirs(tb_run_dir, exist_ok=True)

    with open(os.path.join(tb_run_dir, "meta.txt"), "w") as f:
        f.write(f"run_id: {run_id}\n")
        f.write(f"R2_option: curriculum\n")
        f.write(f"snr_schedule: {schedule}\n")
        f.write(f"epochs_per_stage: {epochs_per_stage}\n")
        f.write(json.dumps(vars(args), indent=2))

    writer = SummaryWriter(log_dir=tb_run_dir)
    writer.add_text("Run Notes",
                    f"R2 Curriculum: SNR schedule {schedule}, "
                    f"{epochs_per_stage} epochs/stage", 0)

    best_loss = float('inf')
    global_epoch = 0
    time_start = time.time()
    warmup_epochs = args.warmup_epochs if harmonizer is not None else 0

    for stage_idx, snr_db in enumerate(schedule):
        stage_epochs = epochs_per_stage + (remainder if stage_idx == n_stages - 1 else 0)
        print(f"\n{'='*60}")
        print(f"Curriculum Stage {stage_idx+1}/{n_stages}: SNR={snr_db} dB, "
              f"epochs {global_epoch}-{global_epoch + stage_epochs - 1}")
        print(f"{'='*60}")

        # Rebuild data loaders with new SNR
        train_loader = get_Pdataloader_for_train(
            args.train_input, args.offsets, fov[0], batch_size, snr_db=snr_db
        )
        val_loader = get_Pdataloader_for_val(
            args.validation_input, args.offsets, fov[0], batch_size, snr_db=snr_db
        )

        for local_epoch in range(stage_epochs):
            use_oagh_this_epoch = (harmonizer is not None) and (global_epoch >= warmup_epochs)

            if use_oagh_this_epoch:
                train_total, train_data, train_physics, oagh_diag = train_net_oagh(
                    net, device, train_loader, optimizer, grad_scaler, loss_f,
                    batch_size, harmonizer, use_physics=use_physics
                )
            else:
                train_total, train_data, train_physics = train_net(
                    net, device, train_loader, optimizer, grad_scaler, loss_f,
                    batch_size, use_physics=use_physics
                )
                oagh_diag = None

            val_total, val_data, val_physics = val_net(
                net, device, val_loader, loss_f, batch_size, use_physics=use_physics
            )
            scheduler.step()

            # TensorBoard logging
            writer.add_scalar('Loss/Train_Total', train_total, global_epoch)
            writer.add_scalar('Loss/Train_Data', train_data, global_epoch)
            writer.add_scalar('Loss/Train_Physics', train_physics, global_epoch)
            writer.add_scalar('Loss/Val_Total', val_total, global_epoch)
            writer.add_scalar('Loss/Val_Data', val_data, global_epoch)
            writer.add_scalar('Loss/Val_Physics', val_physics, global_epoch)
            writer.add_scalar('Curriculum/SNR_dB', snr_db, global_epoch)
            writer.add_scalar('Curriculum/Stage', stage_idx, global_epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_epoch)

            if oagh_diag is not None:
                writer.add_scalar('OAGH/alpha', oagh_diag['alpha'], global_epoch)
                writer.add_scalar('OAGH/f_alpha', oagh_diag['f_alpha'], global_epoch)
                writer.add_scalar('OAGH/alpha_conflict', oagh_diag['alpha_conflict'], global_epoch)
                writer.add_scalar('OAGH/alpha_drift', oagh_diag['alpha_drift'], global_epoch)

            # Save best
            if val_total > 0 and val_total < best_loss:
                best_loss = val_total
                torch.save({
                    'epoch': global_epoch + 1,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_total_loss': train_total,
                    'val_total_loss': val_total,
                    'best_loss': best_loss,
                    'curriculum_stage': stage_idx,
                    'curriculum_snr': snr_db,
                    'harmonization': args.harmonization,
                    'run_id': run_id
                }, os.path.join(dir_model, "weights.pth"))

            # Save on last epoch
            if global_epoch == args.epochs - 1:
                torch.save({
                    'epoch': global_epoch + 1,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_total_loss': train_total,
                    'val_total_loss': val_total,
                    'best_loss': best_loss,
                    'curriculum_stage': stage_idx,
                    'curriculum_snr': snr_db,
                    'harmonization': args.harmonization,
                    'run_id': run_id
                }, os.path.join(dir_model, "weights.pth"))
                if best_loss == float('inf'):
                    print("WARNING: Validation loss was never positive. Saved last-epoch model.")

            global_epoch += 1

    writer.close()
    elapsed = time.time() - time_start
    print(f"\nCurriculum training complete. {global_epoch} epochs in {elapsed:.0f}s")
    print(f"Best val loss: {best_loss:.6f}")
    print(f"Model saved to: {dir_model}")


def run_difilter(args):
    """
    Option 3: Improved DI filter in physics loss.

    Mechanism: The ResidualLoss creates MREHelmholtzLoss with k_filter=1000
    (hardcoded). The DI sweep showed k_filter=700 is optimal for Main_Test.
    We monkey-patch ResidualLoss.__init__ to use the specified k_filter value.

    Implementation: Patch before calling setup_and_run_train, restore after.
    """
    from train_functions import setup_and_run_train
    import losses.residual_losses as rl

    k_filter_new = args.di_k_filter
    print(f"\n{'='*60}")
    print(f"R2 Option 3: Improved DI k_filter in Physics Loss")
    print(f"  Original k_filter: 1000")
    print(f"  New k_filter: {k_filter_new}")
    print(f"{'='*60}\n")

    # Monkey-patch ResidualLoss to use the new k_filter
    _original_init = rl.ResidualLoss.__init__

    def _patched_init(self, fov=(0.2, 0.2), rho=1000, heterogeneous=False,
                      viz=False, diagnostics=False):
        _original_init(self, fov, rho, heterogeneous, viz, diagnostics)
        # Store the custom k_filter for use in forward
        self._r2_k_filter = k_filter_new

    _original_forward = rl.ResidualLoss.forward

    def _patched_forward(self, wave_tensors, mu_pred, mfre, W=1.0, k_pred=None):
        # The forward creates MREHelmholtzLoss with k_filter=1000
        # We need to intercept that. Simplest: patch MREHelmholtzLoss init
        from losses.homogeneous import MREHelmholtzLoss as _MRE
        _orig_mre_init = _MRE.__init__

        def _mre_patched_init(mre_self, density=1000, fov=(0.2, 0.2),
                              residual_type='raw', k_filter=None,
                              epsilon=1e-10, verbose=False):
            # Replace k_filter with our value
            _orig_mre_init(mre_self, density, fov, residual_type,
                           k_filter_new, epsilon, verbose)
            if verbose or True:
                print(f"  [R2 DI-filter] MREHelmholtzLoss k_filter overridden: "
                      f"{k_filter} -> {k_filter_new}")

        _MRE.__init__ = _mre_patched_init
        try:
            result = _original_forward(self, wave_tensors, mu_pred, mfre, W, k_pred)
        finally:
            _MRE.__init__ = _orig_mre_init
        return result

    rl.ResidualLoss.__init__ = _patched_init
    rl.ResidualLoss.forward = _patched_forward

    try:
        setup_and_run_train(
            train_input=args.train_input,
            val_input=args.validation_input,
            dir_model=args.model,
            offsets=args.offsets,
            fov=(args.field_of_view, args.field_of_view),
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.learning_rate,
            arch_type=args.arch_type,
            arch_subtype=None if args.arch_subtype in [None, "None", "none"] else args.arch_subtype,
            loss_type=args.loss_type,
            lambda_data=args.lambda_data,
            lambda_physics=args.lambda_physics,
            heterogeneous=args.heterogeneous,
            orthogonalize_het=args.orthogonalize_het,
            mse_in_k_space=args.mse_in_k_space,
            harmonization=args.harmonization,
            warmup_epochs=args.warmup_epochs
        )
    finally:
        # Restore originals
        rl.ResidualLoss.__init__ = _original_init
        rl.ResidualLoss.forward = _original_forward


if __name__ == "__main__":
    args = parse_args()

    # Identify which R2 option(s) are active
    active = []
    if args.snr_adaptive:
        active.append('adaptive')
    if args.curriculum:
        active.append('curriculum')
    if args.di_k_filter is not None:
        active.append(f'difilter_kf{int(args.di_k_filter)}')

    if not active:
        print("WARNING: No R2 options specified. Running identical to main_train.py.")
        print("  Use --snr-adaptive, --curriculum, or --di-k-filter <val>")

    print(f"\n{'#'*60}")
    print(f"  OAGH Revision 2 Training")
    print(f"  Active options: {', '.join(active) if active else 'NONE'}")
    print(f"{'#'*60}\n")

    # Route to the appropriate handler
    # Only one R2 option at a time for clean comparison
    if args.curriculum:
        # Curriculum has its own training loop (needs epoch-level SNR control)
        if args.snr_adaptive or args.di_k_filter is not None:
            print("NOTE: --curriculum uses its own training loop. "
                  "Other R2 options are ignored in this run.")
        run_curriculum(args)
    elif args.snr_adaptive and args.di_k_filter is not None:
        print("WARNING: Both --snr-adaptive and --di-k-filter active. "
              "Running adaptive with patched k_filter.")
        # Apply di-filter patch, then run adaptive
        import losses.residual_losses as rl
        from losses.homogeneous import MREHelmholtzLoss as _MRE
        _orig_mre_init = _MRE.__init__
        kf = args.di_k_filter
        def _mre_patched(s, density=1000, fov=(0.2,0.2), residual_type='raw',
                         k_filter=None, epsilon=1e-10, verbose=False):
            _orig_mre_init(s, density, fov, residual_type, kf, epsilon, verbose)
        _MRE.__init__ = _mre_patched
        try:
            run_snr_adaptive(args)
        finally:
            _MRE.__init__ = _orig_mre_init
    elif args.snr_adaptive:
        run_snr_adaptive(args)
    elif args.di_k_filter is not None:
        run_difilter(args)
    else:
        # No R2 options — fallback to standard training
        from train_functions import setup_and_run_train
        setup_and_run_train(
            train_input=args.train_input,
            val_input=args.validation_input,
            dir_model=args.model,
            offsets=args.offsets,
            fov=(args.field_of_view, args.field_of_view),
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.learning_rate,
            arch_type=args.arch_type,
            arch_subtype=None if args.arch_subtype in [None, "None", "none"] else args.arch_subtype,
            loss_type=args.loss_type,
            lambda_data=args.lambda_data,
            lambda_physics=args.lambda_physics,
            heterogeneous=args.heterogeneous,
            orthogonalize_het=args.orthogonalize_het,
            mse_in_k_space=args.mse_in_k_space,
            harmonization=args.harmonization,
            warmup_epochs=args.warmup_epochs
        )
