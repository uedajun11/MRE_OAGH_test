"""
main_train_R2.py — Revision 2 training wrapper for OAGH improvements.

One remaining modification, controlled by a CLI flag:
  --curriculum         : Train clean→noisy (SNR schedule over epochs)

When no R2 flags are set, behavior is identical to main_train.py.

This file does NOT modify any existing code — it wraps setup_and_run_train
with a thin layer that patches parameters before calling the original function.
To withdraw: simply delete this file and the R2 sbatch file.

History of tested R2 options:
  - R2 Option 1 (SNR-adaptive, PACE eval 20260410_111851) — WITHDRAWN.
    Paired t-tests showed baseline OAGH won on all 20/20 test conditions
    (p < 1e-7, Cohen's d 0.17–7.11). Code and sbatch removed.
  - R2 Option 3 (DI k_filter=700, same eval) — WITHDRAWN. Ran to completion
    and overwrote the baseline OAGH checkpoint, but produced essentially
    identical test MAE (differences of 0.12–0.34 Pa on Main_Test). OAGH's
    gradient harmonization absorbs k_filter magnitude changes. The DI-optimal
    k_filter does not transfer to OAGH-trained networks. Code and sbatch
    removed.
  - R2 Option 2 (Curriculum) — NEVER TESTED. The previous run crashed at
    startup on an import typo (losses.mse_loss → losses.MSELoss, dead code).
    That bug is fixed; curriculum is the only remaining R2 candidate and is
    reworked below with a fixed-SNR validation loader so that best_loss is
    comparable across stages.
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
    parser.add_argument('--curriculum', action='store_true', default=False,
                        help='R2 Option 2: Curriculum learning — train clean first, '
                             'then gradually introduce noise.')
    parser.add_argument('--curriculum-schedule', type=str, default='30,25,20,15',
                        help='Comma-separated SNR schedule for curriculum stages. '
                             'Default: "30,25,20,15" (epochs split equally)')
    parser.add_argument('--curriculum-val-snr', type=float, default=None,
                        help='Fixed SNR (dB) used for validation throughout curriculum. '
                             'Default: None → uses min(schedule), i.e. worst-case noise. '
                             'Using a fixed val SNR is essential so best_loss is '
                             'comparable across stages; otherwise best_loss would '
                             'always save the stage-1 (cleanest) checkpoint.')
    parser.add_argument('--curriculum-warmup-in-stage1-only', action='store_true',
                        default=True,
                        help='If set, --warmup-epochs of plain training happens only '
                             'at the start of stage 1 (cleanest stage), and OAGH is '
                             'active for all subsequent stages. Default: True.')

    parser.add_argument('--r2-tag', type=str, default=None,
                        help='Tag appended to model directory for identification. '
                             'Default: auto-generated from active R2 options.')

    return parser.parse_args()


def run_curriculum(args):
    """
    Option 2: Curriculum learning on noise.

    Mechanism: Split total epochs into stages. Each stage uses a different
    training SNR, starting from clean (high SNR) and gradually introducing
    noise (low SNR). The model learns the physics from clean gradients first,
    then adapts to noisy gradients in later stages without catastrophic
    forgetting.

    KEY FIX (vs previous version): The validation loader uses a FIXED SNR
    (args.curriculum_val_snr, default = min(schedule)) throughout the entire
    curriculum. This makes best_loss comparable across stages. In the old
    version the val loader was rebuilt per stage at the stage's training SNR,
    which made stage 1 (cleanest) always produce the smallest val_loss and
    best_loss would never update in later stages — effectively saving the
    useless clean-only checkpoint.

    Checkpoints saved:
      weights.pth        — best-by-fixed-SNR-val (robust generalization)
      weights_final.pth  — last-epoch snapshot   (fully noise-adapted)
    """
    from train_functions import (
        set_seed, setup_model, validate_batch_size,
        train_net, train_net_oagh, val_net
    )
    from Data_loader import get_Pdataloader_for_train, get_Pdataloader_for_val
    from losses.residual_losses import CombinedResidualLoss
    from losses.oagh_harmonizer import OAGHHarmonizer
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime

    schedule = [int(s) for s in args.curriculum_schedule.split(',')]
    n_stages = len(schedule)
    epochs_per_stage = args.epochs // n_stages
    remainder = args.epochs % n_stages

    # Pick a fixed validation SNR — default to worst-case (minimum of schedule)
    # so best_loss tracks the hardest generalization target.
    val_snr_db = (args.curriculum_val_snr
                  if args.curriculum_val_snr is not None
                  else float(min(schedule)))

    print(f"\n{'='*60}")
    print(f"R2 Option 2: Curriculum Learning")
    print(f"  Training SNR schedule: {schedule}")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Epochs per stage: {epochs_per_stage} (+ {remainder} in last stage)")
    print(f"  Validation SNR (fixed): {val_snr_db} dB")
    print(f"  Warmup epochs (OAGH-off): {args.warmup_epochs} (at start of stage 1)")
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

    # IMPORTANT: print tb_run_dir so the evaluator notebook
    # (Model Checker Ueda dynamic 2.ipynb) can extract run_YYYYMMDD_HHMMSS
    # via its regex r'(run_\d{8}_\d{6})' on the Slurm report.
    # Without this line the report is flagged as "(no ML run - direct inversion)"
    # and the curriculum checkpoint is never evaluated.
    print(f"TensorBoard logs: {tb_run_dir}", flush=True)
    print(f"run_id: run_{run_id}", flush=True)

    with open(os.path.join(tb_run_dir, "meta.txt"), "w") as f:
        f.write(f"run_id: {run_id}\n")
        f.write(f"R2_option: curriculum\n")
        f.write(f"snr_schedule: {schedule}\n")
        f.write(f"val_snr_db_fixed: {val_snr_db}\n")
        f.write(f"epochs_per_stage: {epochs_per_stage}\n")
        f.write(json.dumps(vars(args), indent=2))

    writer = SummaryWriter(log_dir=tb_run_dir)
    writer.add_text("Run Notes",
                    f"R2 Curriculum: training schedule {schedule}, "
                    f"val_snr={val_snr_db} dB, {epochs_per_stage} epochs/stage", 0)

    # Build the validation loader ONCE with the fixed val SNR.
    # This is the key fix — makes best_loss comparable across stages.
    val_loader = get_Pdataloader_for_val(
        args.validation_input, args.offsets, fov[0], batch_size, snr_db=val_snr_db
    )
    print(f"[curriculum] Built fixed-SNR val loader at {val_snr_db} dB "
          f"({len(val_loader)} batches)")

    best_loss = float('inf')
    best_epoch = -1
    best_stage = -1
    global_epoch = 0
    time_start = time.time()
    # Warmup runs plain training (no OAGH) only at the very beginning.
    warmup_epochs = args.warmup_epochs if harmonizer is not None else 0

    for stage_idx, snr_db in enumerate(schedule):
        stage_epochs = epochs_per_stage + (remainder if stage_idx == n_stages - 1 else 0)
        stage_time_start = time.time()
        print(f"\n{'='*60}")
        print(f"Curriculum Stage {stage_idx+1}/{n_stages}: train SNR={snr_db} dB, "
              f"epochs {global_epoch}-{global_epoch + stage_epochs - 1}")
        print(f"{'='*60}")

        # Rebuild ONLY the training loader at the new stage SNR.
        train_loader = get_Pdataloader_for_train(
            args.train_input, args.offsets, fov[0], batch_size, snr_db=snr_db
        )
        print(f"[curriculum] Stage {stage_idx+1} train loader: {len(train_loader)} batches")

        stage_best = float('inf')
        for local_epoch in range(stage_epochs):
            epoch_start = time.time()
            use_oagh_this_epoch = (harmonizer is not None) and (global_epoch >= warmup_epochs)

            if use_oagh_this_epoch:
                train_total, train_data, train_physics, oagh_diag = train_net_oagh(
                    net, device, train_loader, optimizer, grad_scaler, loss_f,
                    batch_size, harmonizer, use_physics=use_physics
                )
                mode_tag = 'OAGH'
            else:
                train_total, train_data, train_physics = train_net(
                    net, device, train_loader, optimizer, grad_scaler, loss_f,
                    batch_size, use_physics=use_physics
                )
                oagh_diag = None
                mode_tag = 'warm'

            val_total, val_data, val_physics = val_net(
                net, device, val_loader, loss_f, batch_size, use_physics=use_physics
            )
            scheduler.step()
            epoch_elapsed = time.time() - epoch_start

            # TensorBoard logging
            writer.add_scalar('Loss/Train_Total', train_total, global_epoch)
            writer.add_scalar('Loss/Train_Data', train_data, global_epoch)
            writer.add_scalar('Loss/Train_Physics', train_physics, global_epoch)
            writer.add_scalar('Loss/Val_Total', val_total, global_epoch)
            writer.add_scalar('Loss/Val_Data', val_data, global_epoch)
            writer.add_scalar('Loss/Val_Physics', val_physics, global_epoch)
            writer.add_scalar('Curriculum/Train_SNR_dB', snr_db, global_epoch)
            writer.add_scalar('Curriculum/Val_SNR_dB', val_snr_db, global_epoch)
            writer.add_scalar('Curriculum/Stage', stage_idx, global_epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_epoch)

            if oagh_diag is not None:
                writer.add_scalar('OAGH/alpha', oagh_diag['alpha'], global_epoch)
                writer.add_scalar('OAGH/f_alpha', oagh_diag['f_alpha'], global_epoch)
                writer.add_scalar('OAGH/alpha_conflict', oagh_diag['alpha_conflict'], global_epoch)
                writer.add_scalar('OAGH/alpha_drift', oagh_diag['alpha_drift'], global_epoch)

            # Per-epoch progress line (always printed — helps PACE log inspection)
            print(f"[ep {global_epoch:3d}/{args.epochs} st{stage_idx+1} "
                  f"trSNR={snr_db:>2d} vSNR={val_snr_db:>4.1f} {mode_tag}] "
                  f"train={train_total:.3e} "
                  f"val={val_total:.3e} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} "
                  f"t={epoch_elapsed:.1f}s",
                  flush=True)

            # Update per-stage best (diagnostic)
            if val_total > 0 and val_total < stage_best:
                stage_best = val_total

            # Save best-by-fixed-val-SNR checkpoint
            if val_total > 0 and val_total < best_loss:
                best_loss = val_total
                best_epoch = global_epoch + 1
                best_stage = stage_idx
                torch.save({
                    'epoch': global_epoch + 1,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_total_loss': train_total,
                    'val_total_loss': val_total,
                    'best_loss': best_loss,
                    'curriculum_stage': stage_idx,
                    'curriculum_train_snr': snr_db,
                    'curriculum_val_snr': val_snr_db,
                    'harmonization': args.harmonization,
                    'run_id': run_id,
                }, os.path.join(dir_model, "weights.pth"))
                print(f"  -> new best @ epoch {best_epoch} (stage {stage_idx+1}): "
                      f"val={best_loss:.3e}", flush=True)

            # Save last-epoch snapshot (separate file, not overwriting best)
            if global_epoch == args.epochs - 1:
                torch.save({
                    'epoch': global_epoch + 1,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_total_loss': train_total,
                    'val_total_loss': val_total,
                    'best_loss': best_loss,
                    'curriculum_stage': stage_idx,
                    'curriculum_train_snr': snr_db,
                    'curriculum_val_snr': val_snr_db,
                    'harmonization': args.harmonization,
                    'run_id': run_id,
                }, os.path.join(dir_model, "weights_final.pth"))
                # Fallback: if best_loss never got a positive value, promote
                # the final snapshot to the best slot so eval has something
                # to load.
                if best_loss == float('inf'):
                    import shutil
                    shutil.copyfile(
                        os.path.join(dir_model, "weights_final.pth"),
                        os.path.join(dir_model, "weights.pth"),
                    )
                    print("WARNING: val_loss never positive — "
                          "promoted weights_final.pth to weights.pth.",
                          flush=True)

            global_epoch += 1

        stage_elapsed = time.time() - stage_time_start
        print(f"[curriculum] Stage {stage_idx+1}/{n_stages} done in {stage_elapsed:.0f}s. "
              f"Stage best val={stage_best:.3e} (at fixed val SNR={val_snr_db}).",
              flush=True)

    writer.close()
    elapsed = time.time() - time_start
    print(f"\n{'='*60}")
    print(f"Curriculum training complete.")
    print(f"  Total epochs: {global_epoch}")
    print(f"  Wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Best val loss: {best_loss:.6e} (epoch {best_epoch}, stage {best_stage+1})")
    print(f"  Best checkpoint: {os.path.join(dir_model, 'weights.pth')}")
    print(f"  Final checkpoint: {os.path.join(dir_model, 'weights_final.pth')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()

    # Identify which R2 option(s) are active
    active = []
    if args.curriculum:
        active.append('curriculum')

    if not active:
        print("WARNING: No R2 options specified. Running identical to main_train.py.")
        print("  Use --curriculum to enable curriculum learning.")

    print(f"\n{'#'*60}")
    print(f"  OAGH Revision 2 Training")
    print(f"  Active options: {', '.join(active) if active else 'NONE'}")
    print(f"{'#'*60}\n")

    # Route to the appropriate handler
    if args.curriculum:
        run_curriculum(args)
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
