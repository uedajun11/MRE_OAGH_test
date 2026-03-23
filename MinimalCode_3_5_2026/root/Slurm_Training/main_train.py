# +
import argparse
import sys
import os
import json

# ---- Robust PYTHONPATH setup for apptainer/srun ----
# Method 1: Use PYTHONPATH env var (set in sbatch via --env or APPTAINERENV_)
# Method 2: Use __file__ relative path (works when __file__ resolves correctly)
# Method 3: Hardcoded fallback for PACE
_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
_parent_dir = os.path.abspath(os.path.join(_script_dir, '..'))

# Also try the known PACE path as fallback
_pace_root = "/storage/project/r-jueda3-0/jueda3/MRE_OAGH_test/MinimalCode_3_5_2026/root"

for _p in [_parent_dir, _pace_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Debug: print path info so we can diagnose import failures
print(f"[main_train.py] __file__ = {__file__}")
print(f"[main_train.py] script_dir = {_script_dir}")
print(f"[main_train.py] parent_dir = {_parent_dir}")
print(f"[main_train.py] sys.path[:5] = {sys.path[:5]}")
print(f"[main_train.py] PYTHONPATH env = {os.environ.get('PYTHONPATH', 'NOT SET')}")

# Check if key files exist
for _check in ['train_functions.py', 'architectures/UpdatedNetwork.py', 'architectures/__init__.py']:
    _full = os.path.join(_parent_dir, _check)
    print(f"[main_train.py] {_check} exists at parent: {os.path.exists(_full)}")
    _full2 = os.path.join(_pace_root, _check)
    print(f"[main_train.py] {_check} exists at PACE root: {os.path.exists(_full2)}")

from train_functions import setup_and_run_train


# +
def parse_args():
    parser = argparse.ArgumentParser(description="Run training with specified parameters.")

    parser.add_argument('--file', type=str, default="myfile.txt", help='Path to input file')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--root', type=str, default='/path/to/root', help='Root directory path')
    parser.add_argument('--train-input', type=str, required=True, help='Path to training input data')
    parser.add_argument('--validation-input', type=str, required=True, help='Path to validation input data')
    parser.add_argument('--model', type=str, required=True, help='Path to save/load model weights')
    parser.add_argument('--offsets', type=int, default=8, help='Offsets value')
    parser.add_argument('--field-of-view', type=float, default=0.2, help='Field of view value')
    parser.add_argument('--arch-type', type=str, required=True, help='Architecture to train')
    parser.add_argument('--arch-subtype', type=str, default=None, help='Architecture Subtype to train. Default is None')
    
    # Physics loss options
    parser.add_argument('--loss-type', type=str, default='mse', choices=['mse', 'residual', 'ratio'], help='Loss function: mse, residual (hom/het), or ratio')
    
#     # Physics parameters as separate args
#     parser.add_argument('--density', type=float, default=1000.0, help='Density for physics loss')
#     parser.add_argument('--k-filter', type=float, default=1000.0, help='k_filter for physics loss')
#     parser.add_argument('--epsilon', type=float, default=1e-10, help='epsilon for physics loss')
#     parser.add_argument('--verbose', type=lambda x: (str(x).lower() == 'true'), default=False, help='Verbose flag')
#     parser.add_argument('--lambda-hom', type=float, default=1.0)
#     parser.add_argument('--lambda-data', type=float, default=1.0)


    parser.add_argument('--lambda-data', type=float, default=1.0,
                        help='Weight for MSE data loss')
    parser.add_argument('--lambda-physics', type=float, default=1.0,
                        help='Weight for physics loss (residual or ratio)')
    # Loss-specific options
    parser.add_argument('--heterogeneous', action='store_true', default=False,
                        help='For residual loss: use heterogeneous term')
    parser.add_argument('--orthogonalize-het', action='store_true', default=False,
                        help='For ratio loss: orthogonalize T_het against R_hom')
    parser.add_argument('--mse-in-k-space', action='store_true', default=False,
                        help='Compute MSE in k-space instead of mu-space')

    # OAGH / OAGH-C gradient harmonization
    parser.add_argument('--harmonization', type=str, default='none',
                        choices=['none', 'oagh', 'oagh_c'],
                        help='Gradient harmonization method: none (fixed lambda), '
                             'oagh (discrete 3-tier), or oagh_c (continuous bridge). '
                             'Requires --loss-type residual or ratio.')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='Number of standard training epochs before OAGH kicks in. '
                             'During warm-up, uses combined loss with fixed lambdas '
                             'to bring k_pred into physical range. Only used when '
                             '--harmonization is not none. Default: 10')

    return parser.parse_args()

# +
if __name__ == "__main__":
    args = parse_args()
    
    if args.arch_subtype in ["None", "none", "null"]:
        arch_subtype = None
        
    fov_str = str(args.field_of_view)
    if ',' in fov_str:
        fov_parts = fov_str.split(',')
        fov = (float(fov_parts[0]), float(fov_parts[1]))
    else:
        # Single value - duplicate it
        fov_val = float(args.field_of_view)
        fov = (fov_val, fov_val)
    
#     physics_params = None
#     if args.loss_type == 'hom':
#         physics_params = {
#         'density': args.density,
#         'fov': (args.field_of_view, args.field_of_view),
#         'residual_type': args.residual_type,
#         'k_filter': args.k_filter,
#         'epsilon': args.epsilon,
#         'verbose': args.verbose
#     }
#     else:
#         physics_params = None
    
#     print("Physics params:", physics_params)
#     print("lambda_hom:", args.lambda_hom)
#     print("lambda_data:", args.lambda_data)
    
    # Example: call your function here
    setup_and_run_train(
        train_input=args.train_input,
        val_input=args.validation_input,
        dir_model=args.model,
        offsets=args.offsets,
        fov=fov,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        arch_type=args.arch_type,   # or use another value or argument
        arch_subtype=arch_subtype,   # optional, adjust as needed
        loss_type=args.loss_type,
        lambda_data=args.lambda_data,
        lambda_physics=args.lambda_physics,
        heterogeneous=args.heterogeneous,
        orthogonalize_het=args.orthogonalize_het,
        mse_in_k_space=args.mse_in_k_space,
        harmonization=args.harmonization,
        warmup_epochs=args.warmup_epochs
    )
