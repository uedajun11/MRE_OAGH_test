"""
OAGH and OAGH-C Gradient Harmonization for Multi-Task MRE Inversion

Implements:
  - OAGH (Orthogonal Adaptive Gradient Harmonization): 3-tier discrete framework
  - OAGH-C (Continuous): Smooth bridge function replacing discrete tier switching

Reference:
  Ueda & Nieves, "Orthogonal Adaptive Gradient Harmonization for
  Physics-Informed MRE Inversion", Georgia Institute of Technology.

Authors: Jun Ueda and Heriberto Nieves
         George W. Woodruff School of Mechanical Engineering
         Georgia Institute of Technology
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional


# ═══════════════════════════════════════════════════════════════
# Alpha Diagnostic Signal
# ═══════════════════════════════════════════════════════════════

def compute_alpha_conflict(grads: List[torch.Tensor]) -> float:
    """
    Compute directional conflict score α_conflict.

    α_conflict = (1 - min_{i≠j} cos(g_i, g_j)) / 2

    Mapped to [0, 1]:
      - cos = +1 (perfectly aligned) → α_conflict = 0
      - cos =  0 (orthogonal)        → α_conflict = 0.5
      - cos = -1 (opposing)           → α_conflict = 1.0

    Args:
        grads: List of T flattened gradient vectors, each shape (D,)

    Returns:
        α_conflict in [0, 1]
    """
    T = len(grads)
    if T < 2:
        return 0.0

    min_cos = float('inf')
    for i in range(T):
        for j in range(i + 1, T):
            norm_i = torch.norm(grads[i])
            norm_j = torch.norm(grads[j])
            if norm_i < 1e-12 or norm_j < 1e-12:
                continue
            cos_ij = torch.dot(grads[i], grads[j]) / (norm_i * norm_j)
            cos_ij = cos_ij.item()
            if cos_ij < min_cos:
                min_cos = cos_ij

    if min_cos == float('inf'):
        return 0.0

    # Map from [-1, 1] to [0, 1]
    alpha_conflict = (1.0 - min_cos) / 2.0
    return alpha_conflict


def compute_alpha_drift(grads: List[torch.Tensor]) -> float:
    """
    Compute magnitude drift score α_drift.

    α_drift = 1 - 1 / max_{i≠j} max(||g_i||/||g_j||, ||g_j||/||g_i||)

    Mapped to [0, 1):
      - Equal magnitudes → ratio = 1 → α_drift = 0
      - 10× imbalance   → ratio = 10 → α_drift = 0.9
      - Infinite ratio   → α_drift → 1.0

    Args:
        grads: List of T flattened gradient vectors, each shape (D,)

    Returns:
        α_drift in [0, 1)
    """
    T = len(grads)
    if T < 2:
        return 0.0

    norms = [torch.norm(g).item() for g in grads]
    max_ratio = 1.0

    for i in range(T):
        for j in range(i + 1, T):
            if norms[i] < 1e-12 or norms[j] < 1e-12:
                continue
            ratio = max(norms[i] / norms[j], norms[j] / norms[i])
            if ratio > max_ratio:
                max_ratio = ratio

    alpha_drift = 1.0 - 1.0 / max_ratio
    return alpha_drift


def compute_alpha(grads: List[torch.Tensor]) -> Tuple[float, float, float]:
    """
    Compute combined diagnostic score α = max(α_conflict, α_drift).

    Args:
        grads: List of T flattened gradient vectors

    Returns:
        (alpha, alpha_conflict, alpha_drift)
    """
    alpha_conflict = compute_alpha_conflict(grads)
    alpha_drift = compute_alpha_drift(grads)
    alpha = max(alpha_conflict, alpha_drift)
    return alpha, alpha_conflict, alpha_drift


# ═══════════════════════════════════════════════════════════════
# OAGH-C Bridge Function
# ═══════════════════════════════════════════════════════════════

def bridge_function(alpha: float) -> float:
    """
    OAGH-C continuous bridge function f(α).

    f(α) = 0                      if α < 0.2   (Tier I region)
    f(α) = (α - 0.2) / 0.4       if 0.2 ≤ α < 0.6   (bridging region)
    f(α) = 1.0                    if α ≥ 0.6   (Tier III region)

    Properties:
      - Continuous everywhere
      - Monotonically non-decreasing
      - f(0.2) = 0, f(0.6) = 1 (boundary equivalence)
    """
    if alpha < 0.2:
        return 0.0
    elif alpha < 0.6:
        return (alpha - 0.2) / 0.4
    else:
        return 1.0


def get_tier(alpha: float) -> str:
    """Return human-readable tier name for OAGH discrete."""
    if alpha < 0.2:
        return "I (GradNorm)"
    elif alpha < 0.6:
        return "II (CAGrad)"
    else:
        return "III (PCGrad)"


# ═══════════════════════════════════════════════════════════════
# Harmonization Methods
# ═══════════════════════════════════════════════════════════════

def harmonize_pcgrad(grads: List[torch.Tensor]) -> torch.Tensor:
    """
    PCGrad: Full orthogonal projection (Tier III).

    For each conflicting pair (g_i · g_j < 0):
      g_i' = g_i - (g_i · g_j / ||g_j||²) · g_j

    Then average: g* = (1/T) Σ g_i'

    Args:
        grads: List of T gradient vectors, each shape (D,)

    Returns:
        g_harmonized: (D,) combined gradient
    """
    T = len(grads)
    gs = [g.clone() for g in grads]

    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            dot_ij = torch.dot(gs[i], grads[j])
            if dot_ij < 0:
                norm_j_sq = torch.dot(grads[j], grads[j])
                gs[i] = gs[i] - (dot_ij / (norm_j_sq + 1e-12)) * grads[j]

    g_harmonized = sum(gs) / T
    return g_harmonized


def harmonize_gradnorm(grads: List[torch.Tensor]) -> torch.Tensor:
    """
    GradNorm-style magnitude rebalancing (Tier I).

    Scales each gradient to have the average magnitude:
      g_i' = g_i × (||g||_avg / ||g_i||)

    Then averages: g* = (1/T) Σ g_i'

    Args:
        grads: List of T gradient vectors, each shape (D,)

    Returns:
        g_harmonized: (D,) combined gradient
    """
    T = len(grads)
    norms = [torch.norm(g) for g in grads]
    avg_norm = sum(norms) / T

    gs = []
    for i in range(T):
        if norms[i] > 1e-12:
            gs.append(grads[i] * (avg_norm / norms[i]))
        else:
            gs.append(grads[i])

    g_harmonized = sum(gs) / T
    return g_harmonized


def harmonize_oagh_c(grads: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
    """
    OAGH-C: Continuous Soft-Projection Harmonization.

    Algorithm:
      1. Compute α = max(α_conflict, α_drift)
      2. Compute f(α) via bridge function
      3. For each conflicting pair (g_i · g_j < 0):
           g_i' = g_i - f(α) · (g_i · g_j / ||g_j||²) · g_j
      4. Magnitude restoration: ||g_i'|| → ||g_i||
      5. Average: g* = (1/T) Σ g_i'

    Args:
        grads: List of T gradient vectors, each shape (D,)

    Returns:
        g_harmonized: (D,) combined gradient
        diagnostics: dict with alpha, f_alpha, tier, per-task norms
    """
    T = len(grads)
    alpha, alpha_conflict, alpha_drift = compute_alpha(grads)
    f_alpha = bridge_function(alpha)

    # Store original norms for magnitude restoration
    orig_norms = [torch.norm(g) for g in grads]

    # Fractional projection
    gs = [g.clone() for g in grads]

    if f_alpha > 0:
        for i in range(T):
            for j in range(T):
                if i == j:
                    continue
                dot_ij = torch.dot(gs[i], grads[j])
                if dot_ij < 0:  # conflicting pair
                    norm_j_sq = torch.dot(grads[j], grads[j])
                    gs[i] = gs[i] - f_alpha * (dot_ij / (norm_j_sq + 1e-12)) * grads[j]

        # Magnitude restoration
        for i in range(T):
            proj_norm = torch.norm(gs[i])
            if proj_norm > 1e-12 and orig_norms[i] > 1e-12:
                gs[i] = gs[i] * (orig_norms[i] / proj_norm)
    else:
        # f(α) = 0: pure magnitude rebalancing (Tier I behavior)
        avg_norm = sum(orig_norms) / T
        for i in range(T):
            if orig_norms[i] > 1e-12:
                gs[i] = grads[i] * (avg_norm / orig_norms[i])

    # Average
    g_harmonized = sum(gs) / T

    diagnostics = {
        'alpha': alpha,
        'alpha_conflict': alpha_conflict,
        'alpha_drift': alpha_drift,
        'f_alpha': f_alpha,
        'tier': get_tier(alpha),
        'task_norms': [n.item() for n in orig_norms],
    }

    return g_harmonized, diagnostics


def harmonize_oagh(grads: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
    """
    OAGH: Discrete Three-Tier Harmonization.

    Tier I  (α < 0.2):  GradNorm magnitude rebalancing
    Tier II (0.2 ≤ α < 0.6): CAGrad (approximated via moderate projection)
    Tier III (α ≥ 0.6): PCGrad full orthogonal projection

    Args:
        grads: List of T gradient vectors, each shape (D,)

    Returns:
        g_harmonized: (D,) combined gradient
        diagnostics: dict
    """
    alpha, alpha_conflict, alpha_drift = compute_alpha(grads)

    if alpha < 0.2:
        # Tier I: GradNorm
        g_harmonized = harmonize_gradnorm(grads)
        tier = "I (GradNorm)"
    elif alpha < 0.6:
        # Tier II: CAGrad approximation via half-strength projection
        # Use f = 0.5 as fixed moderate projection
        T = len(grads)
        orig_norms = [torch.norm(g) for g in grads]
        gs = [g.clone() for g in grads]

        for i in range(T):
            for j in range(T):
                if i == j:
                    continue
                dot_ij = torch.dot(gs[i], grads[j])
                if dot_ij < 0:
                    norm_j_sq = torch.dot(grads[j], grads[j])
                    gs[i] = gs[i] - 0.5 * (dot_ij / (norm_j_sq + 1e-12)) * grads[j]

        # Magnitude restoration
        for i in range(T):
            proj_norm = torch.norm(gs[i])
            if proj_norm > 1e-12 and orig_norms[i] > 1e-12:
                gs[i] = gs[i] * (orig_norms[i] / proj_norm)

        g_harmonized = sum(gs) / T
        tier = "II (CAGrad)"
    else:
        # Tier III: PCGrad
        g_harmonized = harmonize_pcgrad(grads)
        tier = "III (PCGrad)"

    diagnostics = {
        'alpha': alpha,
        'alpha_conflict': alpha_conflict,
        'alpha_drift': alpha_drift,
        'f_alpha': bridge_function(alpha),
        'tier': tier,
        'task_norms': [torch.norm(g).item() for g in grads],
    }

    return g_harmonized, diagnostics


# ═══════════════════════════════════════════════════════════════
# Main Harmonizer Class
# ═══════════════════════════════════════════════════════════════

class OAGHHarmonizer:
    """
    Unified interface for OAGH/OAGH-C gradient harmonization.

    Usage:
        harmonizer = OAGHHarmonizer(method='oagh_c')

        # In training loop:
        g_data = extract_flat_grad(params)   # after L_data.backward()
        g_phys = extract_flat_grad(params)   # after L_physics.backward()

        g_combined, diag = harmonizer([g_data, g_phys])
        set_flat_grad(params, g_combined)
        optimizer.step()

    Args:
        method: 'oagh_c' (continuous, recommended), 'oagh' (discrete 3-tier),
                'pcgrad', or 'gradnorm'
    """

    def __init__(self, method: str = 'oagh_c'):
        assert method in ('oagh_c', 'oagh', 'pcgrad', 'gradnorm'), \
            f"Unknown method: {method}. Choose from: oagh_c, oagh, pcgrad, gradnorm"
        self.method = method
        self._last_diagnostics = None

    def __call__(self, grads: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Harmonize per-task gradients.

        Gradients are normalized to unit norm before harmonization so that
        directional conflict (not magnitude imbalance) drives tier selection
        and projection.  The harmonized direction is then rescaled by the
        geometric mean of the original norms.

        Args:
            grads: List of T flattened gradient vectors, each shape (D,)

        Returns:
            g_harmonized: (D,) combined gradient
            diagnostics: dict with alpha, f_alpha, tier, task_norms
        """
        # --- Pre-normalize to decouple direction from magnitude ---
        orig_norms = [torch.norm(g) for g in grads]
        # Geometric mean preserves scale without letting one task dominate
        log_norms = [torch.log(n + 1e-12) for n in orig_norms]
        scale = torch.exp(sum(log_norms) / len(log_norms))

        grads_normed = []
        for g, n in zip(grads, orig_norms):
            if n > 1e-12:
                grads_normed.append(g / n)
            else:
                grads_normed.append(g)

        # --- Harmonize on unit-norm gradients ---
        if self.method == 'oagh_c':
            g, diag = harmonize_oagh_c(grads_normed)
        elif self.method == 'oagh':
            g, diag = harmonize_oagh(grads_normed)
        elif self.method == 'pcgrad':
            g = harmonize_pcgrad(grads_normed)
            alpha, ac, ad = compute_alpha(grads_normed)
            diag = {
                'alpha': alpha, 'alpha_conflict': ac, 'alpha_drift': ad,
                'f_alpha': 1.0, 'tier': 'PCGrad (fixed)',
                'task_norms': [n.item() for n in orig_norms],
            }
        elif self.method == 'gradnorm':
            g = harmonize_gradnorm(grads_normed)
            alpha, ac, ad = compute_alpha(grads_normed)
            diag = {
                'alpha': alpha, 'alpha_conflict': ac, 'alpha_drift': ad,
                'f_alpha': 0.0, 'tier': 'GradNorm (fixed)',
                'task_norms': [n.item() for n in orig_norms],
            }

        # --- Rescale harmonized gradient ---
        g = g * scale

        # Record original (pre-normalization) norms for diagnostics
        if 'task_norms' not in diag:
            diag['task_norms'] = [n.item() for n in orig_norms]
        diag['task_norms_raw'] = [n.item() for n in orig_norms]
        diag['scale'] = scale.item()

        self._last_diagnostics = diag
        return g, diag

    @property
    def last_diagnostics(self) -> Optional[Dict]:
        return self._last_diagnostics


# ═══════════════════════════════════════════════════════════════
# Gradient Extraction / Setting Utilities
# ═══════════════════════════════════════════════════════════════

def extract_flat_grad(params) -> torch.Tensor:
    """
    Extract all parameter gradients into a single flattened vector.

    Args:
        params: iterable of nn.Parameter (e.g., net.parameters())

    Returns:
        g_flat: (D,) tensor where D = total number of parameters
    """
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(grads)


def set_flat_grad(params, g_flat: torch.Tensor):
    """
    Set parameter gradients from a single flattened vector.

    Args:
        params: iterable of nn.Parameter
        g_flat: (D,) tensor
    """
    offset = 0
    for p in params:
        numel = p.numel()
        if p.grad is None:
            p.grad = g_flat[offset:offset + numel].view_as(p).clone()
        else:
            p.grad.copy_(g_flat[offset:offset + numel].view_as(p))
        offset += numel
