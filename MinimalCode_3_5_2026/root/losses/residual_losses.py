# residual_losses.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from losses.homogeneous import MREHelmholtzLoss
from losses.ratio_loss import gradient_2d_batch, compute_laplacian_batch

def compute_residual(wave_tensors, mu_pred, mfre, fov=(0.2,0.2),
                     W=1.0, eps_val=1e-8, heterogeneous=False, viz=False):
    """
    Compute raw residual for homogeneous or heterogeneous physics using spatial k^2.

    Args:
        wave_tensors: (B, C, H, W, T) complex tensor or pre-extracted fundamental
        mu_pred: (B,1,H,W) predicted shear modulus [Pa]
        mfre: (B,) tensor, frequency in Hz
        fov: (Fx, Fy) field of view in meters
        W: scalar or (H, W) weight map
        eps_val: small epsilon to avoid div by zero
        heterogeneous: if True, include ∇log(μ)·∇u term
        viz: if True, visualize residuals

    Returns:
        R: (B,H,W) complex residual per slice
        loss_val: scalar per batch (mean squared weighted residual)
    """
    device = wave_tensors.device
    B, _, H, W_dim, T = wave_tensors.shape
    rho = 1000

    hom = MREHelmholtzLoss(density=rho, fov=fov, residual_type='raw', k_filter=1000, epsilon=1e-10)

    # weight map
    W_map = torch.ones((B,H,W_dim), device=device) * W if isinstance(W, (int,float)) else W.to(device)

    # fundamental harmonic
    wave_H = hom.extract_fundamental_frequency_batch(wave_tensors)

    # Laplacian
    lap_u = compute_laplacian_batch(wave_H, fov)

    # normalize weight
    W_norm = W_map / (W_map.view(B,-1).sum(dim=1).view(B,1,1) + eps_val)

    # spatial k^2 from mu_pred
    omega = 2 * torch.pi * mfre.view(B,1,1)
    k2_data = (rho * omega**2 / torch.clamp(mu_pred[:,0], min=eps_val))

    #  homogeneous residual
    R_hom = lap_u + k2_data * wave_H

    if heterogeneous:
        dx, dy = fov[0]/W_dim, fov[1]/H
        log_mu = torch.log(torch.clamp(mu_pred[:,0,:,:], min=eps_val))
        gx_mu, gy_mu = gradient_2d_batch(log_mu, dy, dx)
        gx_u, gy_u   = gradient_2d_batch(wave_H, dy, dx)
        T_het = gx_mu * gx_u + gy_mu * gy_u
        R_het = R_hom + T_het
    else:
        R_het = R_hom
        T_het = torch.zeros_like(R_hom)

    # weighted mean squared loss
    loss_val = torch.sum(W_norm * R_het.abs()**2, dim=(1,2)).mean()
    diag = {
        'wave_H':wave_H,
        'lap_u':lap_u,
        'R_hom':R_hom,
        'k2_data':k2_data,
        'T_het':T_het
    }
    # visualization
    if viz:
        for b in range(B):
            ub = wave_H[b].squeeze()        # (H, W)
            Rb_hom = R_hom[b].squeeze()
            Rb_het = R_het[b].squeeze()
            k2b = k2_data[b].squeeze()
            lap_b = lap_u[b].squeeze()
            T_het_b = T_het[b].squeeze()

            fig, axs = plt.subplots(2, 4, figsize=(16, 8))

            # Top row: amplitude, phase, Laplacian, homogeneous residual
            axs[0,0].imshow(ub.abs().cpu()); axs[0,0].set_title("|u|"); fig.colorbar(axs[0,0].images[0], ax=axs[0,0])
            axs[0,1].imshow(ub.angle().cpu()); axs[0,1].set_title("Phase"); fig.colorbar(axs[0,1].images[0], ax=axs[0,1])
            axs[0,2].imshow(lap_b.abs().cpu()); axs[0,2].set_title("|∇²u|"); fig.colorbar(axs[0,2].images[0], ax=axs[0,2])
            axs[0,3].imshow(Rb_hom.abs().cpu()); axs[0,3].set_title("|R_hom|"); fig.colorbar(axs[0,3].images[0], ax=axs[0,3])

            # Bottom row: heterogeneous terms if requested
            if heterogeneous:
                axs[1,0].imshow(T_het_b.abs().cpu()); axs[1,0].set_title("|∇log(μ)·∇u|"); fig.colorbar(axs[1,0].images[0], ax=axs[1,0])
                axs[1,1].imshow(Rb_het.abs().cpu()); axs[1,1].set_title("|R_het|"); fig.colorbar(axs[1,1].images[0], ax=axs[1,1])
                axs[1,2].imshow((Rb_het.abs() / (Rb_hom.abs() + eps_val)).cpu()); axs[1,2].set_title("Ratio Residual Map"); fig.colorbar(axs[1,2].images[0], ax=axs[1,2])
                axs[1,3].imshow(torch.log10((Rb_het.abs() / (Rb_hom.abs() + eps_val) + eps_val)).cpu()); axs[1,3].set_title("Log Ratio"); fig.colorbar(axs[1,3].images[0], ax=axs[1,3])

            plt.tight_layout()
            plt.show()

    return R_het, loss_val, diag


class ResidualLoss(nn.Module):
    """
    Raw Helmholtz residual loss for MRE.

    Computes mean squared weighted residual energy using spatial k^2.
    Can operate in homogeneous or heterogeneous mode.
    """

    def __init__(self, fov=(0.2, 0.2), rho=1000, heterogeneous=False,
                 viz=False, diagnostics=False):
        super().__init__()
        self.fov = fov
        self.rho = rho
        self.heterogeneous = heterogeneous
        self.viz = viz
        self.diagnostics = diagnostics

    def forward(self, wave_tensors, mu_pred, mfre, W=1.0, k_pred=None):
        """
        Args:
            wave_tensors: (B, C, H, W, T) complex wave field
            mu_pred: (B, 1, H, W) predicted stiffness [Pa]  (used only if k_pred is None)
            mfre: (B,) or scalar, frequency [Hz]
            W: weight map
            k_pred: (B, 1, H, W) predicted wave number [rad/m] (optional).
                     When provided, k^2 and log(mu) are derived directly from k_pred,
                     bypassing the mu clamp and preserving gradient flow.
        """

        eps_val = 1e-8
        device = wave_tensors.device
        B, _, H, W_dim, T = wave_tensors.shape

        hom = MREHelmholtzLoss(
            density=self.rho,
            fov=self.fov,
            residual_type='raw',
            k_filter=1000,
            epsilon=1e-10,
            verbose=False
        )

        # --- weight map ---
        if isinstance(W, (int, float)):
            W_map = torch.ones((B, H, W_dim), device=device) * W
        else:
            W_map = W.to(device)

        # --- extract harmonic ---
        wave_H = hom.extract_fundamental_frequency_batch(wave_tensors)

        # --- Laplacian ---
        lap_u = compute_laplacian_batch(wave_H, self.fov)

        # --- normalize weights ---
        W_norm = W_map / (W_map.view(B, -1).sum(dim=1).view(B, 1, 1) + eps_val)

        omega = 2 * torch.pi * mfre.view(B, 1, 1)

        if k_pred is not None:
            # --- k-space residual: derive k^2 directly from k_pred ---
            # This avoids the k -> mu (clamped) -> k^2 round-trip
            k_safe = torch.clamp(k_pred[:, 0], min=0.1)  # (B, H, W)
            k2_data = k_safe ** 2
        else:
            # --- legacy: spatial k^2 from mu ---
            k2_data = (self.rho * omega**2 /
                       torch.clamp(mu_pred[:, 0], min=eps_val))

        # --- homogeneous residual ---
        R_hom = lap_u + k2_data * wave_H

        # --- heterogeneous term ---
        if self.heterogeneous:
            dx, dy = self.fov[0]/W_dim, self.fov[1]/H
            if k_pred is not None:
                # log(mu) = log(rho*omega^2) - 2*log(k)
                # Gradient of log(mu) = -2 * gradient(log(k))
                log_k = torch.log(k_safe)
                gx_logk, gy_logk = gradient_2d_batch(log_k, dy, dx)
                gx_mu = -2.0 * gx_logk
                gy_mu = -2.0 * gy_logk
            else:
                log_mu = torch.log(torch.clamp(mu_pred[:,0], min=eps_val))
                gx_mu, gy_mu = gradient_2d_batch(log_mu, dy, dx)
            gx_u, gy_u   = gradient_2d_batch(wave_H, dy, dx)
            T_het = gx_mu * gx_u + gy_mu * gy_u
            R_out = R_hom + T_het
        else:
            T_het = torch.zeros_like(R_hom)
            R_out = R_hom

        # --- weighted mean squared residual ---
        loss_val = torch.sum(W_norm * R_out.abs()**2, dim=(1,2)).mean()

        # --- diagnostics ---
        if self.diagnostics:
            diag = {
                'wave_H': wave_H,
                'lap_u': lap_u,
                'R_hom': R_hom,
                'R_out': R_out,
                'T_het': T_het,
                'k2_data': k2_data
            }
        else:
            diag = None

        # --- visualization ---
        if self.viz:
            for b in range(B):
                ub = wave_H[b]
                Rb_hom = R_hom[b]
                Rb_out = R_out[b]
                lap_b = lap_u[b]
                T_het_b = T_het[b]

                fig, axs = plt.subplots(2, 4, figsize=(16, 8))

                axs[0,0].imshow(ub.abs().cpu()); axs[0,0].set_title("|u|")
                fig.colorbar(axs[0,0].images[0], ax=axs[0,0])

                axs[0,1].imshow(ub.angle().cpu()); axs[0,1].set_title("Phase")
                fig.colorbar(axs[0,1].images[0], ax=axs[0,1])

                axs[0,2].imshow(lap_b.abs().cpu()); axs[0,2].set_title("|∇²u|")
                fig.colorbar(axs[0,2].images[0], ax=axs[0,2])

                axs[0,3].imshow(Rb_hom.abs().cpu()); axs[0,3].set_title("|R_hom|")
                fig.colorbar(axs[0,3].images[0], ax=axs[0,3])

                if self.heterogeneous:
                    axs[1,0].imshow(T_het_b.abs().cpu())
                    axs[1,0].set_title("|∇log(μ)·∇u|")
                    fig.colorbar(axs[1,0].images[0], ax=axs[1,0])

                    axs[1,1].imshow(Rb_out.abs().cpu())
                    axs[1,1].set_title("|R_het|")
                    fig.colorbar(axs[1,1].images[0], ax=axs[1,1])

                    axs[1,2].imshow((Rb_out.abs()/(Rb_hom.abs()+eps_val)).cpu())
                    axs[1,2].set_title("Ratio Map")
                    fig.colorbar(axs[1,2].images[0], ax=axs[1,2])

                    axs[1,3].imshow(torch.log10(
                        (Rb_out.abs()/(Rb_hom.abs()+eps_val)+eps_val)).cpu())
                    axs[1,3].set_title("Log Ratio")
                    fig.colorbar(axs[1,3].images[0], ax=axs[1,3])

                plt.tight_layout()
                plt.show()

        if self.diagnostics:
            return loss_val, diag
        return loss_val


class CombinedResidualLoss(nn.Module):
    """
    Combines MSE data loss with Residual physics loss (homogeneous or heterogeneous).
    Returns: total_loss, data_loss, physics_loss
    """
    
    def __init__(self, fov=(0.2, 0.2), rho=1000, 
                 lambda_data=1.0, lambda_physics=1.0,
                 heterogeneous=False, mse_in_k_space=False,
                 viz=False, diagnostics=False):
        """
        Args:
            fov: field of view
            rho: density
            lambda_data: weight for MSE loss (set to 0 to disable)
            lambda_physics: weight for residual physics loss
            heterogeneous: if True, use heterogeneous residual; else homogeneous
            mse_in_k_space: if True, compute MSE in k-space; else μ-space
            viz: visualization flag
            diagnostics: return diagnostics dict
        """
        super().__init__()
        
        self.residual_loss = ResidualLoss(
            fov=fov,
            rho=rho,
            heterogeneous=heterogeneous,
            viz=viz,
            diagnostics=diagnostics
        )
        
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.mse_in_k_space = mse_in_k_space
        self.diagnostics = diagnostics
        
    def forward(self, k_pred, k_gt, mu_pred, mu_gt, wave, mfre, fov=None):
        """
        Args:
            k_pred: (B, 1, H, W) predicted wave number [rad/m]
            k_gt: (B, 1, H, W) ground truth wave number [rad/m]
            mu_pred: (B, 1, H, W) predicted stiffness [Pa]
            mu_gt: (B, 1, H, W) ground truth stiffness [Pa]
            wave: (B, C, H, W, T) complex wave field
            mfre: (B,) frequency [Hz]
            fov: field of view (optional)

        Returns:
            total_loss: weighted combination
            data_loss: MSE component
            physics_loss: residual component
        """
        eps_val = 1e-8

        # Data loss: MSE in k-space or μ-space
        if self.lambda_data > 0:
            if self.mse_in_k_space:
                data_loss = F.mse_loss(k_pred, k_gt)
            else:
                # Compare in μ space
                data_loss = F.mse_loss(mu_pred, mu_gt)
        else:
            data_loss = torch.tensor(0.0, device=k_pred.device)

        # Physics loss: Residual (hom or het)
        # Pass k_pred directly so physics loss can derive k^2 without mu clamp
        mu_pred_pa = mu_pred

        if self.diagnostics:
            physics_loss, diag = self.residual_loss(
                wave, mu_pred_pa, mfre, W=1.0, k_pred=k_pred
            )
        else:
            physics_loss = self.residual_loss(
                wave, mu_pred_pa, mfre, W=1.0, k_pred=k_pred
            )
        
        # Combined loss
        total_loss = (self.lambda_data * data_loss + 
                     self.lambda_physics * physics_loss)
        
        if self.diagnostics:
            return total_loss, data_loss, physics_loss, diag
        
        return total_loss, data_loss, physics_loss
