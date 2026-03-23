import torch
import matplotlib.pyplot as plt
from losses.homogeneous import MREHelmholtzLoss
import torch
import torch.nn as nn
from torch.nn import functional as F


def gradient_2d_batch(u, dy, dx):
    """
    Vectorized 2D MATLAB-style gradient for a batch of images.
    Args:
        u: (B, H, W) tensor
        dy, dx: pixel sizes
    Returns:
        ux, uy: (B, H, W) gradients
    """
    B, H, W = u.shape
    uy = torch.zeros_like(u)
    ux = torch.zeros_like(u)

    # central difference
    uy[:,1:-1,:] = (u[:,2:,:] - u[:,:-2,:]) / (2*dy)
    ux[:,:,1:-1] = (u[:,:,2:] - u[:,:,:-2]) / (2*dx)
    
    # forward/backward edges
    uy[:,0,:]  = (u[:,1,:] - u[:,0,:]) / dy
    uy[:,-1,:] = (u[:,-1,:] - u[:,-2,:]) / dy
    ux[:,:,0]  = (u[:,:,1] - u[:,:,0]) / dx
    ux[:,:,-1] = (u[:,:,-1] - u[:,:,-2]) / dx

    return ux, uy

def compute_laplacian_batch(u, fov):
    """
    Vectorized Laplacian for a batch of 2D images using MATLAB-style gradients
    Args:
        u: (B, H, W) complex tensor
        fov: (Fx, Fy) in meters
    Returns:
        laplacian: (B, H, W) complex tensor
    """
    B, H, W = u.shape
    dx, dy = fov[0] / W, fov[1] / H

    ux, uy = gradient_2d_batch(u, dy, dx)
    uxx, _ = gradient_2d_batch(ux, dy, dx)
    _, uyy = gradient_2d_batch(uy, dy, dx)

    laplacian = uxx + uyy
    return laplacian

def ratio_loss_batch_vectorized(wave_tensors, mu_pred, mfre, fov, rho, W=1.0, eps_val=1e-8, viz=False):
    """
    Fully vectorized ratio loss for a batch.
    """
    device = mu_pred.device
    B, _, H, W_dim, T = wave_tensors.shape
    
    hom = MREHelmholtzLoss(
        density=1000,
        fov=(0.2, 0.2),
        residual_type='raw',
        k_filter=1000,
        epsilon=1e-8,
        verbose=False
    )


    # weight map
    if isinstance(W, (int, float)):
        W_map = torch.ones((B,H,W_dim), device=device) * W
    else:
        W_map = W.to(device)

    # fundamental harmonic
    wave_H = hom.extract_fundamental_frequency_batch(wave_tensors)  # (B,H,W)

    dx, dy = fov[0]/W_dim, fov[1]/H

    # Laplacian
    lap_u = compute_laplacian_batch(wave_H, fov)

    omega = 2 * torch.pi * mfre.view(B,1,1)
    
    # normalize weight
    W_norm = W_map / (W_map.view(B,-1).sum(dim=1).view(B,1,1) + eps_val)

    # homogeneous k^2b
    k2_data = -torch.sum(W_norm * (lap_u * wave_H).real, dim=(1,2)) / \
               (torch.sum(W_norm * wave_H.abs()**2, dim=(1,2)) + eps_val)
    k2_data = torch.clamp(k2_data, min=eps_val)
    
    # homogeneous residual
    R_hom = lap_u + k2_data.view(B,1,1) * wave_H

    # heterogeneous term
    log_mu = torch.log(torch.clamp(mu_pred[:,0], min=eps_val))
    gx_mu, gy_mu = gradient_2d_batch(log_mu, dy, dx)
    gx_u, gy_u   = gradient_2d_batch(wave_H, dy, dx)
    T_het = gx_mu * gx_u + gy_mu * gy_u

    # numerator residual
    R_num = R_hom + T_het

    # ratio loss
    num = torch.sum(W_norm * R_num.abs()**2, dim=(1,2))
    den = torch.sum(W_norm * R_hom.abs()**2, dim=(1,2)) + eps_val
    L_ratio = torch.log(num + eps_val) - torch.log(den)
    L_ratio = L_ratio.mean()

    # diagnostics
    diagnostics = {
        "R_hom": R_hom,
        "R_het": R_num,
        "k2_data": k2_data,
        'lap_u': lap_u,
        "mu_effective": (rho * omega[:,:,0]**2 / k2_data) / 1e3
    }

    # visualization (optional)
    if viz:
        for b in range(B):
            ub = wave_H[b]
            fig, axs = plt.subplots(2, 4, figsize=(16, 8))

            im0 = axs[0,0].imshow(ub.abs().cpu()); axs[0,0].set_title("|u|")
            fig.colorbar(im0, ax=axs[0,0])

            im1 = axs[0,1].imshow(ub.angle().cpu()); axs[0,1].set_title("Phase")
            fig.colorbar(im1, ax=axs[0,1])

            im2 = axs[0,2].imshow(lap_u[b].abs().cpu()); axs[0,2].set_title("|∇²u|")
            fig.colorbar(im2, ax=axs[0,2])

            im3 = axs[0,3].imshow(R_hom[b].abs().cpu()); axs[0,3].set_title("|R_hom|")
            fig.colorbar(im3, ax=axs[0,3])

            im4 = axs[1,0].imshow(T_het[b].abs().cpu()); axs[1,0].set_title("|∇log(μ)·∇u|")
            fig.colorbar(im4, ax=axs[1,0])

            im5 = axs[1,1].imshow(R_num[b].abs().cpu()); axs[1,1].set_title("|R_num|")
            fig.colorbar(im5, ax=axs[1,1])

            im6 = axs[1,2].imshow((R_num[b].abs() / (R_hom[b].abs() + eps_val)).cpu())
            axs[1,2].set_title("Ratio Residual Map")
            fig.colorbar(im6, ax=axs[1,2])

            im7 = axs[1,3].imshow(torch.log10((R_num[b].abs() / (R_hom[b].abs() + eps_val) + eps_val)).cpu())
            axs[1,3].set_title("Log Ratio")
            fig.colorbar(im7, ax=axs[1,3])

            plt.tight_layout()
            plt.show()
            plt.show()

    return L_ratio, diagnostics


class RatioLoss(nn.Module):
    """
    Ratio loss for MRE physics-informed training.
    
    Computes log-ratio of heterogeneous vs homogeneous Helmholtz residuals
    using W-weighted Rayleigh quotient for k²_data estimation.
    Optionally orthogonalizes heterogeneous term against homogeneous residual.
    """
    
    def __init__(self, fov=(0.2, 0.2), rho=1000, orthogonalize_het=False, viz=False, diagnostics=False):
        """
        Args:
            fov: (Fx, Fy) field of view in meters
            rho: tissue density [kg/m³]
            Optionally orthogonalizes heterogeneous term against homogeneous residual.
        """
        super().__init__()
        self.fov = fov
        self.rho = rho
        self.orthogonalize_het = orthogonalize_het
        self.viz = viz
        self.diagnostics = diagnostics
    
    def forward(self,wave_tensors, mu_pred, mfre, W=1.0):
        """
        Compute ratio loss.
        
        Args:
            wave_tensors: (B, C, H, W, T) complex wave field
            mu_pred: (B, 1, H, W) predicted stiffness [Pa]
            mfre: (B,) or scalar, frequency [Hz]
            W: weight map, scalar or (B, H, W) or (H, W)
        
        Returns:
            loss: scalar tensor
            diagnostics: dict (if return_diagnostics=True)
        """
        eps_val=1e-8
        device = mu_pred.device
        B, _, H, W_dim, T = wave_tensors.shape

        hom = MREHelmholtzLoss(
            density=1000,
            fov=self.fov,
            residual_type='raw',
            k_filter=1000,
            epsilon=1e-8,
            verbose=False
        )

        if isinstance(mfre, torch.Tensor):
            if mfre.shape[0] != B:
                mfre = mfre[0:1].expand(B)  # Take first, expand to batch
            mfre = mfre.view(B, 1, 1)
        else:
            mfre = torch.tensor([mfre] * B, device=device).view(B, 1, 1)

        # weight map
        if isinstance(W, (int, float)):
            W_map = torch.ones((B,H,W_dim), device=device) * W
        else:
            W_map = W.to(device)

        # fundamental harmonic
        wave_H = hom.extract_fundamental_frequency_batch(wave_tensors)  # (B,H,W)

        dx, dy = self.fov[0]/W_dim, self.fov[1]/H

        # Laplacian
        lap_u = compute_laplacian_batch(wave_H, self.fov)

        omega = 2 * torch.pi * mfre.view(B,1,1)

        # normalize weight
        W_norm = W_map / (W_map.view(B,-1).sum(dim=1).view(B,1,1) + eps_val)

        # homogeneous k^2b
        k2_data = -torch.sum(W_norm * (lap_u * wave_H).real, dim=(1,2)) / \
                   (torch.sum(W_norm * wave_H.abs()**2, dim=(1,2)) + eps_val)
        k2_data = torch.clamp(k2_data, min=eps_val)

        # homogeneous residual
        R_hom = lap_u + k2_data.view(B,1,1) * wave_H

        # heterogeneous term
        log_mu = torch.log(torch.clamp(mu_pred[:,0], min=eps_val))
        gx_mu, gy_mu = gradient_2d_batch(log_mu, dy, dx)
        gx_u, gy_u   = gradient_2d_batch(wave_H, dy, dx)
        T_het = gx_mu * gx_u + gy_mu * gy_u
        
        # OPTIONAL ORTHOGONALIZATION # Keep false to match with MATLAB which doesnt have this currently
        if self.orthogonalize_het:
            proj_num = torch.sum(W_norm * T_het * R_hom.conj(), dim=(1, 2)).real
            proj_den = torch.sum(
                W_norm * (R_hom.abs() ** 2), dim=(1, 2)
            ) + eps_val

            alpha = (proj_num / proj_den).view(B, 1, 1)
            T_het = T_het - alpha * R_hom

        # numerator residual
        R_num = R_hom + T_het

        # ratio loss
        num = torch.sum(W_norm * R_num.abs()**2, dim=(1,2))
        den = torch.sum(W_norm * R_hom.abs()**2, dim=(1,2)) + eps_val
        L_ratio = torch.log(num + eps_val) - torch.log(den)
        L_ratio = L_ratio.mean()
        
        print(f"DEBUG RatioLoss:")
        print(f"  num (R_het energy): {num.mean().item():.6e}")
        print(f"  den (R_hom energy): {den.mean().item():.6e}")
        print(f"  k2_data: {k2_data.mean().item():.6f}")
        print(f"  R_hom magnitude: {R_hom.abs().mean().item():.6e}")
        print(f"  R_num magnitude: {R_num.abs().mean().item():.6e}")
        print(f"  T_het magnitude: {T_het.abs().mean().item():.6e}")
        print(f"  L_ratio: {L_ratio.item():.6e}")
        

        # visualization (optional)
        if self.viz:
            for b in range(B):
                ub = wave_H[b]
                fig, axs = plt.subplots(2, 4, figsize=(16, 8))

                im0 = axs[0,0].imshow(ub.abs().cpu()); axs[0,0].set_title("|u|")
                fig.colorbar(im0, ax=axs[0,0])

                im1 = axs[0,1].imshow(ub.angle().cpu()); axs[0,1].set_title("Phase")
                fig.colorbar(im1, ax=axs[0,1])

                im2 = axs[0,2].imshow(lap_u[b].abs().cpu()); axs[0,2].set_title("|∇²u|")
                fig.colorbar(im2, ax=axs[0,2])

                im3 = axs[0,3].imshow(R_hom[b].abs().cpu()); axs[0,3].set_title("|R_hom|")
                fig.colorbar(im3, ax=axs[0,3])

                im4 = axs[1,0].imshow(T_het[b].abs().cpu()); axs[1,0].set_title("|∇log(μ)·∇u|")
                fig.colorbar(im4, ax=axs[1,0])

                im5 = axs[1,1].imshow(R_num[b].abs().cpu()); axs[1,1].set_title("|R_num|")
                fig.colorbar(im5, ax=axs[1,1])

                im6 = axs[1,2].imshow((R_num[b].abs() / (R_hom[b].abs() + eps_val)).cpu())
                axs[1,2].set_title("Ratio Residual Map")
                fig.colorbar(im6, ax=axs[1,2])

                im7 = axs[1,3].imshow(torch.log10((R_num[b].abs() / (R_hom[b].abs() + eps_val) + eps_val)).cpu())
                axs[1,3].set_title("Log Ratio")
                fig.colorbar(im7, ax=axs[1,3])

                plt.tight_layout()
                plt.show()
                plt.show()
        if self.diagnostics:
            # diagnostics
            diagnostics = {
                "R_hom": R_hom,
                "R_het": R_num,
                "T_het": T_het,
                "k2_data": k2_data,
                'lap_u': lap_u,
                "mu_effective": (self.rho * omega[:,:,0]**2 / k2_data) / 1e3
            }
            if self.orthogonalize_het:
                diagnostics["alpha_proj"] = alpha
            return L_ratio, diagnostics
        return L_ratio


class CombinedRatioLoss(nn.Module):
    """
    Combines MSE data loss with Ratio physics loss.
    Returns: total_loss, data_loss, physics_loss
    """
    
    def __init__(self, fov=(0.2, 0.2), rho=1000,
                 lambda_data=1.0, lambda_ratio=0.1,
                 orthogonalize_het=False, mse_in_k_space=False,
                 viz=False, diagnostics=False):
        """
        Args:
            fov: field of view
            rho: density
            lambda_data: weight for MSE loss (set to 0 to disable)
            lambda_ratio: weight for ratio physics loss
            orthogonalize_het: orthogonalize T_het against R_hom
            mse_in_k_space: if True, compute MSE in k-space; else μ-space
            viz: visualization flag
            diagnostics: return diagnostics dict
        """
        super().__init__()
        
        self.ratio_loss = RatioLoss(
            fov=fov,
            rho=rho,
            orthogonalize_het=orthogonalize_het,
            viz=viz,
            diagnostics=diagnostics
        )
        
        self.lambda_data = lambda_data
        self.lambda_ratio = lambda_ratio
        self.mse_in_k_space = mse_in_k_space
        self.diagnostics = diagnostics
        
    def forward(self, k_pred, k_gt, mu_pred, mu_gt, wave, mfre, fov=None):
        """
        Args:
            k_pred: (B, 1, H, W) predicted wave number [rad/m]
            k_gt: (B, 1, H, W) ground truth wave number [rad/m]
            mu_pred: (B, 1, H, W) predicted stiffness [kPa]
            mu_gt: (B, 1, H, W) ground truth stiffness [Pa]
            wave: (B, C, H, W, T) complex wave field
            mfre: (B,) frequency [Hz]
            fov: field of view (optional)
        
        Returns:
            total_loss: weighted combination
            data_loss: MSE component
            physics_loss: ratio component
        """
        # Data loss: MSE in k-space or μ-space
        if self.lambda_data > 0:
            if self.mse_in_k_space:
                data_loss = F.mse_loss(k_pred, k_gt)
            else:
                # Compare in μ space (convert mu_gt from Pa to kPa)
                data_loss = F.mse_loss(mu_pred, mu_gt)
        else:
            data_loss = torch.tensor(0.0, device=k_pred.device)
        
        # Physics loss: Ratio loss
        # Convert mu_pred from kPa to Pa for physics
        mu_pred_pa = mu_pred
        
        if self.diagnostics:
            physics_loss, diag = self.ratio_loss(
                wave, mu_pred_pa, mfre, W=1.0
            )
        else:
            physics_loss = self.ratio_loss(
                wave, mu_pred_pa, mfre, W=1.0
            )
        
        # Combined loss
        total_loss = (self.lambda_data * data_loss + 
                     self.lambda_ratio * physics_loss)
        
        if self.diagnostics:
            return total_loss, data_loss, physics_loss, diag
        
        return total_loss, data_loss, physics_loss
