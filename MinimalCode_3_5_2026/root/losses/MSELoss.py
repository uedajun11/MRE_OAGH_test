import torch
from torch.nn import functional as F
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self, mse_in_k_space):
        super().__init__()
        self.mse_in_k_space = mse_in_k_space
        
    def forward(self,  k_pred, k_gt, mu_pred, mu_gt, wave=None, mfre=None, fov=None):
        """
        Args:
            k_pred: (B, 1, H, W) predicted wave number [rad/m]
            k_gt: (B, 1, H, W) ground truth wave number [rad/m]
            mu_pred: (B, 1, H, W) predicted stiffness [kPa]
            mu_gt: (B, 1, H, W) ground truth stiffness [Pa]
            wave, mfre, fov: unused (for interface compatibility)
        Returns:
            total_loss, data_loss, physics_loss 
        """
        if self.mse_in_k_space:
            data_loss = F.mse_loss(k_pred, k_gt)
        else:
            if mu_pred.dim() == 4 and mu_pred.shape[1] == 1:
                mu_pred = mu_pred.squeeze(1)  # (B, 1, H, W) → (B, H, W)
            if mu_gt.dim() == 4 and mu_gt.shape[1] == 1:
                mu_gt = mu_gt.squeeze(1)
            data_loss = F.mse_loss(mu_pred, mu_gt)
        
        physics_loss = torch.tensor(0.0, device=k_pred.device)
        total_loss = data_loss
        
        return total_loss, data_loss, physics_loss
