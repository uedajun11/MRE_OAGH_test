# +
import torch
import torch.nn as nn
import numpy as np

class CombinedLoss(nn.Module):
    def __init__(self, lambda_hom=1.0, lambda_data = 1.0, physics_loss=None):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.lambda_hom = lambda_hom
        self.physics_loss = physics_loss
        self.lambda_data = lambda_data

    def forward(self, pred_wave_number, gt, pred_stiffness=None, wave_tensor=None, frequencies=None):
        # MSE first (always computed)
        data_loss = self.mse_loss(pred_wave_number, gt)
        
        physics_loss_val = torch.tensor(0.0, device=pred_wave_number.device)
        # Physics loss (optional)
        if self.physics_loss is not None:
            if wave_tensor is None or frequencies is None:
                raise ValueError("wave_tensor and frequencies required for physics loss")
            physics_loss_val = self.physics_loss(
                wave_tensor=wave_tensor,
                stiffness_kpa=pred_stiffness,
                frequencies=frequencies
            )
        total_loss = self.lambda_data*data_loss + self.lambda_hom * physics_loss_val

        return total_loss, data_loss,physics_loss_val
