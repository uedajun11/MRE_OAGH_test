import numpy as np
import torch
import os
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


# Convolution operator
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, 1, 1, bias=True),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_out, C_out, 3, 1, 1, bias=True),
            nn.BatchNorm2d(C_out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)

# Up sampling operator
class UpSampling(nn.Module):
    def __init__(self, C_in, C_out):
        super(UpSampling, self).__init__()
        self.Up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(C_in, C_out, 3, 1, 1, bias=True)
        )

    def forward(self, x):
        return self.Up(x)

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.in_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.forget_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.cell_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)

    def forward(self, input, h_state, c_state):
        conc_inputs = torch.cat((input, h_state), 1)

        in_gate = self.in_gate(conc_inputs)
        forget_gate = self.forget_gate(conc_inputs)
        out_gate = self.out_gate(conc_inputs)
        cell_gate = self.cell_gate(conc_inputs)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        c_state = (forget_gate * c_state) + (in_gate * cell_gate)
        h_state = out_gate * torch.tanh(c_state)

        return h_state, c_state

# +
# class set_values(nn.Module):
#     def __init__(self, hidden_size, height, width):
#         super(set_values, self).__init__()
#         self.hidden_size = int(hidden_size)
#         self.height = int(height)
#         self.width = int(width)
#         self.dropout = nn.Dropout(0.7)
#         self.RCell = RNNCell(self.hidden_size, self.hidden_size)

#     def forward(self, seq, xinp):
#         device = xinp.device
        
# #         # Handle both batch-first and sequence-first inputs
# #         #if len(xinp.shape) == 5:  # (seq_len, batch, channels, H, W) or (batch, seq_len, channels, H, W)
# #             # Assume sequence-first based on your usage pattern
# #         seq_len, batch_size = xinp.shape[0], xinp.shape[1]
# #         sequence_first = True
# # #         else:  # (batch, channels, H, W) - single timestep
# # #             batch_size = xinp.shape[0]
# # #             seq_len = 1
# # #             sequence_first = False
# # #             xinp = xinp.unsqueeze(0)  # Add sequence dimension
        
# #         # Initialize output tensor
# #         xout = torch.zeros(seq_len, batch_size, self.hidden_size, self.height, self.width, 
# #                           device=device, dtype=xinp.dtype)
        
# #         # Initialize hidden states with proper batch size
# #         h_state = torch.zeros(batch_size, self.hidden_size, self.height, self.width, 
# #                              device=device, dtype=xinp.dtype)
# #         c_state = torch.zeros(batch_size, self.hidden_size, self.height, self.width, 
# #                              device=device, dtype=xinp.dtype)

# #         for t in range(seq_len):
# #             # Process current timestep
# #             input_t = seq(xinp[t])  # xinp[t] is (batch_size, channels, H, W)
# #             xout[t] = input_t
# #             h_state, c_state = self.RCell(input_t, h_state, c_state)

# #         # Return in same format as input
# #         if not sequence_first and seq_len == 1:
# #             xout = xout.squeeze(0)  # Remove sequence dimension if it was added
            
# #         return self.dropout(h_state), xout

#         xinp = xinp.permute(1, 0, 2, 3, 4)  # -> (seq_len, batch, C, H, W)

#         seq_len, batch_size = xinp.shape[0], xinp.shape[1]

#         xout = torch.zeros(seq_len, batch_size, self.hidden_size, self.height, self.width,
#                            device=device, dtype=xinp.dtype)

#         h_state = torch.zeros(batch_size, self.hidden_size, self.height, self.width,
#                               device=device, dtype=xinp.dtype)
#         c_state = torch.zeros(batch_size, self.hidden_size, self.height, self.width,
#                               device=device, dtype=xinp.dtype)

#         for t in range(seq_len):
#             input_t = seq(xinp[t])  # shape: (batch_size, hidden_size, H, W)
#             xout[t] = input_t
#             h_state, c_state = self.RCell(input_t, h_state, c_state)
#         xout = xout.transpose(0, 1)  # Ensure batch-first for downstream
#         print(f"h_state shape: {h_state.shape}")
#         return self.dropout(h_state), xout
# -

class set_values(nn.Module):
    def __init__(self, hidden_size, height, width):
        super(set_values, self).__init__()
        self.hidden_size = int(hidden_size)
        self.height = int(height)
        self.width = int(width)
        self.dropout = nn.Dropout(0.7)
        self.RCell = RNNCell(self.hidden_size, self.hidden_size)

    def forward(self, seq, xinp):
        """
        xinp: Tensor of shape [batch, channels, seq_len, height, width]
        This function processes the sequence dimension properly.
        """
        device = xinp.device
        batch_size, channels, seq_len, height, width = xinp.shape


        # Output tensor shape: [batch, seq_len, hidden_size, H, W]
        xout = torch.zeros(batch_size, seq_len, self.hidden_size, self.height, self.width,
                           device=device, dtype=xinp.dtype)

        h_state = torch.zeros(batch_size, self.hidden_size, self.height, self.width,
                              device=device, dtype=xinp.dtype)
        c_state = torch.zeros(batch_size, self.hidden_size, self.height, self.width,
                              device=device, dtype=xinp.dtype)

        for t in range(seq_len):
            # Extract the t-th frame: [batch, channels, H, W]
            input_t = seq(xinp[:, :, t, :, :])  # Process with conv layers
            xout[:, t] = input_t
            h_state, c_state = self.RCell(input_t, h_state, c_state)
        permuted_xout = xout.permute(0, 2, 1, 3, 4)
        # permuted_xout shape: [batch_size, self.hidden_size, seq_len, self.height, self.width]

        return self.dropout(h_state), permuted_xout


# Network structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.img_size = 256
        self.input_ch = 1
        self.output_ch = 1
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = Conv(self.input_ch, 16)
        self.set1 = set_values(16, self.img_size, self.img_size)

        self.Conv2 = Conv(16, 32)
        self.set2 = set_values(32, self.img_size // 2, self.img_size // 2)  # Use // for integer division

        self.Conv3 = Conv(32, 64)
        self.set3 = set_values(64, self.img_size // 4, self.img_size // 4)

        self.Conv4 = Conv(64, 128)
        self.set4 = set_values(128, self.img_size // 8, self.img_size // 8)

        self.Conv5 = Conv(128, 256)
        self.set5 = set_values(256, self.img_size // 16, self.img_size // 16)

        self.Up5 = UpSampling(256, 128)
        self.Up_conv5 = Conv(256, 128)

        self.Up4 = UpSampling(128, 64)
        self.Up_conv4 = Conv(128, 64)

        self.Up3 = UpSampling(64, 32)
        self.Up_conv3 = Conv(64, 32)

        self.Up2 = UpSampling(32, 16)
        self.Up_conv2 = Conv(32, 16)

        self.Conv_1x1 = nn.Conv2d(16, self.output_ch, kernel_size=1, stride=1, padding=0)

        # Note: self.pred is defined but never used - consider removing

    def encoder(self, x):
        x1, xout = self.set1(self.Conv1, x)

        x2, xout = self.set2(nn.Sequential(self.Maxpool, self.Conv2), xout)

        x3, xout = self.set3(nn.Sequential(self.Maxpool, self.Conv3), xout)

        x4, xout = self.set4(nn.Sequential(self.Maxpool, self.Conv4), xout)

        x5, xout = self.set5(nn.Sequential(self.Maxpool, self.Conv5), xout)

        return x1, x2, x3, x4, x5

    def forward(self, x):
        # encoding path
        x1, x2, x3, x4, x5 = self.encoder(x)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((d5, x4), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((d4, x3), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
