# +
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import medfilt2d

#hi
class MREHelmholtzLoss(nn.Module):
    """Helmholtz equation loss for MRE: ||∇²u + k²u||² where k² = ρω²/μ"""
    
    def __init__(self, density=1000, fov=(0.2, 0.2), residual_type='normalized', 
                 k_filter=None, epsilon=1e-10,verbose=False):
        super().__init__()
        self.density = density
        self.fov = (fov, fov) if isinstance(fov, float) else fov
        self.residual_type = residual_type
        self.k_filter = k_filter
        self.epsilon = epsilon
        self.verbose = verbose
    
    def extract_fundamental_frequency(self, wave_tensor):
        """
        Extract fundamental frequency component using FFT (following your MATLAB approach)
        
        Args:
            wave_tensor: Input tensor of shape (B, 1, T, H, W) - 8 time steps/phases
            
        Returns:
            wave_H: Complex wave field at fundamental frequency (B, 1, H, W)
        """
        B, C, timesteps, H, W = wave_tensor.shape
        print(f"Time-domain input magnitude: {torch.abs(wave_tensor).mean():.3f}")

        
        if timesteps == 1:
            # If only one timestep, use as is
            wave_H = wave_tensor.squeeze(2).to(torch.complex64)
        else:
            time_dim = None
            for i, s in enumerate(wave_tensor.shape):
                if s > 1 and s <= 32:  # usually 8 or so in MRE
                    time_dim = i
                    break
            if time_dim is None:
                raise ValueError("Could not detect time dimension in wave_tensor.")

            print(f"Detected time dimension: {time_dim}")

            # FFT along detected time axis
            wave_F = torch.fft.fft(wave_tensor.to(torch.complex64), dim=time_dim)
            wave_H = torch.index_select(wave_F, time_dim, torch.tensor([1], device=wave_tensor.device)).squeeze(time_dim)
            wave_H = wave_H #/ wave_tensor.shape[time_dim]

            return wave_H
        
    def extract_fundamental_frequency_batch(self, wave_tensor):
        """
        Extract fundamental frequency component using FFT (following MATLAB approach)

        Args:
            wave_tensor: Input tensor of shape (B, C, H, W, T) or (B, C, T, H, W)
                         where T is typically 8.

        Returns:
            wave_H: Complex tensor of shape (B, C, H, W)
        """
        # Ensure complex dtype
        wave_tensor = wave_tensor.to(torch.complex64)
        shape = wave_tensor.shape
        ndim = wave_tensor.ndim

        # Detect time dimension automatically
        time_dim = -1

        # Compute FFT along the time dimension
        wave_F = torch.fft.fft(wave_tensor, dim=time_dim)


        # Extract index [1] along time dimension for ALL batch samples
        idx = torch.tensor([1], device=wave_tensor.device)
        wave_H_full = torch.index_select(wave_F, dim=time_dim, index=idx)

        # Squeeze ONLY the time dimension explicitly by dimension index
        wave_H_wo_time = wave_H_full.squeeze(time_dim)
        
        # Squeeze the channel dimension (dim 1) which is size 1
        wave_H_wo_channel = wave_H_wo_time.squeeze(1)
        
        if self.verbose:
            print(f"Detected time dimension: {time_dim}")
            print(f"Input shape: {wave_tensor.shape}")
            print(f"After FFT shape: {wave_F.shape}")
            print(f"After index_select shape: {wave_H_full.shape}")
            print(f"After squeeze(time_dim) shape: {wave_H_wo_time.shape}")
            print(f"After squeeze(1) shape: {wave_H_wo_channel.shape}")

            print(f"Mean |wave_tensor|: {torch.abs(wave_tensor).mean():.3e}")
            print(f"Mean |wave_H_wo_channel|: {torch.abs(wave_H_wo_channel).mean():.3e}")

        return wave_H_wo_channel
    
    def apply_spatial_filter(self, wave_H, k_filter=1000):
        """Apply spatial filter in frequency domain (matching your original approach)"""
        if k_filter is None:
            return wave_H

        B, C, H, W = wave_H.shape
        device = wave_H.device

        # Create frequency grid
        pixsize_y, pixsize_x = self.fov[0] / H, self.fov[0] / W
        origin_y, origin_x = H // 2, W // 2

        ky_range = torch.arange(-origin_y, H - origin_y, device=device, dtype=torch.float32)
        kx_range = torch.arange(-origin_x, W - origin_x, device=device, dtype=torch.float32)
        ky_grid, kx_grid = torch.meshgrid(ky_range, kx_range, indexing='ij')

        kx = 2 * np.pi / W / pixsize_x * kx_grid
        ky = 2 * np.pi / H / pixsize_y * ky_grid

        # Create filter: 1/(1+(|k|/k_filter)^4)
        k_magnitude = torch.sqrt(kx**2 + ky**2)
        filter_kernel = 1.0 / (1.0 + (k_magnitude / self.k_filter)**4)
        
        if self.verbose:
            print("\n--- Filter stats ---")
            print(f"min: {filter_kernel.min():.6f}, max: {filter_kernel.max():.6f}, mean: {filter_kernel.mean():.6f}, std: {filter_kernel.std():.6f}")

            print("\nkx center 5x5:\n", kx[H//2-2:H//2+3, W//2-2:W//2+3])
            print("\nky center 5x5:\n", ky[H//2-2:H//2+3, W//2-2:W//2+3])

        # Apply filter in frequency domain
        filtered_wave = torch.zeros_like(wave_H)
        for b in range(B):
            wave_current = wave_H[b, 0].clone()
            #wave_current[wave_current == 0] = 1e-10

            # FFT, filter, IFFT
            ft = torch.fft.fftshift(torch.fft.fft2(wave_current))
            filtered_ft = filter_kernel * ft
            filtered_wave[b, 0] = torch.fft.ifft2(torch.fft.ifftshift(filtered_ft))
# -


#         print("\n--- Filtered wave stats ---")
        abs_filtered = torch.abs(filtered_wave)
#         print(f"min: {abs_filtered.min():.6f}, max: {abs_filtered.max():.6f}, mean: {abs_filtered.mean():.6f}, std: {abs_filtered.std():.6f}")

        return filtered_wave

    def gradient_matlab(self, u, dy, dx):
        # y-gradient
        uy = torch.zeros_like(u)
        uy[1:-1,:] = (u[2:,:] - u[:-2,:]) / (2*dy)
        uy[0,:]    = (u[1,:] - u[0,:]) / dy
        uy[-1,:]   = (u[-1,:] - u[-2,:]) / dy

        # x-gradient
        ux = torch.zeros_like(u)
        ux[:,1:-1] = (u[:,2:] - u[:,:-2]) / (2*dx)
        ux[:,0]    = (u[:,1] - u[:,0]) / dx
        ux[:,-1]   = (u[:,-1] - u[:,-2]) / dx

        return ux, uy
    
    def compute_laplacian(self, wave_field):
        """Compute Laplacian ∇²u matching MATLAB's gradient approach."""
        B, C, H, W = wave_field.shape
        dy, dx = self.fov[0] / H, self.fov[0] / W
        #print(f"Pixel sizes: {dy*1000:.3f} mm × {dx*1000:.3f} mm")

        laplacian = torch.zeros_like(wave_field)
        

        for b in range(B):
            u = wave_field[b, 0]  # (H, W)

            # --- First derivatives (MATLAB style)
            ux, uy = self.gradient_matlab(u, dy, dx)
            # --- Second derivatives
            uxx, _ = self.gradient_matlab(ux, dy, dx)
            _, uyy = self.gradient_matlab(uy, dy, dx)
            lap = uxx + uyy
            laplacian[b, 0] = lap
            
            if self.verbose:
                # --- Print stats ---
                def print_stats(name, arr):
                    arr_abs = torch.abs(arr)
                    print(f"\n--- {name} stats ---")
                    print(f"min: {arr_abs.min():.6e}, max: {arr_abs.max():.6e}, "
                          f"mean: {arr_abs.mean():.6e}, std: {arr_abs.std():.6e}")
                    cy, cx = H // 2, W // 2
                    center_vals = arr[cy-2:cy+3, cx-2:cx+3]
                    print(f"{name} center 5x5 (real + imag):")
                    for row in center_vals:
                        print("  ".join([f"{v.real:.4f}+{v.imag:.4f}i" for v in row]))

                print_stats("ux", ux)
                print_stats("uy", uy)
                print_stats("uxx", uxx)
                print_stats("uyy", uyy)
                print_stats("laplacian", lap)

        return laplacian
    
    def compute_shear_modulus(self, k, freq, rho, medfilt_kernel=None):
        """Compute shear modulus (kPa) from wavenumber k"""
        sm = rho * ((2 * np.pi * freq) / k)**2 / 1000.0 # Pa -> kPa
        if medfilt_kernel is not None:
            sm = torch.tensor(medfilt2d(sm.cpu().numpy(), kernel_size=medfilt_kernel))
        return sm
    
    def directInverse(self, wave_tensor, frequencies,apply_medfilt=True):
        """
        Direct inversion for MRE: mu = rho * omega^2 / k^2

        Args:
            wave_tensor: torch.Tensor, shape (B, 1, H, W) or (B, 1, T, H, W)
            frequencies: scalar or torch.Tensor of shape (B,)
            apply_medfilt: apply 2D median filter to final mu

        Returns:
            sm: shear modulus in kPa, shape (B, H, W)
        """
        B = wave_tensor.shape[0]
        device = wave_tensor.device
        
         # Handle frequency input
        if not isinstance(frequencies, torch.Tensor):
            frequencies = torch.tensor(frequencies, device=device)
        if frequencies.numel() == 1:
            frequencies = frequencies.expand(B)
            
        # Extract fundamental frequency if needed
        if wave_tensor.dim() == 5:  # (B, 1, 8, H, W)
            wave_H = self.extract_fundamental_frequency_batch(wave_tensor)
        else:  # (B, 1, H, W) - already extracted
            wave_H = wave_tensor.to(torch.complex64)
        
        # Apply spatial filtering if specified
        if self.k_filter is not None:
            wave_field = self.apply_spatial_filter(wave_H.unsqueeze(1))
        else:
            wave_field = wave_H.unsqueeze(1)
        
        laplacian = self.compute_laplacian(wave_field)
        
        k = torch.sqrt(laplacian / (-wave_H.unsqueeze(1)))  # complex-safe if lap or wave_H are complex. use wave_H in denom since more robust to noise
        k = k.real  # take only real part, like MATLAB's real()
      
        omega = (2 * np.pi * frequencies).view(B, 1, 1, 1)

        sm = self.density * (omega / k)**2# / 1000.0 # kPa units

        # Optional median filter
        if apply_medfilt:
            sm_np = sm.squeeze(1).cpu().numpy()
            for b in range(B):
                sm_np[b] = medfilt2d(sm_np[b], kernel_size=3)
            sm = torch.tensor(sm_np, device=wave_H.device)
            
        if self.verbose:
            for i in range(wave_H.real.shape[0]):
                plt.figure()
                viridis = cm.get_cmap('viridis')  # closest built-in
                plt.imshow(wave_H[i].real.squeeze(), cmap=viridis, vmin=0, vmax=torch.max(wave_H[i].real[:]))
                plt.axis('equal')
                plt.colorbar()
                plt.title(f'First Harmonic (wave_H.real) {i}')
                plt.show()
                print(f'Mean First Harmonic (wave_H.real) {i}: {torch.mean(wave_H.real[i][:])}')
            
                plt.figure()
                viridis = cm.get_cmap('viridis')  # closest built-in
                plt.imshow(wave_field[i].real.squeeze(), cmap=viridis, vmin=0, vmax=torch.max(wave_field[i].real[:]))
                plt.axis('equal')
                plt.colorbar()
                plt.title(f'Filtered Wave (wave_field) {i}')
                plt.show()
                print(f'Mean Filtered Wave (wave_field) {i}: {torch.mean(wave_field.real[i][:])}')
                
                plt.figure()
                viridis = cm.get_cmap('viridis')  # closest built-in
                plt.imshow(laplacian[i].real.squeeze(), cmap=viridis, vmin=0, vmax=torch.max(laplacian[i].real[:]))
                plt.axis('equal')
                plt.colorbar()
                plt.title(f'Laplacian (laplacian.real) {i}')
                plt.show()
                print(f'Mean Laplacian (laplacian.real) {i}: {torch.mean(laplacian[i].real[:])}')
  
                plt.figure()
                viridis = cm.get_cmap('viridis')  # closest built-in
                plt.imshow(k[i].squeeze(), cmap=viridis, vmin=0, vmax=torch.max(k[i][:]))
                plt.axis('equal')
                plt.colorbar()
                plt.title(f'Wave Number (k) {i}')
                plt.show()
                print(f'Mean Wave Number {i}: {torch.mean(k[i][:])}')
            
                plt.figure()
                viridis = cm.get_cmap('viridis')  # closest built-in
                plt.imshow(sm[i].squeeze(), cmap=viridis, vmin=0, vmax=15)
                plt.axis('equal')
                plt.colorbar()
                plt.title(f'Shear Mod (kPa) {i}')
                plt.show()
                print(f'Mean Shear Modulus (kPa) {i}: {torch.mean(sm[i][:])}')
                
                plt.figure()
                viridis = cm.get_cmap('viridis')  # closest built-in
                plt.imshow(sm[i].squeeze(), cmap=viridis, vmin=0, vmax=15)
                plt.axis('equal')
                plt.colorbar()
                plt.title(f'Shear Mod Median Filtered(kPa) {i}')
                plt.show()
                print(f'Mean Shear Modulus Median Filtered (kPa) {i}: {torch.mean(sm[i][:])}')
        return sm, k


    def forward(self, wave_tensor, stiffness_kpa, frequencies):
        """
        Args:
            wave_tensor: Wave tensor (B, 1, 8, H, W) - 8 time steps OR (B, 1, H, W) if already extracted
            stiffness_kpa: Shear modulus in kPa (B, 1, H, W) 
            frequencies: Frequencies in Hz (B,) or scalar
        """
        
        # Extract fundamental frequency if needed
#         if wave_tensor.dim() == 5:  # (B, 1, 8, H, W)
#             batch_size_before = wave_tensor.shape[0]
#             wave_H = self.extract_fundamental_frequency_batch(wave_tensor)
#             batch_size_after = wave_H.shape[0]
#             assert batch_size_before == batch_size_after, \
#                 f"Batch dimension not conserved! Before: {batch_size_before}, After: {batch_size_after}"
#             print(f"✓ Batch dimension conserved: {batch_size_before}")
#         else:  # (B, 1, H, W) - already extracted
#             wave_H = wave_tensor.to(torch.complex64)
        if wave_tensor.dim() == 5:  # (B, 1, 8, H, W)
            wave_H = self.extract_fundamental_frequency_batch(wave_tensor)
        else:  # (B, 1, H, W) - already extracted
            wave_H = wave_tensor.to(torch.complex64)
        
        # Apply spatial filtering if specified
        if self.k_filter is not None:
            wave_field = self.apply_spatial_filter(wave_H.unsqueeze(1))
        else:
            wave_field = wave_H.unsqueeze(1)
            
        B, C, H, W = wave_field.shape
        device = wave_field.device
        
        # Handle frequency input
        if not isinstance(frequencies, torch.Tensor):
            frequencies = torch.tensor(frequencies, device=device)
        if frequencies.numel() == 1:
            frequencies = frequencies.expand(B)
        
        # After unit conversions
        
        omega = (2 * np.pi * frequencies).view(B, 1, 1, 1)
        mu_pa = stiffness_kpa * 1000  # Convert kPa to Pa
        if self.verbose:
            print(f"Input stiffness: {stiffness_kpa.mean():.1f} kPa")
            print(f"Converted mu: {mu_pa.mean():.0f} Pa")
            print(f"Frequency: {frequencies} Hz")
            print(f"Omega: {omega.mean():.1f} rad/s")
        
        # Numerical stability
        wave_safe = wave_field.clone()
        mask = torch.abs(wave_safe) < self.epsilon
        wave_safe[mask] = torch.complex(
            torch.full_like(wave_safe.real[mask], self.epsilon),
            torch.full_like(wave_safe.imag[mask], self.epsilon)
        )
        
        # Compute Helmholtz residual: ∇²u + k²u
        laplacian = self.compute_laplacian(wave_safe) # Use filtered wave for laplacian
        residual_raw = torch.zeros(B, 1, H, W, device=device)    

        
        for b in range(B):
            # Calculate the predicted k² = ρω²/μ where μ is the predicted stiffness
            k_squared_raw = self.density * omega[b]**2 / torch.clamp(mu_pa[b], min=self.epsilon)
            
            k_squared_median = torch.median(k_squared_raw)
            k_squared = torch.clamp(k_squared_raw,
                                    min=k_squared_median*0.01,
                                    max=k_squared_median*100)

            # Helmholtz residual for this batch item
            helmholtz_term = laplacian[b].squeeze() + k_squared * wave_safe[b] # Originally: use original wave, now trying with wave_safe
            residual_raw[b, 0] = torch.abs(helmholtz_term)**2
            
            
        # Apply residual normalization
        if self.residual_type == 'raw':
            residual = residual_raw
        elif self.residual_type == 'normalized':
            residual = torch.zeros_like(residual_raw)
            for b in range(B):
                r = residual_raw[b]
                r_min, r_max = r.min(), r.max()
                residual[b] = (r - r_min) / (r_max - r_min + self.epsilon)
        elif self.residual_type == 'log':
            residual = torch.log10(residual_raw + self.epsilon)
            
        elif self.residual_type == 'wave_normalized':
            residual = torch.zeros(B, 1, H, W, device=device)
    
            for b in range(B):
                # Normalize by wave magnitude BEFORE squaring
                wave_mag = torch.abs(wave_safe[b]) + self.epsilon
                normalized_term = helmholtz_term / wave_mag
                # Residual
                residual[b, 0] = torch.abs(normalized_term)**2
                
        elif self.residual_type == 'wave_standardized':
            helmholtz_mag = torch.abs(helmholtz_term)
            # Compute mean & std over spatial dims
            mu = helmholtz_mag.mean(dim=(-2,-1), keepdim=True)
            sigma = helmholtz_mag.std(dim=(-2,-1), keepdim=True) + + 1e-6

            # Standardized residual
            residual = ((helmholtz_mag - mu) / sigma) ** 2

        else:
            raise ValueError(f"Unknown residual_type: {self.residual_type}")
        
        return residual.mean()

