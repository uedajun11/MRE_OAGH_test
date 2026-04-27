import numpy as np
import torch
import os
import scipy.io as sio
from sklearn import preprocessing
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split

class WDataset(Dataset):
    def __init__(self, ids, dir_input, offsets, fov=0.2, extension='.mat', transform=None):
        self.dir_input = dir_input
        self.extension = extension
        self.ids = ids
        self.offsets = offsets
        self.data_len = len(self.ids)
        self.fov = fov
        self.transform = transform

    def __getitem__(self, index):
        id_input = self.dir_input + self.ids[index]
        input_mat = sio.loadmat(id_input + self.extension)
        wave = input_mat['wave']
        mu = input_mat['mu']
        fov=self.fov
        mu = np.float64(mu)
        mfre = np.array(input_mat['mfre'])
        mfre = np.float64(np.array(mfre))
        index = np.int64(input_mat['index'])
        # mask = input_mat['mask']
        omega = 2 * torch.pi * mfre
        wave_number = fov * np.divide(omega, ((mu/1000)**(1/2)), out=np.zeros_like(mu), where=mu!=0)
        input = torch.from_numpy(wave).float().unsqueeze(0)
        if self.transform:
            input = self.transform(input)
        
        wave_num = torch.from_numpy(wave_number).float().unsqueeze(0)
        return input, mu, wave_num, mfre, fov, index
    
    def __len__(self):
        return self.data_len


# +
class PDataset(Dataset):
    def __init__(self, dir_input, offsets, fov=0.2, extension=".pt", transform=None, normalize_wave = False):
        self.dir_input = dir_input
        self.extension = extension
        self.offsets = offsets
        self.fov = float(fov)
        self.transform = transform
        self.normalize_wave = normalize_wave
        self.ids = [
            f.replace(extension, "")
            for f in os.listdir(dir_input)
            if f.endswith(extension)
        ]

    def __len__(self):
        return len(self.ids)

#     def __getitem__(self, idx):
#         sample = torch.load(
#             os.path.join(self.dir_input, self.ids[idx] + self.extension),
#             map_location="cpu"
#         )

#         wave = sample["wave"]          # [1,256,256,8]
#         mu = sample["mu"]
#         mfre = sample["mfre"]
#         index = sample["index"]

#         omega = 2 * torch.pi * mfre
#         rho = 1000.0  # kg/m³
#         wave_number = omega * torch.sqrt(rho / mu)  # [256,256] in rad/m
#         wave_num = wave_number.unsqueeze(0)  # [1,256,256]

#         if self.transform:
#             wave = self.transform(wave)

#         return wave,mu, wave_num, mfre, self.fov, index
    def __getitem__(self, idx):
        sample = torch.load(
            os.path.join(self.dir_input, self.ids[idx] + self.extension),
            map_location="cpu"
        )

        wave = sample["wave"]          # [1, 256, 256, N] where N is 4 or 8
        mu = sample["mu"]
        mfre = sample["mfre"]
        index = sample["index"]

        # --- Phase offset normalization ---
        n_offsets = wave.shape[-1]

        if n_offsets == self.offsets:
            pass  # already correct
        elif n_offsets == 8 and self.offsets == 4:
            wave = wave[..., ::2]  # indices 0,2,4,6 (1,3,5,7 in 1-based)
        else:
            raise ValueError(
                f"Unsupported offset combination: file has {n_offsets}, target is {self.offsets}"
            )
        
        # --- Per-sample wave normalization to [-1, 1] ---
        if self.normalize_wave:
            C = wave.abs().max().clamp(min=1e-10)
            wave = wave / C
        
        omega = 2 * torch.pi * mfre
        rho = 1000.0 # kg/m³
        wave_number = omega * torch.sqrt(rho / mu) # [256,256] in rad/m
        wave_num = wave_number.unsqueeze(0) # [1,256,256]

        if self.transform:
            wave = self.transform(wave) 

        return wave, mu, wave_num, mfre, self.fov, index


# -

def get_dataloader_for_train(dir_input, offsets=8, fov=0.2, batch_size=10, snr_db=None):
    #ids = [f[:-4] for f in os.listdir(dir_input)]
    ids = [f[:-4] for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f))]
    transform = AddGaussianNoiseSNR(snr_db) if snr_db is not None else None
    dset = WDataset(ids, dir_input, offsets, fov,transform=transform)
    dataloaders = {}
    dataloaders['train'] = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last = False, num_workers=2, pin_memory=True, persistent_workers=True)

    return dataloaders['train']

def get_Pdataloader_for_train(dir_input, offsets=8, fov=0.2, batch_size=10, snr_db=None, normalize_wave=False):
    #ids = [f[:-4] for f in os.listdir(dir_input)]
    #ids = [f[:-4] for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f))]
    transform = AddGaussianNoiseSNR(snr_db) if snr_db is not None else None
    dset = PDataset(dir_input, offsets, fov,transform=transform, normalize_wave=normalize_wave)
    dataloaders = {}
    dataloaders['train'] = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last = False, num_workers=8, pin_memory=True, persistent_workers=True)

    return dataloaders['train']


def get_dataloader_for_val(dir_input, offsets=8, fov=0.2, batch_size=10, snr_db=None):
    #ids = [f[:-4] for f in os.listdir(dir_input)]
    ids = [f[:-4] for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f))]
    transform = AddGaussianNoiseSNR(snr_db) if snr_db is not None else None
    dset = WDataset(ids, dir_input, offsets, fov,transform=transform)
    dataloaders = {}
    dataloaders['val'] = DataLoader(dset, batch_size=batch_size, shuffle=False, drop_last = False, num_workers=2, pin_memory=True, persistent_workers=True)

    return dataloaders['val']

def get_Pdataloader_for_val(dir_input, offsets=8, fov=0.2, batch_size=10, snr_db=None, normalize_wave=False):
    #ids = [f[:-4] for f in os.listdir(dir_input)]
    #ids = [f[:-4] for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f))]
    transform = AddGaussianNoiseSNR(snr_db) if snr_db is not None else None
    dset = PDataset(dir_input, offsets, fov,transform=transform, normalize_wave=normalize_wave)
    dataloaders = {}
    dataloaders['val'] = DataLoader(dset, batch_size=batch_size, shuffle=False, drop_last = False, num_workers=8, pin_memory=True, persistent_workers=True)

    return dataloaders['val']


# +
# def get_dataloader_for_test(dir_input, offsets=8, fov=0.2, batch_size=10, snr_db=None):
#     #ids = [f[:-4] for f in os.listdir(dir_input)]
#     ids = [f[:-4] for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f))]
#     transform = AddGaussianNoiseSNR(snr_db) if snr_db is not None else None
#     dset = WDataset(ids, dir_input, offsets, fov,transform=transform)
#     dataloaders = {}
#     dataloaders['test'] = DataLoader(dset, batch_size=batch_size, shuffle=False, drop_last = False, num_workers=2, pin_memory=True, persistent_workers=True)

#     return dataloaders['test']
# -

def get_dataloader_for_test(dir_input, offsets=8, fov=0.2, batch_size=10, snr_db=None, normalize_wave=False):
    #ids = [f[:-4] for f in os.listdir(dir_input)]
    #ids = [f[:-4] for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f))]
    transform = AddGaussianNoiseSNR(snr_db) if snr_db is not None else None
    dset = PDataset(dir_input, offsets, fov,transform=transform, normalize_wave=normalize_wave)
    dataloaders = {}
    dataloaders['test'] = DataLoader(dset, batch_size=batch_size, shuffle=False, drop_last = False, num_workers=8, pin_memory=True, persistent_workers=True)

    return dataloaders['test']


class AddGaussianNoiseSNR:
    def __init__(self, snr_db):
        self.snr_db = snr_db

    def __call__(self, signal):
        if isinstance(signal, torch.Tensor):
            signal = signal.clone().detach()
            signal_np = signal.cpu().numpy()
        else:
            signal_np = signal

        signal_power = np.mean(signal_np ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_np = np.random.normal(0, np.sqrt(noise_power), size=signal_np.shape)
        noisy_signal_np = signal_np + noise_np

        return torch.from_numpy(noisy_signal_np).float()
