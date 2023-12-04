'''
Librería de preprocesamiento EEG
--
Ammi Beltrán & Fernanda Borja
'''
import os
#
import numpy as np

import copy
import pandas as pd
import torch 

from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

SANE_P =[ "h01",
        "h02",
        "h03",
        "h04",
        "h05",
        "h06",
        "h07",
        "h08",
        "h09",
        "h10",
        "h11",
        "h12",
        "h13",
        "h14",
        ]

ABNORMAL_P = ["s01",
            "s02",
            "s03",
            "s04",
            "s05",
            "s06",
            "s07",
            "s08",
            "s09",
            "s10",
            "s11",
            "s12",
            "s13",
            "s14",
]

class CustomEEGDataset(Dataset):
    def __init__(self, csv_file , root_dir , transform = None, ):

        try:
            self.loc_df = pd.read_csv(os.path.join(root_dir,csv_file)).drop(labels="Unnamed: 0", axis = 1)
        except:
            self.loc_df = pd.read_csv(os.path.join(root_dir,csv_file))

        self.transform = transform
        self.root_dir = root_dir

    def __len__(self,):
        return len(self.loc_df)
        
    def __getitem__(self, idx):

        eeg_file = os.path.join(self.root_dir,
                                self.loc_df.iloc[idx, 2])
        eeg = torch.load(eeg_file)
            
        if self.transform is not None:
            ch = self.transform(eeg[0])
            return ch, eeg[1]

        return eeg
    
class DFSpliter():
    def __init__(self, train_size= 0.8, val_size = 0.2, save = False, seed = 69, mode = "down_ch") -> None:
        self.train_size = train_size
        self.val_size = val_size
        self.save = save
        self.seed = seed
        self.mode = mode

    def __call__(self, csv_file, root_path):
        try:
            loc_df = pd.read_csv(os.path.join(root_path,csv_file)).drop(labels="Unnamed: 0", axis = 1)
        except:
            loc_df = pd.read_csv(os.path.join(root_path,csv_file))
        # loc_df = csv_file
        patients = loc_df["Patient"].unique()

        sanes = [pat for pat in patients if pat in SANE_P]
        abnormals = [pat for pat in patients if pat in ABNORMAL_P]


        np.random.seed(self.seed)

        np.random.shuffle(sanes)
        np.random.shuffle(abnormals)

        s_end_idx = round(len(sanes)*self.train_size)
        a_end_idx = round(len(abnormals)*self.train_size)

        train_patients = [*sanes[:s_end_idx], *abnormals[:a_end_idx]]
        val_patients = [*sanes[s_end_idx:], *abnormals[a_end_idx:]]
        
        train_df = pd.DataFrame()
        for patient in train_patients:
            train_df = pd.concat([train_df,loc_df[loc_df["Patient"] == patient]])

        val_df = pd.DataFrame()
        for patient in val_patients:
            val_df = pd.concat([val_df,loc_df[loc_df["Patient"] == patient]])
        
        val_df.reset_index(inplace=True, drop= True)
        train_df.reset_index(inplace=True, drop=True)

        if self.save:
            train_df.to_csv(f"{self.mode}_train_feats.csv", encoding= "utf-8", index = False)
            val_df.to_csv(f"{self.mode}_val_feats.csv", encoding="utf-8", index=False)
        print("CSVs creados")
        return train_df,val_df
    
def Masking(channel: torch.Tensor, window: int= 250):
    '''
    Set to zero 
    Input: \\
    -channel = Tensor \\
    -window = Number of samples to set to zero
    Output: Numpy array masked
    '''
    # st = time.time()
    # New
    size = channel.size()
    mask = torch.ones(size)
    for i in range(size[0]):
        lenght = torch.randint(0, window, (1,)).item()
        start = torch.randint(0, size[1] - lenght + 1, (1,))
        mask[i, start:(start + lenght)] = 0
    masked = channel * mask
    # et = time.time()
    # print(f"total time = {et - st} s")
    return masked

def DCVoltage(channel : torch.Tensor, max_magnitude: float = 0.5):
    ''' 
    Add a DC component between [-max_mangitude, max_magnitude]\\
    Input:  \\
    -channel = Tensor \\
    -max_magnitude = max value to be added
    Output: Numpy array 
    '''
    # st = time.time()
    size = channel.size()
    dc_comp = (torch.rand(size[0])*2 - 1)*max_magnitude
    dc_comp = dc_comp.view(-1, 1)
    # print(dc_comp.size())
    channel = torch.add(channel, dc_comp)
    # et = time.time()
    # print(f"total time = {et - st} s")
    return channel      

def GaussianNoise(channel: torch.Tensor, std: float = 0.2):
    '''
    Add Gaussian Noise with zero mean and std deviation
    Input:  -channel = Numpy array
            -std = Gaussian std
    Output: Channel with additive gaussian noise added
    '''
#     st = time.time()
    size = channel.size()
    noise = torch.normal(mean = 0.0, std = std, size = size)
    # noise = np.random.normal(loc = 0, scale= std, size= channel_size)
    noisy_channel = torch.add(channel, noise)
#     et = time.time()
#     print(f"total time = {et - st} s")
    return noisy_channel

def Time_Shift(channel: torch.Tensor, min_shift: int = -50, max_shift: int = 50):
    # st = time.time()
    shift = int(torch.randint(low = min_shift, high = max_shift, size = (1, ))/2)
    shift_ch = F.pad(channel.unsqueeze(2), (0, 0, shift, -shift), mode = "reflect").squeeze(2)

    # et = time.time()
    # print(f"total time = {et - st} s")
    return shift_ch

def Amplitude(channel: torch.Tensor,min_amp:float = 0.5, max_amp: float = 2):
    # st = time.time()
    size = channel.size()
    factors = (torch.rand(size[0])*(min_amp -max_amp)) + max_amp
    factors = factors.view(-1, 1)
    channel = torch.mul(channel, factors)
    # et = time.time()
    # print(f"total time = {et - st} s")
    return channel


#Augmentation set
AUGMENTATIONS = [Time_Shift,
                 Amplitude,
                 DCVoltage,
                 GaussianNoise,
                 Masking]

class Augmentations(nn.Module):
    def __init__(self, n_aug, multi = False, augmentations = None) -> None:
        self.n_aug = n_aug
        self.multi = multi
        self.augmentations = augmentations

    def __call__(self, channel):

        aug_batch = channel
        for i in range(self.n_aug):
            aug_batch = self.augmentations[i](aug_batch)

        return aug_batch