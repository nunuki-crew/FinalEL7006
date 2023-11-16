import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class CustomEEGDataset(Dataset):
    def __init__(self, csv_file , root_dir , transform = None, multi = False, ):

        try:
            self.loc_df = pd.read_csv(os.path.join(root_dir,csv_file)).drop(labels="Unnamed: 0", axis = 1)
        except:
            self.loc_df = pd.read_csv(os.path.join(root_dir,csv_file))
        # self.loc_df = loc_df
        self.transform = transform
        self.root_dir = root_dir
        self.multi = multi
    def __len__(self,):
        return len(self.loc_df)
        
    def __getitem__(self, idx):

        eeg_file = os.path.join(self.root_dir,
                                self.loc_df.iloc[idx, 3])
        eeg = torch.load(eeg_file)
            
        if self.transform is not None:
            ch1 = self.transform(eeg)
            ch2 = self.transform(eeg)
            return ch1, ch2

        return eeg, eeg
    
class DFSpliter():
    def __init__(self, train_size= 0.8, val_size = 0.2, save = False, seed = 69) -> None:
        self.train_size = train_size
        self.val_size = val_size
        self.save = save
        self.seed = seed

    def __call__(self, csv_file, root_path):
        try:
            loc_df = pd.read_csv(os.path.join(root_path,csv_file)).drop(labels="Unnamed: 0", axis = 1)
        except:
            loc_df = pd.read_csv(os.path.join(root_path,csv_file))
        # loc_df = csv_file
        patients = loc_df["Patient"].unique()
        np.random.seed(self.seed)
        np.random.shuffle(patients)
        end_idx = round(len(patients)*self.train_size)

        train_patients = patients[:end_idx]
        val_patients = patients[end_idx:]
        
        train_df = pd.DataFrame()
        for patient in train_patients:
            train_df = pd.concat([train_df,loc_df[loc_df["Patient"] == patient]])

        val_df = pd.DataFrame()
        for patient in val_patients:
            val_df = pd.concat([val_df,loc_df[loc_df["Patient"] == patient]])
        
        val_df.reset_index(inplace=True, drop= True)
        train_df.reset_index(inplace=True, drop=True)

        if self.save:
            train_df.to_csv("train_feats.csv", encoding= "utf-8", index = False)
            val_df.to_csv("val_feats.csv", encoding="utf-8", index=False)
        print("CSVs creados")
        return train_df,val_df
    
def Masking(channel: np.array, window: int= 150):
    '''
    Set to zero 
    Input:  -channel = Numpy array
            -window = Number of samples to set to zero
    Output: Numpy array masked
    '''
    channel_size = len(channel)
    first = np.random.randint(0,channel_size- window)
    masked = channel.copy()
    masked[first:first+window] = 0

    return masked

def DCVoltage(channel : np.array, max_magnitude: float = 0.5):
    ''' 
    Add a DC component between [-max_mangitude, max_magnitude]
    Input:  -channel = Numpy array
            -max_magnitude = max value to be added
    Output: Numpy array 
    '''
    dc_comp = (np.random.random(1)*2 - 1)*max_magnitude
    dispaced_channel = channel + dc_comp
    return dispaced_channel    

def GaussianNoise(channel: np.array, std: float = 0.2):
    '''
    Add Gaussian Noise with zero mean and std deviation
    Input:  -channel = Numpy array
            -std = Gaussian std
    Output: Channel with additive gaussian noise added
    '''
    channel_size = len(channel)
    noise = np.random.normal(loc = 0, scale= std, size= channel_size)
    noisy_channel = channel + noise
    return noisy_channel

def Time_Shift(channel: np.array, min_shift: int = 0, max_shift: int = 50 ):
    ''' 
    Shifts the channel n samples between min_shift and max_shift using reflection pad
    Input:  -channel = Numpy array
            -min_shift = Min number of samples to shift
            -max_shhift = Max number of samples to shift  
    Output: Shifted channel
    '''
    n_shift = np.random.randint(min_shift,max_shift)
    channel_size = len(channel)
    padded_array = np.pad(channel,pad_width= n_shift, mode = "reflect")
    right_left = np.random.choice((0,2))
    shifted_array = padded_array[n_shift*right_left:channel_size + n_shift*right_left]
    return shifted_array
def Amplitude(channel :np.array, max_amplitude: float = 1.5):
    '''
    Modifies the ampliude of the channel values between [1+max_amplitude,1-max_amplitude]
    Input:  -channel = Numpy array
            -max_amplitude = Max aplitude to add
    Output: Boosted channel
    '''
    amplitude = 1 + ((np.random.random(1)*2 -1) * max_amplitude)
    boosted_channel = channel*amplitude
    return boosted_channel

def Permutation(channel: np.array, win_samples: int = 500):
    '''
    Permutates the arrays by secuences of win_samples len
    Ensure its divisible by the total len of the array or the len of the output secuence will be wrong
    Input:  -channel = Numpu array
            -win_samples = Number of samples per secuences (N_sec = len(channel)// win_samples)
    Output: Permutated secuence
    '''

    n_seqs = len(channel)// win_samples
    random_idx = np.random.choice(np.arange(0,n_seqs, 1), n_seqs, replace=False ) 
    permutated = np.concatenate([channel[win_samples*i: win_samples*(i+1)] for i in random_idx])
    return permutated
def Temporal_Invertion(channel: np.array):
    ''' 
    Return the array reversed
    Input:  -channel = Numpy array
    Output: Reversed array
    '''
    reversed = channel[::-1]
    return reversed

def Negation(channel: np.array):
    '''
    Inverts the full array
    Input: -channel = Numpy array
    Output: Inverted array
    '''
    negated = channel * (-1)
    return negated

#Augmentation set
AUGMENTATIONS = [Negation,
                 Time_Shift,
                 Amplitude,
                 DCVoltage,
                 GaussianNoise,
                 Temporal_Invertion,
                 Permutation,
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
