'''
Librería de preprocesamiento EEG
--
Ammi Beltrán & Fernanda Borja
'''
# Required libraries
import os
import mne
import glob
#
import numpy as np
import math
import matplotlib.pyplot as plt

import copy
import pandas as pd

import torch 
# from pyFiles.dataprep2 import Masking
# FOR TESTING
# EDFDIR = "D:\\OneDrive\\OneDrive - Universidad de Chile\\Semestre X\\Inteligencia\\Proyecto\\dataset\\tuh_eeg"
EDFDIR = "c:\\Users\\TheSy\\Desktop\\tuh_eeg"

SANE =[ "h01.edf",
        "h02.edf",
        "h03.edf",
        "h04.edf",
        "h05.edf",
        "h06.edf",
        "h07.edf",
        "h08.edf",
        "h09.edf",
        "h10.edf",
        "h11.edf",
        "h12.edf",
        "h13.edf",
        "h14.edf",
        ]

ABNORMAL = ["s01.edf",
            "s02.edf",
            "s03.edf",
            "s04.edf",
            "s05.edf",
            "s06.edf",
            "s07.edf",
            "s08.edf",
            "s09.edf",
            "s10.edf",
            "s11.edf",
            "s12.edf",
            "s13.edf",
            "s14.edf",
]

def channel_select(data, channels):
    '''
    Selects channels from array 
    '''
    extracted = data.pick(channels, exclude="bads")
    return extracted

def clip(data, channels,max= 500e-6):
    def cliper(array):
        for i in range(len(array)):
            if abs(array[i]) > max:
                array[i] = math.copysign(max,array[i])
        return array
    data.apply_function(cliper, picks=channels, channel_wise= True)

def eeg_filter(data, lfreq = 1, hfreq= 70):
    '''
    
    '''
    data_copy = copy.copy(data)
    filtered = data_copy.filter(#l_freq = lfreq,
                                l_freq = lfreq,
                                h_freq = hfreq,
                                method = "iir",
                                )
    return filtered

def temporal_crop(data, tin = 60, tfin = 12*60):
    ''' 
    Cut the channels from the second "tin" to "tfin"
    '''
    data_copy = copy.copy(data)
    croped = data_copy.crop(tmin = tin, tmax = min(tfin, int(data.times[-1])))
    return croped

def get_epochs(data, channels, window = 10):
    ''' 
    window es la ventana de tiempo
    '''
    data_copy = copy.copy(data)
    # Create events
    events = mne.make_fixed_length_events(data_copy, duration = window, first_samp = True)
    # Divide accordingly
    picks = channels
    epochs = mne.Epochs(raw = data_copy, events = events, picks = picks, preload = True,
                        tmin = 0., tmax = window, baseline = None,
                        flat = dict(eeg = 1e-6))
    
    epochs.drop(-1,reason = "Unfixed duration")
    return epochs

def downsample(epoch, freq = 100): # original 200
    ''' 
    Downsamples the data given by a factor
    En nuestro caso, down corresponde a (frecuencia que queremos)/(frecuencia actual)
    '''
    down = epoch.resample(freq, npad = "auto")
    return down

def normalization(epochs):
    obj = mne.decoding.Scaler(info = epochs.info, scalings='mean')
    values = obj.fit_transform(epochs.get_data())
    return values

def EDFprep(edf, n_channels = 19, norm = True, random = True, ):
    '''
    Pipeline
    '''

    #Random channel select
    channels = edf.ch_names
    if random:
        ch = np.random.choice(channels[:-3], size = n_channels, replace=False)
    else:
        ch = channels[:n_channels]
    
    channel_data = channel_select(edf, ch)
    filtered = eeg_filter(channel_data)
    clip(filtered, ch)
    trimmed_data = temporal_crop(filtered)
    re_ref = trimmed_data.copy().set_eeg_reference(ref_channels="average")
    down_data = downsample(re_ref)
    epochs = get_epochs(down_data,down_data.ch_names)

    if norm:
        norm_data = normalization(epochs)
        norm_data = np.delete(norm_data,-1,2)
        return norm_data 
    return epochs

def Save_win(data,loc_df, final_dir, patient_id,label, save = False):

    for i, win in enumerate(data):
        sdir = os.path.join(final_dir,f"{patient_id}_w{i+1}.pt")
        loc_df.loc[len(loc_df)] = [patient_id,i+1,sdir]

        win_save = torch.from_numpy(win).type(torch.FloatTensor)
        if save:
            torch.save((win_save, label),sdir)
    return loc_df

def Save_ch(data,loc_df, final_dir, patient_id,label, save = False):
    
    for i, win in enumerate(data):
        for j, ch in enumerate(win):
            # sdir = f"{save_dir}/{patient_id}/{patient_id}_{session_id}_w{i+1}_ch{j+1}.pt"
            sdir = os.path.join(final_dir, f"{patient_id}_w{i+1}_ch{j+1}.pt")
            loc_df.loc[len(loc_df)] = [patient_id,i+1,sdir]
            
            if save:
                ch_save = torch.from_numpy(ch).type(torch.FloatTensor)
                ch_save = ch_save.unsqueeze(dim = 0)
                torch.save((ch_save, label), sdir)
    return loc_df

def prep(path, save = False, mode = "per_win", save_dir = "data"):
    ''' 
    Lectura de todos los edfs de cada paciente, guardado de ventanas temporales y csv de direcciones

    Inputs:
        -path
    Output:
        -dir_csv: csv con todos los datos de guardado de las ventanas de cada edf.    
    
    '''
    folders = ["sane", "abnormal"]
    LEN_PAT = 7
    loc_df = pd.DataFrame(columns= ["Patient","N_Win", "Dir"], )
    save_dir = os.path.join(save_dir, mode)

    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print("Data directory created :D")
    

    patient_path = glob.glob(path + "/*.edf" )
    for patient in patient_path:

        #Para guardar la id en el DF
        patient_id = patient[-LEN_PAT:]

        if patient_id in SANE:
            label = torch.zeros(1)
        if patient_id in ABNORMAL:
            label = torch.ones(1)
        
        patient_id = patient_id[:-4]
        
        folder = folders[int(label)]
        final_dir = os.path.join(save_dir, folder)

        if save:
            if not os.path.exists(final_dir):
                os.makedirs(final_dir)

        raw = mne.io.read_raw_edf(patient,preload=True)
        try:
            data = EDFprep(raw,random = False)
        except:
            print(f"{patient_id}_failed")
            continue

        if mode == "per_win":
            loc_df = Save_win(data,loc_df,final_dir,patient_id,label, save)
        elif mode == "per_channel":
            loc_df = Save_ch(data,loc_df,final_dir,patient_id,label, save)

    if save:
        if mode == "per_channel":
            loc_df.to_csv("down_prep_channels.csv", encoding= "utf-8" ,index = False)
        elif mode == "per_win":
            loc_df.to_csv("down_prep_windows.csv", encoding= "utf-8", index = False)
                
    return loc_df


