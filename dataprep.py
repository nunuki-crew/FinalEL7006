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
# FOR TESTING
# EDFDIR = "D:\\OneDrive\\OneDrive - Universidad de Chile\\Semestre X\\Inteligencia\\Proyecto\\dataset\\tuh_eeg"
EDFDIR = "c:\\Users\\TheSy\\Desktop\\tuh_eeg"

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

def eeg_filter(data, lfreq = 0.1, hfreq= 30):
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

def get_epochs(data, channels, window = 20):
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

def downsample(epoch, freq = 200):
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
    clip(channel_data,ch)
    if(int(edf.times[-1]) < 100):
        return np.empty(0)
    filtered = eeg_filter(channel_data)
    trimmed_data = temporal_crop(filtered)
    down_data = downsample(trimmed_data)
    epochs = get_epochs(down_data, ch)
    if norm:
        norm_data = normalization(epochs)
        norm_data = np.delete(norm_data,-1,2)
        return norm_data
    epochs = np.delete(epochs,-1,2)    
    return epochs

def Save_win(data,loc_df, save_dir, patient_id,session_id, save = False):

    for i, win in enumerate(data):
        sdir = f"{save_dir}\\{patient_id}\\{patient_id}_{session_id}_w{i+1}.pt"
        loc_df.loc[len(loc_df)] = [patient_id,session_id,i+1,sdir]

        if save:
            torch.save(win,sdir)
    return loc_df

def Save_ch(data,loc_df, save_dir, patient_id,session_id, save = False):
    
    for i, win in enumerate(data):
        for j, ch in enumerate(win):
            sdir = f"{save_dir}\\{patient_id}\\{patient_id}_{session_id}_w{i+1}_ch{j+1}.pt"
            loc_df.loc[len(loc_df)] = [patient_id,session_id,i+1,sdir]
            
            if save:
                torch.save(ch,sdir)
    return loc_df

def prep(path, save = False,mode = "per_win", save_dir = "data", sep = "\\"):
    ''' 
    Lectura de todos los edfs de cada paciente, guardado de ventanas temporales y csv de direcciones

    Inputs:
        -path
    Output:
        -dir_csv: csv con todos los datos de guardado de las ventanas de cada edf.    
    '''
    LEN_PAT = 8
    SESION_LEN = 15
    loc_df = pd.DataFrame(columns= ["Patient", "Session","N_Win", "Dir"], )
    save_dir = save_dir + mode

    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print("Data directory created :D")
    

    patient_path = glob.glob(path + '/**')
    for patient in patient_path:

        #Para guardar la id en el DF
        patient_id = patient[-LEN_PAT:]

        if save:
            if not os.path.exists(os.path.join(save_dir, patient_id)):
                os.makedirs(os.path.join(save_dir, patient_id))
        
        sessions = glob.glob(patient + '/**')
        for session in sessions:

            #Para guardar la sesion correspondiente
            session_id = session[-SESION_LEN:-(SESION_LEN - 4)]

            edfs = glob.glob(session+ "/**/*.edf")
            for edf in edfs:

                raw = mne.io.read_raw_edf(edf,preload=True)
                try:
                    data = EDFprep(raw,random = False)
                except:
                    print(f"{patient_id}_{session_id} failed")
                    continue
                if mode == "per_win":
                    loc_df = Save_win(data,loc_df,save_dir,patient_id,session_id, save)
                elif mode == "per_channel":
                    loc_df = Save_ch(data,loc_df,save_dir,patient_id,session_id,save)

    if save:
        if mode == "per_channel":
            loc_df.to_csv("prep_channels.csv", encoding= "utf-8" ,index = False)
        elif mode == "per_win":
            loc_df.to_csv("prep_windows.csv", encoding= "utf-8", index = False)
                
    return loc_df

def EDFplot(edf_path, vs_prep = False, win_s = 20, f_prep = 200,n_epoch = 0 ):
    '''
    Plot the first window after 60s of the edf 
    '''
    print(f"Ploting {edf_path}")
    names = ["Original", "Prep"]
    raw = mne.io.read_raw_edf(edf_path, preload= True)
    raw_freq = int(raw.info["sfreq"])
    raw_data = raw.get_data()

    first_channel = raw_data[0]

    n_axes = 1 + vs_prep * 1

    fig, axes = plt.subplots(nrows = n_axes, figsize=(30, 15))

    n_ep = n_epoch
    # first_win_original = first_channel[(60)*raw_freq:(60+win_s)*raw_freq]
    first_win_original = first_channel[(60 + n_ep*win_s)*raw_freq:(60 + n_ep*win_s + win_s)*raw_freq]
    
    if vs_prep:
        first_win_processed = np.squeeze(EDFprep(raw,n_channels = 1,norm = True, random = False)[n_ep])
        axes[0].plot(first_win_original, color = "black", label = names[0])
        axes[0].set_title(f"{names[0]} edf")
        axes[0].set_ylabel("uVoltage [Vs]")

        axes[1].plot(first_win_processed, color = "gray", label = names[1])
        axes[1].set_title(f"{names[1]} edf")
        axes[1].set_ylabel("uVoltage [Vs]")
        axes[1].set_xlabel("Samples [s*Hz]")
    else:
        axes.plot(first_win_original,color = "black", label = names[0])
        axes.set_title(f"{names[0]} edf")
        axes.set_ylabel("uVoltage [Vs]")
        axes.set_xlabel("Samples [s*Hz]")
    plt.show()
