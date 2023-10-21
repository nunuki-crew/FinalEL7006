"""
Models designed for SSL training
Ammi Beltrán y Fernanda Borja
"""
# Import libraries
import os
import dataprep as prep
import numpy as np
import torch
from torch import nn

# Models
"""
Convolutional Encoder 
"""

# Aux nn.Module
class Permute(nn.Module):
    def __init__(self, a, b, c):
        super(Permute, self).__init__()
        self.a = a
        self.b = b
        self.c = c
    def forward(self, x):
        return x.permute(self.a , self.b, self.c)
    
##################################################

class Convolutional_Enc(nn.Module):

    def __init__(self, out_size = 4):
        super(Convolutional_Enc, self).__init__()
        self.out_size = out_size
        # Reflection pad
        # We begin with 3 convolutional blocks (top to bottom)
        self.conv1 = nn.Sequential(
            # To fit kernel size = 128
            nn.ReflectionPad1d((63,64)),
            nn.Conv1d(in_channels = 1, out_channels = 100, kernel_size = 128),
        )
        self.conv2 = nn.Sequential(
            # To fit kernel size = 64
            nn.ReflectionPad1d((31,32)),
            nn.Conv1d(in_channels = 1, out_channels = 100, kernel_size = 64),
        )
        self.conv3 = nn.Sequential(
            # To fit kernel size = 16
            nn.ReflectionPad1d((7,8)),
            nn.Conv1d(in_channels = 1, out_channels = 50, kernel_size = 16),
        )
        # Repeated block
        self.iterable = nn.Sequential(
            Permute(0, 2, 1),
            nn.Linear(in_features = 250, out_features = 250),
            nn.ReLU(),
            Permute(0, 2, 1),
            nn.BatchNorm1d(num_features = 250), # [batch size, canales]
            nn.ReflectionPad1d((31, 32)), 
            nn.Conv1d(in_channels = 250, out_channels = 250, kernel_size = 64),
        )
        # Outer layer
        self.outer = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(num_features = 250),
            nn.ReflectionPad1d((31, 32)),
            nn.Conv1d(in_channels = 250, out_channels = self.out_size, kernel_size = 64),
        )

    def forward(self, x):
        # Pass trough convolutionals
        xc1 = self.conv1(x)
        xc2 = self.conv2(x)
        xc3 = self.conv3(x)
        # Concatenate
        cat = torch.cat((xc1, xc2, xc3), dim = 1)
        # Iterable phase
        for i in range(4):
            cat = self.iterable(cat)
        # End layer
        end = self.outer(cat)
        #
        return end
    
"""
Projector
"""
class Projector(nn.Module):
    def __init__(self):
        super(Projector, self).__init__()

        # LSTM's bidireccionales
        self.lstm1 = nn.LSTM(
            input_size = 4,
            hidden_size = 256,
            batch_first = True,
            bidirectional = True
        )
        self.lstm2 = nn.LSTM(
            input_size = 4,
            hidden_size = 128,
            batch_first = True,
            bidirectional = True
        )
        self.lstm3 = nn.LSTM(
            input_size = 4,
            hidden_size = 64,
            batch_first = True,
            bidirectional = True
        )
        # Outer Layer
        # Takes first and last output
        self.outer = nn.Sequential(
            nn.Linear(in_features = 4*(64 + 128 + 256), out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 32),
        )
    def forward(self, x):
        # Downsampling
        half = nn.functional.interpolate(x, scale_factor = 0.5)
        quarter = nn.functional.interpolate(x, scale_factor = 0.25)
        # Pass through LSTM
        x = x.permute(0, 2, 1)
        half = half.permute(0, 2, 1)
        quarter = quarter.permute(0, 2, 1)
        lstm1, (h_n, c_n) = self.lstm1(x)
        lstm2, (h_n, c_n) = self.lstm2(half)
        lstm3, (h_n, c_n) = self.lstm3(quarter)
        # Get First and Last
        flo1 = lstm1[:, [0, -1], :].reshape(lstm1.shape[0], -1)
        flo2 = lstm2[:, [0, -1], :].reshape(lstm2.shape[0], -1)
        flo3 = lstm3[:, [0, -1], :].reshape(lstm3.shape[0], -1)
        # Concatenate
        cat = torch.cat((flo1, flo2, flo3), dim = -1)
        # Last Layer
        end = self.outer(cat)
        #
        return end

"""
Classifier
"""
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # LSTM's bidireccionales
        self.lstm1 = nn.LSTM(
            input_size = 4,
            hidden_size = 256,
            batch_first = True,
            bidirectional = True
        )
        self.lstm2 = nn.LSTM(
            input_size = 4,
            hidden_size = 128,
            batch_first = True,
            bidirectional = True
        )
        self.lstm3 = nn.LSTM(
            input_size = 4,
            hidden_size = 64,
            batch_first = True,
            bidirectional = True
        )
        # Outer Layer
        # Takes first and last output
        self.outer = nn.Sequential(
            nn.Linear(in_features = 4*(64 + 128 + 256), out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 2), # num of classes
            # nn.LogSoftmax()
        )

    def forward(self, x):
        # Downsampling
        
        half = nn.functional.interpolate(x, scale_factor = 0.5)
        quarter = nn.functional.interpolate(x, scale_factor = 0.25)
        # Pass through LSTM
        x = x.permute(0, 2, 1)
        half = half.permute(0, 2, 1)
        quarter = quarter.permute(0, 2, 1)
        lstm1, (h_n, c_n) = self.lstm1(x)
        lstm2, (h_n, c_n) = self.lstm2(half)
        lstm3, (h_n, c_n) = self.lstm3(quarter)
        # Get First and Last
        flo1 = lstm1[:, [0, -1], :].reshape(lstm1.shape[0], -1)
        flo2 = lstm2[:, [0, -1], :].reshape(lstm2.shape[0], -1)
        flo3 = lstm3[:, [0, -1], :].reshape(lstm3.shape[0], -1)
        # Concatenate
        cat = torch.cat((flo1, flo2, flo3), dim = -1)
        # Last Layer
        end = self.outer(cat)
        #
        return end
    
"""
Assembled structures
"""
# Pretext Barlow Twins
class Pretext_BT(nn.Module):
    def __init__(self):
        super(Pretext, self).__init__()
        self.encoder = Convolutional_Enc(out_size = 100) 
        # No projector!!!!!!
        # self.projector = Projector()
    def forward(self, x):
        first = self.encoder(x)
        end = self.projector(first)
        return end
# Pretext Task
class Pretext(nn.Module):
    def __init__(self):
        super(Pretext, self).__init__()
        self.encoder = Convolutional_Enc(out_size = 4) 
        self.projector = Projector()
    def forward(self, x):
        first = self.encoder(x)
        end = self.projector(first)
        return end

# Downstream Task
class Downstream(nn.Module):
    def __init__(self):
        super(Downstream, self).__init__()
        self.encoder = Convolutional_Enc() 
        self.classify = Classifier()
        # Freeze encoder parameters
        for params in self.encoder.parameters():
            params.requires_grad = False
    def forward(self, x):
        first = self.encoder(x)
        end = self.classify(first)
        return end