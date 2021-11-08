import torch
import torchvision.transforms as transforms

import numpy as np
import math
import os
import sys
import pdb

from model_utils import *
sys.path.insert(0,'./INDEX/')

class LatConv(nn.Module):
    def __init__(self, in_ch, kernel_size):
        super(LatConv, self).__init__()        

        self.ksize = kernel_size
        self.adpative_conv = base_conv(in_ch, in_ch, kernel_size = self.ksize, stride = self.ksize) # stride = kerne_size

    def load_index(self, H, W, ksize):    

        idx_filename = os.path.join(f'./models/INDEX/latconv_idx_ksize_{ksize}_{H}_{W}.npy')  
        idx_latconv = torch.from_numpy(np.load(idx_filename))     
        
        return idx_latconv

    def latitude_dependent_scaled(self, x, ksize):
        B, C, H, W = x.size()     
        self.in_device = x.device

        x1d = x.view(B,C,H*W)
        idx_flat = self.load_index(H, W, ksize)
        idx_flat = idx_flat.to(device=self.in_device)

        latconv_patches = x1d[:,:, idx_flat.long()]
        scaled_x = latconv_patches.view(B, C, ksize*H, ksize*W).to(device=self.in_device)       

        return scaled_x
        
    def forward(self, x):

        B, C, H, W = x.size()
        scaled_x = self.latitude_dependent_scaled(x, self.ksize)
        out = self.adpative_conv(scaled_x)

        return out







