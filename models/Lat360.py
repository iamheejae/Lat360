import torch
import torch.nn as nn
import torch.nn.functional as F

from DE360 import DE360
from model_utils import *

import numpy as np
import pdb
import cv2

class Lat360(nn.Module):
    """Estimate Optical flow with ERP Dataset"""
    def __init__(self):
        super(Lat360, self).__init__()
        self.DE360 = DE360()   
        self.ReconNet = reconstruct(16,6,3,64,1) 
        self.OMG = OMG(92, 1) 

    def forward(self, E_tar, E_ref):
        B, C, H, W = E_tar.size()
        in_device = E_tar.device

        """ Flow estimation with Long-range 360 Disparity Estimator """ # DE360
        d_ref_to_tar = self.DE360(E_tar, E_ref)      

        disparity_r_to_t = 20 * F.interpolate(d_ref_to_tar['flow2'], (H, W), mode = 'bicubic') # bicubic upsampling (as in PWC-Net)
        E_ref_to_tar = fwarp(E_ref, disparity_r_to_t)
        
        """ Estimate occlusion & errors in warped reference image """ # OMG
        # input for OMG
        corr2 = F.interpolate(d_ref_to_tar['corr2'], (H, W), mode = 'bicubic')
        diff = torch.ones((B, C, H, W)).to(device=in_device)-(E_tar-E_ref_to_tar.to(device=in_device))

        occ_mask = self.OMG(torch.cat((diff, disparity_r_to_t, E_ref_to_tar, E_tar, corr2),1))
        reference = (occ_mask * E_ref_to_tar) + ((1-occ_mask) * E_tar) # Warped & filtered reference

        """ Synthesis """ #ReconNet       
        E_sr = self.ReconNet(E_tar, reference) 
        SR_OUT = E_sr.clone()   

        ################################# For warping loss (L_warp) #################################        
        ref_lev2 = F.interpolate(E_ref, (int(H/4), int(W/4)), mode = 'bicubic')
        ref_lev3 = F.interpolate(E_ref, (int(H/8), int(W/8)), mode = 'bicubic')
        ref_lev4 = F.interpolate(E_ref, (int(H/16), int(W/16)), mode = 'bicubic')
        ref_lev5 = F.interpolate(E_ref, (int(H/32), int(W/32)), mode = 'bicubic')
        ref_lev6 = F.interpolate(E_ref, (int(H/64), int(W/64)), mode = 'bicubic')
        
        warp_ref_lev2 = fwarp(ref_lev2, d_ref_to_tar['flow2'])   
        warp_ref_lev3 = fwarp(ref_lev3, d_ref_to_tar['flow3'])  
        warp_ref_lev4 = fwarp(ref_lev4, d_ref_to_tar['flow4'])  
        warp_ref_lev5 = fwarp(ref_lev5, d_ref_to_tar['flow5'])  
        warp_ref_lev6 = fwarp(ref_lev6, d_ref_to_tar['flow6'])       
        #############################################################################################

        net_output = {'SR' : SR_OUT, 'warp_source': E_ref_to_tar, 'warp_lev2':warp_ref_lev2, 'warp_lev3':warp_ref_lev3, 'warp_lev4':warp_ref_lev4, 'warp_lev5':warp_ref_lev5, 'warp_lev6':warp_ref_lev6} 

        return net_output


