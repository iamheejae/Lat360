import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Lat360 import Lat360
from model_utils import *

import numpy as np
import sys
import pdb
import os
import cv2

class transfer_Lat360(nn.Module):

    def __init__(self):
        super(transfer_Lat360, self).__init__()

        self.Lat360 = Lat360()  
        self.transfer_head = transfer(8,3,3,64,1)
        self.transfer_tail = transfer(8,3,3,64,1)

    def forward(self, target_LR_img, source_HR_img):
        B, C, H, W = target_LR_img.size()  
      
        target_LR_syn = self.transfer_head(target_LR_img)
        source_HR_syn = self.transfer_head(source_HR_img)        

        syn_out = self.Lat360(target_LR_syn, source_HR_syn) 
        syn_SR = syn_out['SR']

        recon = self.transfer_tail(syn_SR)
        SR_OUT = recon.clone()   

        net_output = {'SR' : SR_OUT}   

        return net_output


