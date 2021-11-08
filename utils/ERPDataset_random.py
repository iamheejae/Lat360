from __future__ import division
import os
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import pdb 
import scipy.misc
from random import *
from torch.utils.data import Dataset

class create_random_dataset(Dataset):

    def __init__(self, root, view_mode):       
        path = os.path.join(root)
        f = h5py.File(path, 'r')

        self.view_mode = view_mode
       
        #*****Load each dataset as array: [numData X numView X channel X H X W]*****#
        self.hr_dataset = f.get('/HR_dataset')
        self.lr_dataset = f.get('/LR_dataset')

        self.hr_dataset = np.array(self.hr_dataset, dtype = np.float32)
        self.lr_dataset = np.array(self.lr_dataset, dtype = np.float32)

        self.numView = self.hr_dataset.shape[1]

    def __len__(self):
        return self.hr_dataset.shape[0]

    def GenerateViewPair(self, view_mode):

        if (view_mode == 'random'):            

            v_tar = 0
            v_source = np.random.randint(1,self.numView)

        elif (view_mode == 'fixed'):            

            v_tar = 0
            v_source = 1

        else:
            raise Exception('Wrong view mode')    

        return v_tar, v_source

    def __getitem__(self, idx):      
 
        # generate view position
        v_tar, v_source = self.GenerateViewPair(self.view_mode)

        img_t = self.lr_dataset[idx, v_tar, :, :, :] # target: Low-resolution image
        img_gt = self.hr_dataset[idx, v_tar, :, :, :] # ground-truth
        img_s = self.hr_dataset[idx, v_source, :, :, :] # source: High-resolution image
    
        # transform np image to torch tensor
        img_gt_tensor = torch.Tensor(img_gt)
        img_t_tensor = torch.Tensor(img_t)        
        img_s_tensor = torch.Tensor(img_s)              
      
        return img_gt_tensor, img_t_tensor, img_s_tensor

