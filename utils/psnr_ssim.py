import math
import numpy as np
import logging
import cv2
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_psnr_and_ssim(sr, hr):
    ### args:
        # sr: pytorch tensor, range [-1, 1]
        # hr: pytorch tensor, range [-1, 1]

    ### prepare data
    sr = (sr+1.) * 127.5
    hr = (hr+1.) * 127.5
    if (sr.size() != hr.size()):
        h_min = min(sr.size(2), hr.size(2))
        w_min = min(sr.size(3), hr.size(3))
        sr = sr[:, :, :h_min, :w_min]
        hr = hr[:, :, :h_min, :w_min]

    img1 = np.transpose(sr.squeeze().round().cpu().numpy(), (1,2,0))
    img2 = np.transpose(hr.squeeze().round().cpu().numpy(), (1,2,0))

    psnr = calc_psnr(img1, img2)
    ssim = calc_ssim(img1, img2)

    return psnr, ssim
