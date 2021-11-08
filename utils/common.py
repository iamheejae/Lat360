import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

import math

from tensorboardX import SummaryWriter

def count_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _wrap_variable(input1, input2, use_gpu, in_device):
    if use_gpu:
        input1, input2 = (input1.to(in_device), input2.to(in_device))
    else:
        input1, input2 = (input1.cpu(), input2.cpu())
    return input1, input2

def _comput_PSNR(imgs1, imgs2, pixel_max):

    N = imgs1.size()[0]
    imdiff = imgs1 - imgs2
    imdiff = imdiff.view(N, -1)
    rmse = torch.sqrt(torch.mean(imdiff**2, dim=1))
    psnr = 20*torch.log(pixel_max/rmse)/math.log(10) 
    psnr = torch.sum(psnr)
    return psnr

def write_train_summary(summary, curr_epoch_loss, average_L_photo, average_L_warping, start_epoch, epoch):
    summary.add_scalar('Train/Total Loss per Epoch', curr_epoch_loss, start_epoch + epoch)
    summary.add_scalar('Train/Photometric loss per Epoch', average_L_photo, start_epoch + epoch)
    summary.add_scalar('Train/Warping loss per Epoch', average_L_warping, start_epoch + epoch)

def write_val_summary(summary, val_psnr, val_ssim, start_epoch, epoch):
    summary.add_scalar('Validation/PSNR per Epoch', val_psnr, start_epoch + epoch)
    summary.add_scalar('Validation/SSIM per Epoch', val_ssim, start_epoch + epoch)

### Loss ###
def loss_recon(loss_fn, net_output, img_t_GT, H, W):

    pred = net_output['SR']
    L_photo = loss_fn(pred, img_t_GT.to(device = pred.device))         

    return L_photo


def loss_all(loss_fn, net_output, img_t_GT, H, W):

    pred = net_output['SR']

    l_weight = [(0.32/2**scale) for scale in range(5)]
    l_final = 1 - sum(l_weight)

    # for warping loss  
    warping, warp_lev2, warp_lev3, warp_lev4, warp_lev5, warp_lev6  = net_output['warp_source'], net_output['warp_lev2'], net_output['warp_lev3'], net_output['warp_lev4'], net_output['warp_lev5'], net_output['warp_lev6']   
    img_t_GT_lev2 = F.interpolate(img_t_GT.to(device = warp_lev2.device),(int(H/4), int(W/4)), mode = 'bicubic')
    img_t_GT_lev3 = F.interpolate(img_t_GT.to(device = warp_lev3.device),(int(H/8), int(W/8)), mode = 'bicubic')
    img_t_GT_lev4 = F.interpolate(img_t_GT.to(device = warp_lev4.device),(int(H/16), int(W/16)), mode = 'bicubic')
    img_t_GT_lev5 = F.interpolate(img_t_GT.to(device = warp_lev5.device),(int(H/32), int(W/32)), mode = 'bicubic')
    img_t_GT_lev6 = F.interpolate(img_t_GT.to(device = warp_lev6.device),(int(H/64), int(W/64)), mode = 'bicubic')
           

    # define entire loss
    L_photo = loss_fn(pred, img_t_GT.to(device = pred.device))            
    L_warping= l_final*loss_fn(warping, img_t_GT.to(device = warping.device))+l_weight[0]*loss_fn(warp_lev2,img_t_GT_lev2.to(device = warp_lev2.device))+l_weight[1]*loss_fn(warp_lev3,img_t_GT_lev3.to(device = warp_lev3.device))+l_weight[2]*loss_fn(warp_lev4,img_t_GT_lev4.to(device = warp_lev4.device))+l_weight[3]*loss_fn(warp_lev5,img_t_GT_lev5.to(device = warp_lev5.device))+l_weight[4]*loss_fn(warp_lev6,img_t_GT_lev6.to(device = warp_lev6.device))

    return L_photo, L_warping

### Load Model ###
def load_pwc(model):

    load_pwc_ckpt = torch.load('./saved_model/pwc_net_chairs.pth.tar')
    pwc_ckpt = {f'module.DE360.{k}':v for k, v in load_pwc_ckpt.items()}
    model_dict = model.state_dict()            
    pretrained_dict = {k: v for k, v in pwc_ckpt.items() if k in model_dict}
    if pretrained_dict=={}:
        raise Exception('NO MATCHED KEYS FROM PRETRAINED PWC')

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    return model

def load_syntheticLat360(model):

    ckpt = torch.load('./saved_model/synthetic_lat360.pt')
    load_lat360_ckpt = ckpt['model_state_dict']
    lat360_ckpt = {f'module.Lat360.{k}':v for k, v in load_lat360_ckpt.items()}

    model_dict = model.state_dict()  

    pretrained_dict = {k: v for k, v in lat360_ckpt.items() if k in model_dict}

    if pretrained_dict=={}:
        raise Exception('NO MATCHED KEYS FROM PRETRAINED synthetic_Lat360')

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    #Freeze synthetic Lat360 model
    for name, p in model.named_parameters():
        if "Lat360" in name:
            p.requires_grad = False

    return model


