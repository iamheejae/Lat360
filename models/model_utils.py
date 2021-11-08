import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

import pdb
import cv2

#--------------------------------- common ---------------------------------#
def save_feature(feature, filename):    
    feature_np = feature.cpu().detach().numpy()    
    #feature_np = feature_np.transpose(1,2,0)
    feature_np_rescale = 255*(feature_np-feature_np.min())/(feature_np.max()-feature_np.min())
    feature_np_rescale = feature_np_rescale.astype(np.uint8)
    cv2.imwrite(filename, feature_np_rescale)    
    return

def colorize_flow(flow):
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[0,:,:].astype(np.float32), flow[1,:,:].astype(np.float32))

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    # Normalize
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img

def sub_mean(x):
    mean = x.mean(2, keepdim=True).mean(3, keepdim=True)
    x -= mean
    return x, mean

def fwarp(x, flo):

    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] %im2
    flo: [B, 2, H, W]
    """
    B, C, H, W = x.size()

    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = F.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = F.grid_sample(mask, vgrid)
        
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
        
    return output*mask

#----------------------------- Reconstruction ---------------------------------#
class resblock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, res_scale=1):
       
        super(resblock, self).__init__()
        self.res_scale = res_scale
        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=stride, padding=(5//2), bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size=5, stride=stride, padding=(5//2), bias=True)
        
    def forward(self, x):
        x_init = x
        out = self.conv_1(x)
        out = self.relu(out)
        out = self.conv_2(out)
        out = out * self.res_scale + x_init
        return out       

class reconstruct(nn.Module):

    def __init__(self, num_blocks, in_channel, out_channel, n_feats, res_scale):
        super(reconstruct, self).__init__()

        self.num_blocks = num_blocks
        self.head= nn.Conv2d(in_channel, n_feats, kernel_size=5, stride=1, padding=(5//2), bias=True)

        self.reconNet = nn.ModuleList()
        for i in range(self.num_blocks):
            self.reconNet.append(resblock(in_channel = n_feats, out_channel=n_feats, res_scale = res_scale))       

        self.tail_1 = nn.Conv2d(n_feats, n_feats, kernel_size=5, stride=1, padding=(5//2), bias=True)
        self.tail_2 = nn.Conv2d(n_feats, out_channel, kernel_size=5, stride=1, padding=(5//2), bias=True)

    def forward(self, input1, input2):

        x = torch.cat((input1 ,input2),1) 

        x = F.relu(self.head(x))
        x1 = x #out channel : 64
        for i in range(self.num_blocks):
            x = self.reconNet[i](x)
        x = self.tail_1(x)
        x = x + x1

        out = self.tail_2(x)

        return out

#--------------------------------- Occlusion Mask Generator ---------------------------------#
class OMG(nn.Module):
    """A five-layer network for predicting mask"""

    def __init__(self, in_ch, out_ch):
        super(OMG, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 96, 3, padding=1)
        self.conv5 = nn.Conv2d(96, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.01)

        x = self.conv6(x)
        x = F.sigmoid(x)

        return x

#--------------------------------- LatConv ---------------------------------#

# convolutions
def base_conv(in_channel, out_channel, kernel_size, stride, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=True),            
            nn.LeakyReLU(0.1)            
            )

#---------------------------- Transfer Learning ----------------------------#
class transfer(nn.Module):

    def __init__(self, num_blocks, in_channel, out_channel, n_feats, res_scale):
        super(transfer, self).__init__()

        self.num_blocks = num_blocks
        self.head= nn.Conv2d(in_channel, n_feats, kernel_size=5, stride=1, padding=(5//2), bias=True)

        self.reconNet = nn.ModuleList()
        for i in range(self.num_blocks):
            self.reconNet.append(resblock(in_channel = n_feats, out_channel=n_feats, res_scale = res_scale))       

        self.tail_1 = nn.Conv2d(n_feats, n_feats, kernel_size=5, stride=1, padding=(5//2), bias=True)
        self.tail_2 = nn.Conv2d(n_feats, out_channel, kernel_size=5, stride=1, padding=(5//2), bias=True)

    def forward(self, x):

        x = F.relu(self.head(x))
        x1 = x 

        for i in range(self.num_blocks):
            x = self.reconNet[i](x)

        x = self.tail_1(x)
        x = x + x1
        out = self.tail_2(x)

        return out

