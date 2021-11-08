import torch
import torch.nn as nn
import torch.nn.functional as func

import pdb
class CharbonnierLoss(nn.Module):

    def __init__(self):
        super(CharbonnierLoss,self).__init__()

    def forward(self,pre,gt):

        B,C,H,W = pre.shape

        diff = torch.sum(torch.sqrt((pre - gt )**2 + 0.001 **2)) / (B*C*H*W)

        return diff

class CharbonnierLoss_L1(nn.Module):

    def __init__(self):
        super(CharbonnierLoss_L1,self).__init__()

    def forward(self,loss):

        B,C,H,W = loss.shape

        diff = torch.sum(torch.sqrt(loss**2 + 0.00001 **2)) / (B*C*H*W)

        return diff

class EuclideanLoss(nn.Module):

    def __init__(self):
        super(EuclideanLoss,self).__init__()

    def forward(self,pre,gt):

        N = pre.shape[0]        
        diff = torch.sum((pre - gt ).pow(2)) / (N * 2)

        return diff


class smoothness_loss(nn.Module):
    def __init__(self):
        super(smoothness_loss, self).__init__()
        
    def CharbonnierLoss(self,loss):

        B,C,H,W = loss.shape

        diff = torch.sum(torch.sqrt(loss**2 + 0.001 **2)) / (B*C*H*W)

        return diff

    def forward(self, flow):
        B, C, H, W = flow.size()

        up_crop = flow[:, :, 1:, :]
        down_crop = flow[:, :, :-1, :]
        left_crop = flow[:, :, :, 1:]
        right_crop = flow[:, :, :, :-1]

        up_l_crop = flow[:, :, 1:, 1: ]
        up_r_crop = flow[:, :, 1:, :-1]
        down_l_crop = flow[:, :, :-1, 1:]
        down_r_crop = flow[:, :, :-1, :-1]

        error = self.CharbonnierLoss(up_crop-down_crop) + self.CharbonnierLoss(left_crop-right_crop) + self.CharbonnierLoss(up_l_crop-down_r_crop) + self.CharbonnierLoss(up_r_crop-down_l_crop)

        return error

#class flow_smoothness(nn.Module):
#    def __init__(self):
#        super(flow_smoothness,self).__init__()

#    def forward(self,f_flow, b_flow):

#        N = f_flow.shape[0]        

#        return diff

