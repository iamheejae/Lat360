import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os
import argparse
import sys
import pdb
import time
import cv2
import math

sys.path.insert(0,'./models/')
sys.path.insert(0,'./utils/')

from models import mvsr
from torch.nn import DataParallel as DP

#******************** Get options for training********************#

parser = argparse.ArgumentParser(description = 'Argument for Estimate Optical Flow of ERP with Deformable Convolutional Layer')

parser.add_argument('--target', metavar='tar', type=str, default='target', help='target low resolution image')
parser.add_argument('--reference', metavar = 'ref', type=str, default='ref', help = 'reference high resolution image')
parser.add_argument('--gt', metavar = 'gt', type=str, default='', help = 'gt high resolution image')

parser.add_argument('--result_root',dest='result_root',  type=str, default='./results', help='directory to save test image')
parser.add_argument('--result_dir',dest='result_dir',  type=str, default='./results/test.png', help='directory to save test image')

parser.add_argument('--checkpoint',dest='checkpoint',  type=str, default='./best_model.pt', help='select two views: target view and neigbor view')

args = parser.parse_args()

def get_full_path(dataset_path):
    """
    Get full path of data based on configs and target path
    example: datasets/train    """
    
    return os.path.join(dataset_path)

def display_config():

    for arg in vars(args):
        print("%15s: %s" % (str(arg), str(getattr(args, arg))))
    print('')


def _psnr(img1, img2):
    mse = np.mean( (np.float64(img1) - np.float64(img2)) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255
    
    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def test():
    display_config()

    ###############################################################
    """ absolute directory"""
    target_root = get_full_path(args.target)
    reference_root = get_full_path(args.reference)
    gt_root = get_full_path(args.gt)
    checkpoint_root = get_full_path(args.checkpoint)

    result_root = args.result_root
    if not os.path.exists(result_root):
        os.makedirs(result_root)

    result_dir = get_full_path(args.result_dir)
    ##############################################################

    imgt = cv2.imread(target_root).transpose(2,0,1)
    imgs = cv2.imread(reference_root).transpose(2,0,1)
    gt = cv2.imread(gt_root).transpose(2,0,1)   

    imgt = torch.tensor(imgt).float()
    imgt = imgt.unsqueeze(0)
    imgt = imgt/255
    
    imgs = torch.tensor(imgs).float()
    imgs = imgs.unsqueeze(0)
    imgs = imgs/255

    gt = torch.tensor(gt).float()

    #Set model
    
    ckpt = torch.load(checkpoint_root)
    model = mvsr.mvsr()
    model.load_state_dict(ckpt['model_state_dict'])
    model=model.cuda()
    model.eval()

    #Calculate test loss and psnr       
    #start = time.time() 
    test_out = model(imgt.cuda(), imgs.cuda())
    #end = time.time()
    #print(end-start)
    out = test_out['SR']
    out = out*255 
    out = torch.clamp(out, 0, 255)
    out = out.squeeze(0).cpu().detach().numpy()

    test_psnr = _psnr(out.astype(np.uint8), gt.detach().numpy().astype(np.uint8))
    print('test psnr: ', test_psnr)
    #log_path = f'./result/test_log.txt'
    #f = open(log_path, 'a')
    #f.write('%s %.5f' % (args.target, test_psnr))
    #f.write('\n')

    #Save output as an image
    output = out.transpose(1,2,0)
    output = output.astype(np.uint8)
    cv2.imwrite(result_dir, output)

    # Save Result
    log_path = f'./test_log.txt'
    f = open(log_path, 'a')
    f.write(' Tar: %s / Ref: %s >> PSNR: %.5f' % (args.target, args.reference, test_psnr))
    f.write('\n')
 
if __name__ == '__main__':
    test()

