import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
#summary=SummaryWriter('./training_summary/log')

from torch.nn import DataParallel
import torch.optim.lr_scheduler as lr_scheduler

import math
import os
import time
import sys
import numpy as np
from tqdm import tqdm
from shutil import copyfile

sys.path.insert(0,'./models/')
sys.path.insert(0,'./utils/')
from utils import common
import CustomLoss 
import pytorch_ssim
import pdb
import cv2

class Solver(object):

    def __init__(self, model, check_point, **kwargs):

        self.model = model
        self.check_point = check_point
        self.num_epochs = kwargs.pop('num_epochs')
        self.batch_size = kwargs.pop('batch_size')
        self.learning_rate = kwargs.pop('learning_rate')
        self.optimizer = kwargs.pop('optimizer')
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=1)
        self.fine_tune = kwargs.pop('fine_tune', False)
        self.fine_tune_pwc = kwargs.pop('fine_tune_pwc', True)
        self.transfer_learning = kwargs.pop('transfer_learning', True)
        self.start_epoch = kwargs.pop('start_epoch')
        self.verbose = kwargs.pop('verbose', False)
        self.num_gpus = kwargs.pop('num_gpus')
        self.parallel = kwargs.pop('parallel')
        self._reset()

        self.loss_fn = CustomLoss.CharbonnierLoss()

    def _reset(self):
        
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:  
            self.device = torch.device('cuda')          
            if self.parallel:
                self.model = DataParallel(self.model).to(self.device)
            else: 
                self.model = self.model.cuda()

    def _epoch_step(self, dataset, epoch):        

        self.model.train() 
        num_params=common.count_param(self.model)
        print('Total number of parameters: ', num_params)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_gpus)         
        num_iters = len(dataset)//self.batch_size

        # settings for loss 
        running_loss, L_photo_check, L_warping_check = 0, 0, 0

        for i, (img_t_GT, img_t_LR, img_s_HR) in enumerate(tqdm(dataloader)):            

            B, C, H, W = img_t_LR.size()

            # Wrap with torch Variable
            img_t_LR, img_s_HR = common._wrap_variable(img_t_LR, img_s_HR, self.use_gpu, self.device)

            # zero the grad
            self.optimizer.zero_grad()

            # Forward     
            net_output = self.model(img_t_LR, img_s_HR)    

            if self.transfer_learning:
                loss = common.loss_recon(self.loss_fn, net_output, img_t_GT, H, W)
                running_loss += loss.item()
            
            else:
                L_photo, L_warping = common.loss_all(self.loss_fn, net_output, img_t_GT, H, W) 
                loss = L_photo + L_warping 

                running_loss += loss.item()
                L_photo_check += L_photo.item()
                L_warping_check += L_warping.item()
            
            loss.backward()
            #nn.utils.clip_grad_norm_(self.model.parameters(), 0.4)
            self.optimizer.step()

        average_loss = running_loss/num_iters
        average_L_photo = L_photo_check/num_iters
        average_L_warping = L_warping_check/num_iters
    
        # Save Result
        log_path = f'./log.txt'
        f = open(log_path, 'a')
        f.write('(Training) Epoch  %5d: Loss %.5f' % (self.start_epoch + epoch, average_loss))
        f.write('\n')

        if self.verbose:
            print('Epoch  %5d, loss %.5f' % (epoch, average_loss))
        return average_loss, average_L_photo, average_L_warping

    def _validation(self, dataset, is_test=False):

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_gpus)
        self.model.eval()
        avr_psnr = 0
        avr_ssim = 0

        for batch, (img_t_GT, img_t_LR, img_s_HR) in enumerate(dataloader):
            img_t_LR, img_s_HR = common._wrap_variable(img_t_LR, img_s_HR, self.use_gpu, self.device)

            net_output = self.model(img_t_LR, img_s_HR)
            output_batch = net_output['SR']

            # SSIM 
            ssim = pytorch_ssim.ssim(output_batch.to(self.device) + 0.5, img_t_GT.to(self.device) + 0.5, size_average=False)
            ssim = torch.sum(ssim).item()
            avr_ssim += ssim

            # PSNR 
            PIXEL_MAX = 255
            output = output_batch.to(self.device).data
            label = img_t_GT.to(self.device)   
            psnr = common._comput_PSNR(output*PIXEL_MAX, abel*PIXEL_MAX, PIXEL_MAX)
            psnr = psnr.item()
            avr_psnr += psnr

        epoch_size = len(dataset)
        avr_psnr /= epoch_size
        avr_ssim /= epoch_size

        return avr_psnr, avr_ssim, output

    def train(self, train_dataset, val_dataset):

        if self.transfer_learning: 
            summary=SummaryWriter('./training_summary/log_transfer_learning')
        else:
            summary=SummaryWriter('./training_summary/log')

        # Check 'fine_tuning' options 
        fine_tune_path = os.path.join(self.check_point, 'model_fine_tune.pt')
        if self.fine_tune and not os.path.exists(fine_tune_path):
            raise Exception('Cannot find %s.' % fine_tune_path)
        elif self.fine_tune and os.path.exists(fine_tune_path):
            if self.verbose:
                print('Loading %s for finetuning.' % fine_tune_path)            
            ckpt = torch.load(fine_tune_path)
            self.model.module.load_state_dict(ckpt['model_state_dict'])

        if self.fine_tune_pwc and not self.fine_tune:        
            self.model = common.load_pwc(self.model)

        if self.transfer_learning:
            self.model = common.load_syntheticLat360(self.model)      

        best_val_psnr = -1

        # Train the model        
        for epoch in range(self.num_epochs):
            curr_epoch_loss, average_L_photo, average_L_warping = self._epoch_step(train_dataset, epoch)  

            common.write_train_summary(summary, curr_epoch_loss, average_L_photo, average_L_warping, self.start_epoch, epoch)  

            self.scheduler.step()

            # Validation 
            if self.verbose:
                print('Validation with current epoch')

            val_psnr, val_ssim, val_out = self._validation(val_dataset)

            log_path = f'./log.txt'
            f = open(log_path, 'a')
            f.write('(Validation) Epoch %5d: PSNR  %.3fdB, SSIM %.3f' % (self.start_epoch + epoch, val_psnr, val_ssim))
            f.write('\n')

            if self.verbose:
                print('Val PSNR: %.3fdB. Val ssim: %.3f'% (val_psnr, val_ssim))

            common.write_val_summary(summary, val_psnr, val_ssim, self.start_epoch, epoch)  
                
            print('Saving model')

            if not os.path.exists(self.check_point):
                os.makedirs(self.check_point)

            # Directory for saving checkpoint
            model_path = os.path.join(self.check_point, 'epoch{}.pt'.format(epoch+self.start_epoch))
          
            if self.parallel:
                torch.save({'epoch': self.start_epoch + epoch, 'model_state_dict': self.model.module.state_dict(), 'loss':curr_epoch_loss, 'lr':self.learning_rate}, model_path)
            else:
                torch.save({'epoch': self.start_epoch + epoch, 'model_state_dict': self.model.state_dict(), 'loss':curr_epoch_loss, 'lr':self.learning_rate}, model_path)

            if best_val_psnr < val_psnr:
                print('Copy best model')
                best_path = os.path.join(self.check_point, 'best_model.pt')
                copyfile(model_path, best_path)
                best_val_psnr = val_psnr
            print('')

        summary.close()


