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

sys.path.insert(0,'./models/')
sys.path.insert(0,'./utils/')

from ERPDataset_random import create_random_dataset
from solver import Solver
from models import Lat360, transfer_Lat360
import CustomLoss 

#******************** Get options for training********************#

parser = argparse.ArgumentParser(description = 'Argument for 360 Reference-based SR')

parser.add_argument('--train_set', type=str, default='./data/train.h5', help='training dataset root')
parser.add_argument('--val_set', type=str, default='./data/val.h5', help='validation dataset root')
parser.add_argument('--view_mode', type=str, default='random', help='if randomly select view or not')
parser.add_argument('--fine_tune_pwc', action='store_true', help='If fine-tune flow estimator')
parser.add_argument('--save_path', type=str, default='./check_point', help='root for save trained models')

parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--fine_tune', action='store_true')

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--num_gpus', type=int, default=2)
parser.add_argument('--parallel', action='store_true', help='If use DataParallel or not. Default false')
parser.add_argument('--verbose', action='store_true')

parser.add_argument('--transfer_learning', action='store_true', help='If transfer learning')

args = parser.parse_args()

def get_full_path(dataset_path):    
    return os.path.join(dataset_path)

def display_config():
    print('##########################################################################')
    print('#    Lat360 (360 Image Reference-based Super-Resoluton using LatConv)    #')
    print('#                                 Pytorch                                #')
    print('#                   Heejae Kim, EwahWomans University                    #')
    print('##########################################################################')
    print('')
    print('-------------------------------YOUR SETTINGS------------------------------')
    for arg in vars(args):
        print("%15s: %s" % (str(arg), str(getattr(args, arg))))
    print('')


def main():
    display_config()

    train_root = get_full_path(args.train_set)
    val_root = get_full_path(args.val_set)

    #Set view mode
    view_mode = args.view_mode
    
    print('Contructing dataset...')

    start_trainset = time.process_time()
    train_dataset = create_random_dataset(train_root, args.view_mode)
    end_trainset = time.process_time()
    print('Train Dataset is Ready ...')
    print('Loading train-set takes ', end_trainset-start_trainset,' seconds')
    print()
    
    start_valset = time.process_time()
    val_dataset = create_random_dataset(val_root, args.view_mode)
    end_valset = time.process_time()
    print('Validation Dataset is Ready ...', )
    print('Loading validation-set takes ', end_valset - start_valset,' seconds')
    
    if args.transfer_learning:
        model = transfer_Lat360.transfer_Lat360()
    else:
        model = Lat360.Lat360()

    #**********Optimiazion Method**********#
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum = 0.9, weight_decay = 0.0005)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas = (0.9,0.999), eps = 1e-08, weight_decay = 0.00005, amsgrad=False)   
    elif args.optim == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate, lr_decay=0, weight_decay=0.00005, initial_accumulator_value=0, eps=1e-10)   
    
    #************** Checkpoint **************#
    save_path = args.save_path
    check_point = os.path.join(save_path)

    solver = Solver(
        model, check_point, batch_size=args.batch_size,
        num_epochs=args.num_epochs, learning_rate=args.learning_rate, optimizer=optimizer,
        fine_tune=args.fine_tune, fine_tune_pwc=args.fine_tune_pwc, verbose=args.verbose, num_gpus=args.num_gpus, parallel=args.parallel, start_epoch=args.start_epoch, transfer_learning=args.transfer_learning)

    print('Now Start Training ...')
    solver.train(train_dataset, val_dataset)

if __name__ == '__main__':
    main()

