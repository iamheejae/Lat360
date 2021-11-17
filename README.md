### <center>Official PyTorch implementation of the paper "360 Image Reference-based Super-Resolution using Latitude Aware Convolution Learned from Synthetic to Real" ###

# <center>Hee-Jae Kim, Je-Won Kang, Byung-Uk Lee #

<center><img src="https://user-images.githubusercontent.com/42056469/141826157-30379a39-4bcd-4789-835c-5bfdcbc5fde4.png" vspace="25px"></center>
  
## Dependencies ###
  - Python 3.6
  - PyTorch >= 1.0.0
  - numpy
  - h5py
  - scipy
  - cv2
  - matplotlib
  - tqdm
  - tensorboardX

## Installation ###  
1. We used correlation package from PWC-Net. 
To install correlation package, please follow the instruction in [PWC-Net](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch)
  
2. For LatConv, 
You can download pre-computed indices for LatConv in [LatConv_Index]().
Place the files in './models/Index'
We will provide the code for generating indices soon.
  
## Prepare dataset ###
We used Synthetic360 dataset and Real360 dataset to train our model. 

- Before generate dataset, randomly rotate ERP images for data augmentation. 
  Please refer to (https://github.com/iamheejae/360-Image-XYZ-Axis-Rotation).

- Run generate_traindataset.py & generate_valdataset.py to prepare dataset. 
  Each dataset is an hdf5 file, which contains '/HR_dataset' and '/LR_dataset'.
  
## How to train Lat360 ###
- Train using Synthetic360 dataset, 
  '''
  ./train.sh
  '''
  
- Transfer Learning using Real360 dataset
  '''
  ./transfer_learning.sh
  '''
  
  
