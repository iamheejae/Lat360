## <center>Official PyTorch implementation of the paper "360 Image Reference-based Super-Resolution using Latitude Aware Convolution Learned from Synthetic to Real" ##

### <center>Hee-Jae Kim, Je-Won Kang, Byung-Uk Lee ###
  
[[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9617634) [[Project Page]](https://iamheejae.github.io/lat360.github.io/)
  
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
- You can download pre-computed indices for LatConv in [LatConv_Index](https://drive.google.com/file/d/1ahBSPm0QjcWcToalhzRWrSsK4xQZRben/view?usp=sharing).
- Place the files in './models/Index'. 
- We will also provide generation code soon.
  
## Prepare dataset ###
We used Synthetic360 dataset and Real360 dataset to train our model. 

- Before generating datasets, randomly rotate ERP images for data augmentation. 
  Please refer to our codes in (https://github.com/iamheejae/360-Image-XYZ-Axis-Rotation).

- Run generate_traindataset.py & generate_valdataset.py to prepare dataset. 
  Each dataset is an hdf5 file, which contains '/HR_dataset' and '/LR_dataset'.
  
## How to train Lat360 ###
- First, train using Synthetic360 dataset
  ```
  ./train.sh
  ```

- Then, transfer Learning using Real360 dataset
  ```
  ./transfer_learning.sh
  ```
  
## Experimental Results ###
  
<center><img src="https://user-images.githubusercontent.com/42056469/143803751-ce62b1ef-a5ce-4050-b9f5-5c72b9587826.PNG" vspace="25px"></center>
