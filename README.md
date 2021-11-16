### <center>Official PyTorch implementation of the paper "360 Image Reference-based Super-Resolution using Latitude Aware Convolution Learned from Synthetic to Real" ###

### <center>Hee-Jae Kim, Je-Won Kang, Byung-Uk Lee ###

<center><img src="https://user-images.githubusercontent.com/42056469/141826157-30379a39-4bcd-4789-835c-5bfdcbc5fde4.png" height="500" vspace="25px"></center>
  
<hr style="border: solid 1px gray;">
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

## Prepare dataset ###
We used Synthetic360 dataset and Real360 dataset to train our model. 

Prepare an hdf5 file, which contains '/HR_dataset' and '/LR_dataset'. 
  
<hr style="border: solid 1px gray;">
## How to train Lat360 ###
- Train using Synthetic360 dataset
- Transfer Learning using Real360 dataset
  
