### <center>Official PyTorch implementation of the paper "360 Image Reference-based Super-Resolution using Latitude Aware Convolution Learned from Synthetic to Real"

### <center>Hee-Jae Kim, Je-Won Kang, Byung-Uk Lee
### <center><i>Ewha Womans University 

<center><img src="https://user-images.githubusercontent.com/42056469/141826157-30379a39-4bcd-4789-835c-5bfdcbc5fde4.png" vspace="25px"></center>
  
<hr style="border: solid 1px gray;">
## Directory Structure  
  
project
│   README.md
|   run_synthetic.sh - main script to train Lat360
|   run_transfer_learning.sh - script for transfer learning
|   train.py - main file to run train/val
|   solver.py - check & change training/testing configurations here
|   test.py - defines different loss functions
└───model
│   │   common.py
│   │   cain.py - main model
|   |   cain_noca.py - model without channel attention
|   |   cain_encdec.py - model with additional encoder-decoder
└───data - implements dataloaders for each dataset
│   |   vimeo90k.py - main training / testing dataset
|   |   video.py - custom data for testing
│   └───symbolic links to each dataset
|       | ...
