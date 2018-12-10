## References
"""
"""

# Normal python libraries
import numpy as np 
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import glob
import os


# Pytorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

## Import utils files. First one used for pre-processing, others for different tasks related to the assignment
# import utils_processing
import utils

## Architecture for model
class nav_nw(nn.Module):

	def __init__(self):
		super(nav_nw, self).__init__()

		# Conv layers

		# Batch normalization

		# Activation functions

	def forward(self, x):
		return out

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if use_cuda:
	print('Using GPU:')
else:
	print('Using CPU')


# Instantiating net and specifying optimizer and loss function
net = nav_nw()

if use_cuda:
	net.to(device)

# Specifying the loss criterion and optimizer
criterion = nn.BCELoss(reduction='none')

# Optimizer based on gradient descent with lr = 0.001
# optimizer = optim.SGD(net.parameters(), lr=0.001)
