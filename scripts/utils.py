## References
"""
1. https://stackoverflow.com/questions/642154/how-to-convert-strings-into-integers-in-python
"""

# Normal python libraries
import numpy as np 
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import glob
import os
from numpy import load
import pickle

# Pytorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
import torch.optim as optim
# from torch.utils.data.sampler import SubsetRandomSampler


def read_npz(directory):

	vel_current = []
	kinect_img = []
	vel_next = []

	for folder in os.listdir(directory):
		npz_files = []
		for file in sorted(os.listdir(directory + folder )):
			filename, _ = os.path.splitext(file)
			npz_files.append(filename)	

		file_seq = [int(i) for i in npz_files]
		file_seq.sort()

		for i in range(0, len(file_seq)):

			## Extracting current velocity and image data
			current_data = load(directory + folder + '/' +
			 str(file_seq[i]) + '.npz')

			# arr_1 corresponds to velocity, arr_0 corresponds to images
			titles = current_data.files

			# Saving velocity and images for current frame
			vel_current.append(current_data[titles[0]])
			kinect_img.append(current_data[titles[1]])


			if i == len(file_seq) - 1:
				vel_next.append([0,0,0])
				break

			## Extracting next velocity and image data. Only velocity is used.
			next_data = load(directory + folder + '/' + 
				str(file_seq[i+1]) + '.npz')

			# Saving velocities from next frame
			vel_next.append(next_data[titles[0]])


	return vel_current, kinect_img, vel_next


data_directory = 'data/'

pickles = 0
for file in os.listdir(os.getcwd() + '/pickles/'):
	pickles+=1

if pickles < 3:

	vel_c, img_c, vel_n = read_npz(data_directory)

	with open(os.getcwd() + '/pickles/' 'vel_c.pickle', 'wb') as handle :
		pickle.dump(vel_c, handle, protocol = pickle.HIGHEST_PROTOCOL)

	with open(os.getcwd() + '/pickles/' 'img_c.pickle', 'wb') as handle :
		pickle.dump(img_c, handle, protocol = pickle.HIGHEST_PROTOCOL)

	with open(os.getcwd() + '/pickles/' 'vel_n.pickle', 'wb') as handle :
		pickle.dump(vel_n, handle, protocol = pickle.HIGHEST_PROTOCOL)
else:

	print('Pickles already exist')
	with open(os.getcwd() + '/pickles/' 'vel_c.pickle', 'rb') as handle:
		vel_c = pickle.load(handle)	
	with open(os.getcwd() + '/pickles/' 'img_c.pickle', 'rb') as handle:
		img_c = pickle.load(handle)	
	with open(os.getcwd() + '/pickles/' 'vel_n.pickle', 'rb') as handle:
		vel_n = pickle.load(handle)	


print(len(vel_c))
print(len(vel_n))
print(len(img_c))