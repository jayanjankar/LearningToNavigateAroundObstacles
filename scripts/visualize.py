import cv2
import matplotlib.pyplot as plt 
import numpy as np 
import scipy
import os


# img = np.load('10.npy')
# normalizedImg = np.zeros(img.shape)
# img = cv2.normalize(img, normalizedImg, 0, 255, cv2.NORM_MINMAX )
# print(img)
# # cv2.imshow('2', img)
# # cv2.waitKey(0
# plt.imshow(img, cmap='viridis')
# plt.show()

img_dir = os.getcwd() + '/npy_files_low/'
for file in sorted(os.listdir(img_dir)):
	# print(file)
	if file.endswith('.npy'):
		img = np.load(img_dir + '/' + file)
		plt.imshow(img, cmap = 'viridis')
		plt.savefig('imgs_low/' + file + '.png')
		# plt.show()



