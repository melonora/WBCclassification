from scipy import misc, ndimage
from opensimplex import OpenSimplex
import numpy as np
from random import randint as ri
from torch import load as tl
import random
import torchvision as tv
import torch
from PIL import Image
import math


def genCircle(n):
	# Function to create the white circles that represent the cores.
	face = np.ones((n,n))
	lx, ly = face.shape
	X, Y = np.ogrid[0:lx, 0:ly]
	mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
	face[mask] = 0
	return face

def gausBlur(img, howmuch=1):
	# Gaussian blur used to blur the synthetic cell image.
	try:
		for i in xrange(img.shape[2]):
			img[:,:, i] = gausBlur(img[:,:,i], howmuch)
	except: img = ndimage.gaussian_filter(img, howmuch)
	return img

def rgb2gray(rgb):
	# Function to convert an RGB image in numpy to a grayscale image
	img = np.zeros((rgb.shape[0],rgb.shape[1],3))
	res = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
	for i in xrange(3):
		img[:,:,i] = res
	return img

def genNoise(pixels, zoom=1, seed=-1):
	# Function to create a noise map.
	if seed == -1: seed = random.randint(0, 10e6)
	gen = OpenSimplex(seed=seed)
	pix = np.linspace(0, pixels*zoom, pixels)
	noi = np.array([[gen.noise2d(x,y)for x in pix] for y in pix])
	noi = noi.reshape((noi.shape[0], noi.shape[1],1))
	return (noi-noi.min())/(-noi.min()+noi.max())

def genSparseNoise(pixels, zoom, threshold, blur=0):
	# Same as genNoise, but a sparse version.
	tmp = genNoise(pixels, zoom)
	tmp[tmp<=threshold] = 0
	if blur:
		tmp = gausBlur(tmp, blur)
	return minmax(tmp,0,1)

def minmax(noise, min=0, max=1):
	# Thresholding the noise map.
	return noise*(max-min) + min

def kleur(img, mask, color):
	# Used to apply a basic color to either background, cytoplasm or cell core.
	color = np.array(color, dtype='float32')
	if color.max()>1: color /= 255.
	img[mask>0.5] = color
	return img

def cfield(imsize, color):
	# Creating a color field that alongside a mask can be used to color the synthetic image.
	color = np.array(color, dtype='float32')
	if color.max()>1: color /= 255.
	return np.ones((imsize, imsize, 3)) * color

def getEdge(mask, n=10):
	# Function used to obtain the edges of a mask, for example when supplying a mask of the cell a mask with the edge of the cell is returned. N determines the width of that edge. 
	n = n - (n % 2)
	m1 = mask.copy() * 1.0
	m2 = misc.imresize(m1, (mask.shape[0]-(n*2), mask.shape[1]-(n*2)))
	m1[n:n+m2.shape[0],n:n+m2.shape[1]][m2>0.5] = 0
	m1[m1>.01] = 1
	return m1[:,:,np.newaxis]

def radialNoise(imgSize):
	# Function used to create a radial (based on distance from center of the image) noise map. 
	radNoise = np.array([[(abs(y-imgSize/2)**2+abs(x-imgSize/2)**2)**.5 for x in xrange(imgSize)] for y in xrange(imgSize)])
	radNoise = (radNoise-radNoise.min())/(-radNoise.min()+radNoise.max())
	return radNoise

def interpolation(img1, img2, imap, mask=None):
	# Function used to apply noise to the synthetic image. Img1 is the synthetic image, img2 is the image containing the noise color, imap is the noisemap and mask is the mask to determine which area to apply the noise to.
	if type(mask)==type(None): mask = np.ones((img1.shape[0], img1.shape[1]))
	try: mask.shape[1]
	except: mask = minmax(mask.copy().astype(int),0,1)
	if len(mask.shape)==2: mask = mask[:,:,np.newaxis]
	return (img1*imap + img2*(1-imap))*mask +  img1*(1-mask)

def augment(img):
	# Function used to apply different forms of augmentation prior to feeding synthetic training images to the CNN. This function is adjusted as required for the experiment.
	colors = [(249/255., 179/255., 171/255.), (248/255., 211/255., 205/255.),\
			(207/255., 221/255., 227/255.), (246/255., 215/255., 182/255.)]
	alfa = ri(40,100)/100.
	out = interpolation(img, cfield(img.shape[0], colors[ri(0,len(colors)-1)]), np.ones((img.shape[0], img.shape[1],1))*alfa)

	out = Image.fromarray(np.uint8(img*255))

	sa = ri(0,20)/100.
	co = ri(0,20)/100.
	br = ri(71,81)/100.
	hu = random.random()-.5
	out = tv.transforms.ColorJitter(saturation=sa,brightness=br,contrast=co,hue=hu)(out) #saturation=sa,brightness=br,contrast=co,hue=hu
	out = np.array(out.getdata()).reshape((img.shape[0],img.shape[1],3))/255.
	return out

def mask2label(mask):
	# Convert a grayscale ground truth segmentation to an RGB tensor.
	assert len(mask.shape)==2
	if mask.max() > 1: mask = mask/255.
	new = np.zeros((3,120,120))
	new[0] = (mask < .1).astype(int)
	new[1] = ((mask > 0)*(mask < .8)).astype(int)
	new[2] = (mask > .8).astype(int)
	return torch.Tensor(new).float()

def misc2torch(img):
	# Convert a numpy array to a torch tensor
	new = np.zeros((3,img.shape[0],img.shape[1]))
	for i in xrange(3):
		new[i] = img[:,:,i]
	return torch.Tensor(new).float()

def torch2misc(img):
	# Convert a torch tensor to a numpy array which is seen as an image by scipy misc.
	if len(img.size())==4:
		img = img.view(img.size(1),img.size(2),img.size(3))
	try: img = img.data
	except:pass
	try: img = img.numpy()
	except:pass
	new = np.zeros((img.shape[1],img.shape[2],3))
	for i in xrange(3):
		new[:,:,i] = img[i]
	return new

if __name__ == "__main__":
	import time
	img = misc.imread('0.png')/255.
	while 1:
		misc.imsave('a1.png', augment(img.copy())[:,:,0])
		time.sleep(1)
