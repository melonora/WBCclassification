import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import PiecewiseAffineTransform, warp
from scipy import misc
from random import randint as ri
import random

def elastic(image, alpha, sigma, seed = None):
	# Function that performs the elastic deformation.
	assert len(image.shape)==2
	rs = np.random.RandomState(seed)
	shape = image.shape
	dx = gaussian_filter((rs.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
	dy = gaussian_filter((rs.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
	x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
	indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
	return map_coordinates(image, indices, order=1).reshape(shape)
	
def circlePoints(radius, distance, center):
	# Function that obtains each point for the source and destination grid for the piecewise affine transformation. These grids ensure that a square images is transformed into a round image.
	points = 2*np.pi/(2*np.sin(.5*distance/radius))
	radians = np.linspace(0,2*np.pi,points)
	dstGrid = [(center[0]+(radius*np.sin(i)), center[1]+(radius*np.cos(i))) for i in radians]
	srcGrid = []
	for i in radians:
		dsty = center[0]+(radius*np.sin(i))
		dstx = center[1]+(radius*np.sin(i))
		if .25*np.pi <= i <= .75*np.pi:
			y = radius + center[0]
			x = radius/np.tan(i) + center[1]
		elif .75*np.pi < i <= 1.25*np.pi:
			y = -radius*np.tan(i) + center[0]
			x = -radius + center[1]
		elif 1.25*np.pi < i <= 1.75*np.pi:
			y = -radius + center[0]
			x = -radius/np.tan(i) + center[1]
		else:
			x = radius + center[0]
			y = radius*np.tan(i) + center[1]
		srcGrid.append((y,x))
	
	return srcGrid, dstGrid
	
def saveGrid(image,rows,cols):
	# Function which obtains the source and destination grid for piecewise affine transformatoin and saves it.
	center = np.array([int((image.shape[0]+1)/2-1),int((image.shape[0]+1)/2-1)])
	bins = 10
	radii = np.linspace(0,center[0],bins)
	distance = radii[1]
	totSrc = []
	totDst = []
	for i in radii[1:]:
		src,dst = circlePoints(i,distance,center)
		totSrc.append(src)
		totDst.append(dst)
	
	totSrcF = np.array([item for sublist in totSrc for item in sublist])
	totDstF = np.array([item for sublist in totDst for item in sublist])
	np.save('srcGrid', totSrcF)
	np.save('dstGrid', totDstF)
	return totDstF, totSrcF
	
def square2Circle(image,dst,src):
	# Function applying the piecewise affine transformation
	tform = PiecewiseAffineTransform()
	tform.estimate(dst, src)
	out = warp(image, tform, output_shape=(image.shape[:2]))	
	return out
	
if __name__ == '__main__':
	pass














