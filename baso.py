from transforms import *
from funcs import *
import random
import os

ri = random.randint
# Function used to create synthetic basophil images. Check neut.py for basic outline of the pipeline
def baso(showsteps = False):
	imsize = 120
	tmp = np.zeros((imsize,imsize))
	dst, src = saveGrid(tmp, 10, 10)
	dst = (dst-dst.mean(0))*(ri(6,7)/10.) + dst.mean(0)
	mask = (tmp>0.1).astype(int)
	mask = square2Circle(mask, dst, src)
	mask = elastic(mask, 200, ri(7,12))*(1/mask.max())

	cell = np.zeros((imsize, imsize,3))
	kleur(cell, mask>0.1, (151,84,170))   
	kleur(cell, mask>-0.1, (1,1,1))		   
	

	noise1 = minmax(genSparseNoise(imsize, .3,.8),random.random()/8,1)
	cell = interpolation(cell, cfield(imsize, (173,87,177)), noise1, mask>.1) 
	cell = gausBlur(cell, 1)
	cell = interpolation(cell, cfield(imsize, (1,1,1)), minmax(genNoise(imsize, 0.1),.8))
	
	if not ri(0,3):
		noise5 = minmax(getEdge(mask, 30)*genSparseNoise(imsize, 0.05, 0.7))
		noise6 = minmax(genSparseNoise(imsize, .3,.6) * noise5, 0,.5)
		cell = interpolation(cell, cfield(imsize, (1,1,1)), 1-noise5)
		cell = interpolation(cell, cfield(imsize, (173,87,177)), 1-noise6)
	
	noise2 = minmax(genSparseNoise(imsize, .2,.6)*genNoise(imsize,.1))
	cell = interpolation(cell, cfield(imsize, (151/1.2,84/1.2,170/1.2)), 1-noise2, mask>.1) 
	noise3 = minmax(genSparseNoise(imsize, ri(1,60)/1000.,ri(1,30)/100.)*genNoise(imsize, 0.05),0.2,1) 
	cell = interpolation(cell, cfield(imsize, (85,53,118)), 1-noise3, mask>.1)
	
	noise4 = minmax(getEdge(mask,8) * genSparseNoise(imsize, .2,.6),0,1)
	cell = interpolation(cell, cfield(imsize, (65,43,110)), 1-noise4, mask>.1)

	cell = gausBlur(cell, 1)
	cell[0,0] = 0
	if showsteps: misc.imsave('mooie_cell.png', misc.imresize(cell, 250))
	if showsteps: misc.imsave('mooie_cell2.png', misc.imresize(noise6[:,:,0],250))
	return cell, mask

def multiple(cols, rows, outname='baso.png', imsize=120):
	ls = np.ones((imsize*rows, imsize*cols,3))
	tel = 0
	for y in xrange(rows):
		for x in xrange(cols):
			ls[y*imsize:(y+1)*imsize, x*imsize:(x+1)*imsize] = baso()
			ls[y*imsize,x*imsize] = 1
			tel += 1
			print tel
	ls[0,0] = 0
	misc.imsave(outname, ls)
alles = 2

for i in xrange(100000):
	a,b = baso()
	misc.imsave('../dataset/baso/cell/%i.png'%i, a)
	misc.imsave('../dataset/baso/mask/%i.png'%i, b)
	if i%500 == 0:
		print i
