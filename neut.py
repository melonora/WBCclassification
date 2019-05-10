from transforms import *
from funcs import *
import random
import os

ri = random.randint
#Function used to create synthetic neutrophil images and corresponding ground truth segmentation.
def neutro(showsteps = False):
  try:
  	# Basic parameters image size to be created, min and max diameter of circles that represent core and amount of circles to represent core.
	imsize = 120
	mincoreP = 0.35
	maxcoreP = 0.40
	cores = 5
	
	# Initialize temporary image, list with the various diameters of the circles to be created, actual list c containing the circles and a list p which contains the first position to put the first circle in the list at.
	tmp = np.zeros((imsize,imsize))
	radii = [ri(int(mincoreP*imsize), int(maxcoreP*imsize))for i in xrange(cores)]
	c = [genCircle(i)for i in radii]
	p = [(ri(0,(imsize-c[0].shape[0])-2),ri(0,(imsize-c[0].shape[1])-2))]
	
	# Loop that obtains the positions to put the remainder of the circle at. The position can not be too far but also not to close to the position of the previous circle
	teller = 0
	while len(p) != cores :
		teller += 1
		height = c[len(p)-1].shape[0]
		coor = (ri(0, (imsize - height) - 1), ri(0, (imsize - height) - 1))
		vec = np.linalg.norm(np.array(p[len(p)-1])-coor)
		if 1.4 * max(radii[len(p)-2],radii[len(p)-1]) > random.random() * vec \
				> 1.1 * max(radii[len(p)-2],radii[len(p)-1]):
			p.append(coor)
		if teller >5000:
			return neutro(showsteps)
			
	# Put the circles at the positions in the temporary image.
	for i in xrange(cores):
		tmp[p[i][0]:p[i][0] + c[i].shape[0], p[i][1]:p[i][1]+c[i].shape[1]] += c[i]
	
	# Obtaining the source and destination grid for the piecewise affine transformation.	
	dst, src = saveGrid(tmp, 10, 10)
	dst = (dst-dst.mean(0))*.6 + dst.mean(0)
	mask = (tmp>0.1).astype(int)
	
	
	try: os.mkdir('../stapneut')
	# Elastic deformation of the image with white circles, after which a piecewise affine transformation is performed to trainsform the square image in a round image, after which another elastic deformation is performed to obtain the final ground truth image of the synthetic image to be created. 
	except: pass
	if showsteps:misc.imsave('../stapneut/0.png', misc.imresize(mask,400))
	mask = elastic(mask, 275, ri(9,12))
	if showsteps:misc.imsave('../stapneut/1.png', misc.imresize(mask,400))
	mask = square2Circle(mask, dst, src)
	if showsteps:misc.imsave('../stapneut/2.png', misc.imresize(mask,400))
	mask = elastic(mask, 200, 10)*(1/mask.max())
	if showsteps:misc.imsave('../stapneut/3.png', misc.imresize(mask,400))
	
	# Initiatize RGB image and applying the basic color of each component of the image.
	cell = np.zeros((imsize, imsize,3))
	kleur(cell, mask>0.1, (243,187,220))   #-> cytoplasm
	if showsteps:misc.imsave('../stapneut/5.png', misc.imresize(cell,400))
	kleur(cell, (mask>0.5), (154,102,202)) #-> kern
	if showsteps:misc.imsave('../stapneut/6.png', misc.imresize(cell,400))
	kleur(cell, mask<0.1, (1,1,1))		   #-> background
	if showsteps:cell[0,0] = 0
	if showsteps:misc.imsave('../stapneut/7.png', misc.imresize(cell,400))

	# Creating noisemaps for cytoplasm
	ruis1 = minmax(genNoise(imsize,0.05, ri(0,1000))**2, 0.5, 1)
	ruis2 = genNoise(imsize, 0.1)*genNoise(imsize, 0.1)
	ruis2 = ruis2 * 1/ruis2.max()
	ruis2 = minmax(1 - (getEdge(mask>.1, 8) * ruis2),0.,1)
	ruis3 = minmax(1-genSparseNoise(imsize, .4, .7, 1), 0.2, 1)
	
	# Applying different kinds of cytoplasm noise to the rgb synthetic image.
	cell = interpolation(cell, cfield(imsize, (177,161,207)), ruis2, (mask>.1)*(mask<.5))
	if showsteps:misc.imsave('../stapneut/8.png', misc.imresize(cell,400))
	cell = interpolation(cell, cfield(imsize, (92,61,121)), ruis3, (mask>.1)*(mask<.5))
	if showsteps:misc.imsave('../stapneut/9.png', misc.imresize(cell,400))
	cell = interpolation(cell, cfield(imsize, (1,1,1)), ruis1, (mask>.1)*(mask<.5))
	if showsteps:misc.imsave('../stapneut/10.png', misc.imresize(cell,400))

	# Create noise maps for the cell core.
	kernnoise1 = 1 - (getEdge(mask>.5, 7) * genNoise(imsize, 0.08,ri(0,1000)))
	kernnoise2 = genNoise(imsize, 0.3,ri(0,1000))
	kernnoise3 = genNoise(imsize, 0.1,ri(0,1000))**2
	
	# Applying core noise to the synthetic cell image
	cell = interpolation(cell, cfield(imsize, (233,187,220)), kernnoise2, mask>.7)
	if showsteps:misc.imsave('../stapneut/11.png', misc.imresize(cell,400))
	cell = interpolation(cell, cfield(imsize, (139,95,185)), kernnoise3, mask>.7)
	if showsteps:misc.imsave('../stapneut/12.png', misc.imresize(cell,400))
	cell = interpolation(cell, cfield(imsize, (0.41,0.25,0.48)), kernnoise1)
	if showsteps:misc.imsave('../stapneut/13.png', misc.imresize(cell,400))
	
	# Performing a gaussian blur to the synthetic cell image
	cell = gausBlur(cell, 1)
	
	#When saving an image one pixel must be set to white in order for colors to show properly
	if showsteps:cell[0,0] = 0
	if showsteps:misc.imsave('../stapneut/14.png', misc.imresize(cell,400))
	cell[0,0] = 0
	if showsteps: misc.imsave('../stapneut/final.png', misc.imresize(cell,400))
	return cell, mask
  except: return neutro(showsteps)


def multiple(cols, rows, outname='Hele_boel.png', imsize=120):
	#Function used to create an image of multiple synthetic cell images.
	ls = np.ones((imsize*rows, imsize*cols,3))
	ms = np.ones((imsize*rows, imsize*cols))
	tel = 0
	for y in xrange(rows):
		for x in xrange(cols):
			a,b = neutro()
			ls[y*imsize:(y+1)*imsize, x*imsize:(x+1)*imsize] = a
			ls[y*imsize, x*imsize] = 1
			ms[y*imsize:(y+1)*imsize, x*imsize:(x+1)*imsize] = b
			tel += 1
			print tel
	ms[ms>.8] = .8
	ms[ms<.1] = 1
	ms[ms==.8] = 0
	ms[(ms!=1)*(ms!=0)] = .5
	ls[0,0] = 0
	ms[0,0] = 0
	misc.imsave(outname, ls)
	misc.imsave(outname.replace('.png','_mask.png'), ms)
# For loop to create all synthetic neutrophil images.
for i in xrange(100000):
	a,b = neutro()
	misc.imsave('../dataset/neut/cell/%i.png'%i, a)
	misc.imsave('../dataset/neut/mask/%i.png'%i, b)
	if i%500 == 0:
		print i
