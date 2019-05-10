from transforms import *
from funcs import *
import random
import os

ri = random.randint

def eosi(showsteps = False):
	try:
		imsize = 120
		mincoreP = 0.35
		maxcoreP = 0.5
		cores = 3
		tmp = np.zeros((imsize,imsize))
		radii = [ri(int(mincoreP*imsize), int(maxcoreP*imsize))for i in xrange(cores)]
		c = [genCircle(i)for i in radii]
		p = [(ri(0,(imsize-c[0].shape[0])-2),ri(0,(imsize-c[0].shape[1])-2))]
		teller = 0
		while len(p) != cores:
			teller += 1
			height = c[len(p)-1].shape[0]
			coor = (ri(0, (imsize - height) - 1), ri(0, (imsize - height) - 1))
			vec = np.linalg.norm(np.array(p[len(p)-1])-coor)
			if 1.5 * max(radii[len(p)-2],radii[len(p)-1]) > random.random() * vec \
				> 1.1 * max(radii[len(p)-2],radii[len(p)-1]):
				p.append(coor)
			if teller >5000:
				return eosi(showteps)

		for i in xrange(cores):
			tmp[p[i][0]:p[i][0] + c[i].shape[0], p[i][1]:p[i][1]+c[i].shape[1]] += c[i]
		dst, src = saveGrid(tmp, 10, 10)
		dst = (dst-dst.mean(0))*.72 + dst.mean(0)
		mask = (tmp>0.1).astype(int)

		try: os.mkdir('../stapeosi')
		except: pass

		if showsteps:misc.imsave('../stapeosi/0.png', misc.imresize(mask,400))
		mask = elastic(mask, 275, ri(16,18))
		if showsteps:misc.imsave('../stapeosi/1.png', misc.imresize(mask,400))
		mask = square2Circle(mask, dst, src)
		if showsteps:misc.imsave('../stapeosi/2.png', misc.imresize(mask,400))
		mask = elastic(mask, 200, ri(16,18))*(1/mask.max())
		if showsteps:misc.imsave('../stapeosi/3.png', misc.imresize(mask,400))

		cell = np.zeros((imsize, imsize,3))
		kleur(cell, mask>0.1, (240,222,239)) #-> cytosol (230,169,222) 240,222,239
		if showsteps:misc.imsave('../stapeosi/5.png', misc.imresize(cell,400))
		kleur(cell, (mask>0.5), (217,43,166)) #-> kern
		if showsteps:misc.imsave('../stapeosi/6.png', misc.imresize(cell,400))
		kleur(cell, mask<0.1, (1,1,1))		   #-> background
		if showsteps:cell[0,0] = 0
		if showsteps:misc.imsave('../stapeosi/7.png', misc.imresize(cell,400))

		#cytoplasm noise
		noise1 = minmax(1-genNoise(imsize,.08, ri(0,1000)),0.5,1)**2
		noise2 = minmax(1-genSparseNoise(imsize, .4, .7, 1), 0.2, 1)
		noise3 = minmax(1-genSparseNoise(imsize, .4, .7, 1), 0.2, 1)
		noise4 = minmax(1-genSparseNoise(imsize, .4, .7, 1), 0.5, 1)

		cell = interpolation(cell, cfield(imsize, (210,37,62)), noise1, (mask>.1)*(mask<.5))   
		if showsteps:misc.imsave('../stapeosi/8.png', misc.imresize(cell,400))
		cell = interpolation(cell, cfield(imsize, (159,32,73)), noise2, (mask>.1)*(mask<.5))
		if showsteps:misc.imsave('../stapeosi/9.png', misc.imresize(cell,400))
		cell = interpolation(cell, cfield(imsize, (151,50,90)), noise2, (mask>.1)*(mask<.5))
		if showsteps:misc.imsave('../stapeosi/10.png', misc.imresize(cell,400))
		cell = interpolation(cell, cfield(imsize, (151,50,90)), noise3, (mask>.1)*(mask<.5))
		if showsteps:misc.imsave('../stapeosi/11.png', misc.imresize(cell,400))
		cell = interpolation(cell, cfield(imsize, (170,89,156)), noise4,(mask>.1)*(mask<.5))
		if showsteps:misc.imsave('../stapeosi/12.png', misc.imresize(cell,400))

		#core
		coreNoise1 = 1 - (getEdge(mask>.5, 7) * genNoise(imsize, 0.08,ri(0,1000)))
		coreNoise2 = genNoise(imsize, 0.3,ri(0,1000))
		coreNoise3 = minmax(1-genSparseNoise(imsize, .2, .8, 1), 0.1, 1)
		coreNoise4 = minmax(1-genSparseNoise(imsize, .3, .8, 1), 0.1, 1)

		cell = interpolation(cell, cfield(imsize, (70,1,142)), coreNoise1, mask>.5)
		if showsteps:misc.imsave('../stapeosi/13.png', misc.imresize(cell,400))
		cell = interpolation(cell, cfield(imsize, (161,0,165)), coreNoise2, mask>.5)
		if showsteps:misc.imsave('../stapeosi/14.png', misc.imresize(cell,400))
		cell = interpolation(cell, cfield(imsize, (179,41,52)), coreNoise3, mask>.5)
		if showsteps:misc.imsave('../stapeosi/15.png', misc.imresize(cell,400))
		cell = interpolation(cell, cfield(imsize, (240,133,213)), coreNoise4, mask>.5)
		if showsteps:misc.imsave('../stapeosi/15.png', misc.imresize(cell,400))

		cell = gausBlur(cell,1)
		cell[0,0] = 0

		if showsteps:misc.imsave('../stapeosi/final.png', misc.imresize(cell,400))
		return cell, mask

	except: return eosi(showsteps)

if __name__ == '__main__':
	for i in xrange(5001,100000):
		a, b = eosi()
		misc.imsave('../dataset/eosi/cell/%i.png'%i, a)
		misc.imsave('../dataset/eosi/mask/%i.png'%i, b)
		if i%500 == 0:
			print i
